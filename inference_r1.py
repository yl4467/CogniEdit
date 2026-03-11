import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple
import types
from PIL import Image
from tqdm import tqdm
import logging
import json
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_image.qw_pipelines.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwen_image.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from typing import Any, Callable, Dict, List, Optional, Union
from safetensors.torch import load_file
import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
#from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from qwen_image.models.transformers.transformer_qwen_dynamic import QwenImageTransformer2DModel
#from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from qwen_image.qw_pipelines.pipeline_output import QwenImagePipelineOutput
from safetensors import safe_open
from peft import LoraConfig
import re
CONDITION_IMAGE_SIZE = 384 * 384
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

def extract_structured_content(text):
    """
    Extract content from <info>, <think>, and <answer> tags
    """
    patterns = {
        'info': r'<info>(.*?)</info>',
        'think': r'<think>(.*?)</think>', 
        'answer': r'<answer>(.*?)</answer>',
        'object': r'<object>(.*?)</object>'
    }
    
    result = {}
    for tag, pattern in patterns.items():
        matches = re.findall(pattern, text, re.DOTALL)
        result[tag] = [match.strip() for match in matches] if matches else []
    
    return result

def load_large_safetensors(model_dir):
    """使用safe_open处理大型分片模型"""
    state_dict = {}
    safetensors_files = sorted([f for f in os.listdir(model_dir) 
                               if f.endswith(".safetensors")])
    
    for filename in safetensors_files:
        file_path = os.path.join(model_dir, filename)
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # 按需加载每个张量
                state_dict[key] = f.get_tensor(key)
    
    return state_dict


def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = "<|im_start|>system\nYou are tasked with analyzing an image to generate an exhaustive and detailed description. Your goal is to extract and describe all possible information from the image, including but not limited to objects, numbers, text, and the relationships between these elements. The description should be as fine and detailed as possible, capturing every nuance. After generating the detailed description, you need to analyze the instruction and provide step-by-step simple reasoning for the given instruction, please avoid repeating the same points. Finally, provide concise answer to what will happen with the given instruction and provide the specific instruction to modify the image, and list the properties of the modified image, including number, color and position of the object. The description, reasoning process, answer and property are enclosed within <info> </info>, <think> </think>, <answer> </answer> and <object> </object> tags, respectively, i.e., <info> image description here </info> <think> reasoning process here </think> <answer> answer here </answer> <object> list the perperties of modified image here </object>. <|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]
        print(txt)
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        output_ids = self.text_encoder.generate(**model_inputs, max_new_tokens=1024)
        #generated_ids_trimmed = [
        #    out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, output_ids)
        #]
        output_ids = output_ids[:, model_inputs.input_ids.shape[1]:]
       
        #print(generated_ids_trimmed.shape)
        output_text = self.processor.batch_decode(
            output_ids, #skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        #with open('')
        content = extract_structured_content(output_text[0])
        template = self.prompt_template_encode
        #new_text = "<|im_start|>" + content['answer'][0] + "|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        #txt = [new_text.format(base_img_prompt)]
        if len(content['object']) == 0:
            content['object'].append(prompt[0])
        txt = [template.format(base_img_prompt + e) for e in [content['object'][0]]]
        print(txt)
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        print(model_inputs.attention_mask.shape, model_inputs.input_ids.shape, output_ids.shape)
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            #input_ids=generated_ids_trimmed,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

@torch.no_grad()
def __call__(
    self,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    
    image_size = image[-1].size if isinstance(image, list) else image.size
    calculated_width, calculated_height = calculate_dimensions(512 * 512, image_size[0] / image_size[1])
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # 3. Preprocess image
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        if not isinstance(image, list):
            image = [image]
        condition_image_sizes = []
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in image:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(
                CONDITION_IMAGE_SIZE, image_width / image_height
            )
            condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
           
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=condition_images,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    
    return


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=5.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of the CFG range."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="A dog running in the park",
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to input image(s)."
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output.png",
        help="Path to save output image."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    return parser.parse_args()

def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> QwenImageEditPlusPipeline:
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    pipeline.transformer = QwenImageTransformer2DModel.from_pretrained(
        args.transformer_path,
        #subfolder="transformer",
        torch_dtype=weight_dtype,
        
    )
    
    pipeline.transformer.locate_pos = pipeline.transformer.locate_pos.to_empty(device=pipeline.transformer.device)
    
    pipeline.transformer.set_dynamic_tokens(n_dc_layers=10, token_length=64)
    
    target_modules = ["to_k", "to_v", "to_out.0"]
    lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
    pipeline.transformer.add_adapter(lora_config)
    pipeline.transformer.to_empty(device=accelerator.device)
    state_dict = load_large_safetensors(args.transformer_path)
    pipeline.transformer.load_state_dict(state_dict, strict=True)
    
    pipeline.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/home/ma-user/work/yanli/test_yl/PeBR_R1/PeBR_R1_7B",
        torch_dtype=weight_dtype,
    )
    
    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(accelerator.device)
    pipeline._get_qwen_prompt_embeds = types.MethodType(_get_qwen_prompt_embeds, pipeline)
    #pipeline.__call__ = types.MethodType(__call__, pipeline)
    return pipeline

def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]
            
    return input_images

def run(args: argparse.Namespace, 
        accelerator: Accelerator, 
        pipeline: QwenImageEditPlusPipeline,
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        image=input_images,
        #width=args.width,
        #height=args.height,
        num_inference_steps=args.num_inference_step,
        #max_sequence_length=1024,
        guidance_scale=1.0,
        negative_prompt=" ",
        true_cfg_scale=4.0,
        #text_guidance_scale=args.text_guidance_scale,
        #image_guidance_scale=args.image_guidance_scale,
        #cfg_range=(args.cfg_range_start, args.cfg_range_end),
        #negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        #output_type="pil",
    )
    return results

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)

def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    input_images = preprocess(args.input_image_path)
    args.instruction = args.instruction #item["instruction"]
    results = run(args, accelerator, pipeline, args.instruction, args.negative_prompt, input_images)
    
    if results.images is not None:
        vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
        output_image = create_collage(vis_images)
        output_path = os.path.join(args.output_image_path, os.path.basename(args.input_image_path))
        output_image.save(output_path)
        print(f"Image saved to {args.output_image_path}")
            

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)