import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple
import types
from PIL import Image
from omegaconf import OmegaConf
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
import sys
import json
from edit_train_dataset_smart import OmniGen2TrainDataset_smart, OmniGen2Collator
sys.path.append('..')
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_image.qw_pipelines.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwen_image.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from qwen_image.qw_pipelines.pipeline_output import QwenImagePipelineOutput
import re
import math
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 512 * 512

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

def extract_structured_content(text):
    """
    Extract content from <info>, <think>, and <answer> tags
    """
    patterns = {
        'info': r'<info>(.*?)</info>',
        'think': r'<think>(.*?)</think>', 
        'answer': r'<answer>(.*?)</answer>',
        'object':  r'<object>(.*?)</object>'
    }
    
    result = {}
    for tag, pattern in patterns.items():
        matches = re.findall(pattern, text, re.DOTALL)
        result[tag] = [match.strip() for match in matches] if matches else []
    
    return result

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

        template =  "<|im_start|>system\nYou are tasked with analyzing an image to generate an exhaustive and detailed description. Your goal is to extract and describe all possible information from the image, including but not limited to objects, numbers, text, and the relationships between these elements. The description should be as fine and detailed as possible, capturing every nuance. After generating the detailed description, you need to analyze the instruction and provide step-by-step simple reasoning for the given instruction, please avoid repeating the same points. Finally, provide concise answer to what will happen with the given instruction and provide the specific instruction to modify the image, and list the properties of the modified image, including number, color and position of the object. The description, reasoning process, answer and property are enclosed within <info> </info>, <think> </think>, <answer> </answer> and <onject> </object> tags, respectively, i.e., <info> image description here </info> <think> reasoning process here </think> <answer> answer here </answer> <object> list the perperties of modified image here </object>. <|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]
        #print(txt)
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
        #print(output_text)
        content = extract_structured_content(output_text[0])
        template = self.prompt_template_encode
        #new_text = "<|im_start|>" + content['answer'][0] + "|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        #txt = [new_text.format(base_img_prompt)]
        edit_txt = prompt[0]
        if len(content['object'])==0:
            edit_txt = prompt[0]
        else:
            edit_txt = content['object'][0]
        

        txt = [template.format(base_img_prompt + e) for e in [edit_txt]]
        #print(txt)
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

        return prompt_embeds, encoder_attention_mask, content


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.encode_prompt
def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 1024,
):
    r"""

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        image (`torch.Tensor`, *optional*):
            image to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
    """
    device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask, content = self._get_qwen_prompt_embeds(prompt, image, device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

    return prompt_embeds, prompt_embeds_mask, content

def generate(
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
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
                Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
                equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
                enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
                encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
                lower image quality.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to None):
                A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
                where the guidance scale is applied during inference through noise prediction rescaling, guidance
                distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
                scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
                that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
                parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
                ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
                please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
                enable classifier-free guidance computations).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        image_size = image[-1].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(512 * 512, image_size[0] / image_size[1])
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
            target_vae_images = []
            for img in image:
                image_width, image_height = img.size
                condition_width, condition_height, _ = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height, _ = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask, content = self.encode_prompt(
            image=condition_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size

        # 6. Denoising loop
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)
        #input_save_dict["encoder_hidden_states"] = 
                
        return latent_model_input, prompt_embeds, prompt_embeds_mask, content


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ma-user/work/yanli/test_yl/Qwen-Image-Edit-2509",
        #required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default='/home/ma-user/work/yanli/test_yl/Qwen-Image-Edit-2509/transformer',
        #required=True,
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
    pipeline.generate = types.MethodType(generate, pipeline)
    pipeline.encode_prompt = types.MethodType(encode_prompt, pipeline)
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

    results = pipeline.generate(
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
    train_dataset = OmniGen2TrainDataset_smart(
            "/home/ma-user/work/yanli/test_yl/OmniGen2/data_configs/train/example/ic/ic_single_edit.yml",
            use_chat_template=False,
            prompt_dropout_prob=0.0,
            ref_img_dropout_prob=0.0,
            max_input_pixels=1024 * 1024,
            max_output_pixels=1024 * 1024,
            max_side_length=2048,
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=4,
        worker_init_fn=None,
        drop_last=True,
        collate_fn=OmniGen2Collator(tokenizer=pipeline.tokenizer, max_token_len=512)
    )

    for step, batch in enumerate(train_dataloader):
            # Number of bins, for loss recording
        #data_item = {}
        input_images = batch['input_images']
        output_image = batch['output_image']
        input_image_path = ["/home/ma-user/work/yanli/test_yl/editworld_dataset/visual/"+f'source_{step}.png'] #batch['input_images_path']
        batch_instruction = batch['instruction'][0]
        output_paths = ["/home/ma-user/work/yanli/test_yl/editworld_dataset/visual/"+f'target_{step}.png'] #batch['output_image_path']
        #input_images = preprocess(args.input_image_path)
        input_images[0][0].save(input_image_path[0])
        output_image[0].save(output_paths[0])
        print(input_images[0][0].size)
        # Generate and save image
        input_image_id = os.path.basename(input_image_path[0])[:-4]
        root_path = "/home/ma-user/work/yanli/test_yl/editworld_dataset/input_data_rl_512"
        #save_path = os.path.join('/home/ma-user/work/yanli/test_yl/Datasets/input_data_rl', )
        encoder_hidden_path = os.path.join(root_path, input_image_id + '_prompt_embed.pt')
        if os.path.exists(encoder_hidden_path):
            print('pass')
            continue
        encoder_hidden_mask_path = os.path.join(root_path, input_image_id + '_prompt_embed_mask.pt')
        input_latent_path = os.path.join(root_path, input_image_id + "_input_latent.pt")
        with torch.no_grad():
            latent_model_input, prompt_embeds, prompt_embeds_mask, content = run(args, accelerator, pipeline, batch_instruction , args.negative_prompt, input_images[0])
        torch.save(latent_model_input, input_latent_path)
        torch.save(prompt_embeds, encoder_hidden_path)
        torch.save(prompt_embeds_mask, encoder_hidden_mask_path)
        data = {
            "target_image": f"images/{os.path.basename(output_paths[0])}",
            "source_image": f"images/{os.path.basename(input_image_path[0])}", 
            "instruction": batch_instruction,
            "info": content['info'],
            "think": content['think'],
            "answer": content["answer"],
            "object": content["object"],
            "prompt_embeds":  input_image_id + '_prompt_embed.pt', 
            "prompt_embeds_mask": input_image_id + '_prompt_embed_mask.pt',
            "input_latent_path":   input_image_id + "_input_latent.pt",
        }
        
        with open("/home/ma-user/work/yanli/test_yl/editworld_dataset/" + f'test_512_{step//10000}.jsonl', 'a+', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

        torch.cuda.empty_cache()
        #print(content)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)



