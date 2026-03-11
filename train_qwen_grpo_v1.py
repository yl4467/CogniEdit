import dotenv

dotenv.load_dotenv(override=True)
from safetensors.torch import save_file
import time
import inspect
from copy import deepcopy
import argparse
import logging
import math
import os
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm
#os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
#from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler  
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms.functional import crop, to_pil_image, to_tensor
from PIL import Image, ImageOps 
from PIL import ImageDraw, ImageFont
from einops import repeat, rearrange
from torch.utils import checkpoint
import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.cuda.amp import autocast as autocast
import transformers
from transformers import Qwen2Tokenizer
from transformers import AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration as TextEncoder
import torch.distributed as dist
from transformers import Qwen2VLProcessor
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models import AutoencoderKLQwenImage
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from vllm_utils.vllm_request import evaluate_batch
from peft import LoraConfig

from omnigen2.training_utils import EMAModel
from omnigen2.utils.logging_utils import TqdmToLogger
from qwen_image.transport import create_transport
#from omnigen2.dataset.omnigen2_train_dataset import OmniGen2TrainDataset, OmniGen2Collator
#from qwen_image.dataset.omnigen2_train_multi_turn_dataset import OmniGen2TrainMultiDataset, OmniGen2Collator, OmniGen2TestMultiDataset, calculate_dimensions
from qwen_image.dataset.qwen_train_rl_dataset import Qwen2TrainRlDataset, OmniGen2Collator, calculate_dimensions
#from omnigen2.dataset.omnigen2_test_dataset import OmniGen2TestDataset
#from qwen_image.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from qwen_image.models.transformers.transformer_qwen_dynamic import QwenImageTransformer2DModel
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from qwen_image.utils.loss import dense_score
from diffusers.image_processor import VaeImageProcessor
import re
from checkpoints import save_checkpoint, save_lora_checkpoint
logger = get_logger(__name__)


template = (
            "You are presented with an edited image (the first image) and its original image (the second image), and the associated text caption of the edited image. Your task is to analyze the edited image across multiple dimensions in relation to the caption and its original image. Specifically:\n\n"
            "1. Extract key word related to: subject, object, color, number, lighting, style and activities in the caption based on how well it is visually represented in the edited image.\n"
            #Assign a numerical score to every key word using the format:\n"
            #"   Word-wise Scores: [[\"key word1\", score1], [\"key word2\", score2], ..., [\"key wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
            "   - A higher score indicates that the word is less well represented in the image.\n"
            "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
            "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms of content.\n"
            "- Coherence Score: How logically consistent the image is (object location error, absence of visual glitches, object distortions, object is missing or extra, color or quantity error etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n"
            "- Consistency Score: How well the edited image consistent to its original image.\n\n"
            "Output your evaluation using the format below and also provide the reason why you give this score:\n\n"
            "---\n\n"
            "Word-wise Scores: [[\"key word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n"
            "Consistency Score (1-5): W\n\n"
            "Output the basis and reasons for the scores given above \n"
            "Your task is provided as follows:\nText Caption: [{prompt}]"
            )
def extract_normalized_rewards(sample_list):
    pattern = r"(\w+) Score \(1-5\):\s*([0-5](?:\.\d+)?)"

    all_scores = []
    for response in sample_list:
        matches = re.findall(pattern, response)
        scores = {key: float(value) for key, value in matches}
        if 'Coherence' in scores:
            del scores['Coherence']
        all_scores.append(scores)

    if not all_scores:
        return []

    keys = set()
    for s in all_scores:
        keys.update(s.keys())
    keys = sorted(keys) 

    dim_scores_raw = {k: [s[k] for s in all_scores if k in s] for k in keys}
    dim_means = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in dim_scores_raw.items()}


    alignment_scores = []
    style_scores = []
    consistency_scores = []
    log_alignment_scores = []
    log_style_scores = []

    for s in all_scores:
        alignment_score = s.get("Alignment", dim_means['Alignment'])
        style_score = s.get("Style", dim_means['Style'])
        consis_score = s.get("Consistency", dim_means['Consistency'])
        alignment_scores.append(torch.tensor(alignment_score, device="cuda").unsqueeze(0))
        style_scores.append(torch.tensor(style_score, device="cuda").unsqueeze(0))
        consistency_scores.append(torch.tensor(consis_score, device="cuda").unsqueeze(0))
        log_alignment_scores.append(alignment_score)
        log_style_scores.append(style_score)

    dim_array = {
        'Alignment': log_alignment_scores, 
        'Style': log_style_scores,
        'Consis': consistency_scores
    }

    return alignment_scores, style_scores, consistency_scores, dim_array
    
def parse_args(root_path) -> OmegaConf:
    parser = argparse.ArgumentParser(description="OmniGen2 training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data path.",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        help="Whether to use group sampling for multiple generations per prompt.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of generations per prompt when using group sampling.",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    #output_dir = os.path.join(root_path, 'experiments', conf.name)
    output_dir = "/home/ma-user/work/yanli/qwen_output_grpo_dense_dynamic_test_n/"
    conf.root_dir = root_path
    conf.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'output', exist_ok=True)
    conf.config_file = args.config

    # Override config with command line arguments
    if args.global_batch_size is not None:
        conf.train.global_batch_size = args.global_batch_size
    
    if args.data_path is not None:
        conf.data.data_path = args.data_path
    return conf

def setup_logging(args: OmegaConf, accelerator: Accelerator) -> None:
    """
    Set up logging configuration for training.
    
    Args:
        accelerator: Accelerator instance
        args: Configuration object
        logging_dir: Directory for log files
    """

    logging_dir = Path(args.output_dir, "logs")
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set verbosity for different processes
    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    transformers.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)


def log_model_info(name: str, model: torch.nn.Module):
    """Logs parameter counts for a given model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"--- {name} ---")
    logger.info(model)
    logger.info(f"Total parameters (M): {total_params / 1e6:.2f}")
    logger.info(f"Trainable parameters (M): {trainable_params / 1e6:.2f}")


def log_time_distribution(transport, device, args):
    """Samples time steps from transport and plots their distribution."""
    with torch.no_grad():
        dummy_tensor = torch.randn((64, 16, int(math.sqrt(args.data.max_output_pixels) / 8), int(math.sqrt(args.data.max_output_pixels) / 8)), device=device)
        ts = torch.cat([transport.sample(dummy_tensor, AcceleratorState().process_index, AcceleratorState().num_processes)[0] for _ in range(1000)], dim=0)
    
    ts_np = ts.cpu().numpy()
    percentile_70 = np.percentile(ts_np, 70)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ts_np, bins=50, edgecolor='black', alpha=0.7, label="Time Step Distribution")
    plt.axvline(percentile_70, color='red', linestyle='dashed', linewidth=2, label=f'70th Percentile = {percentile_70:.2f}')
    plt.title('Distribution of Sampled Time Steps (t)')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = Path(args.output_dir) / 't_distribution.png'
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Time step distribution plot saved to {save_path}")

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def generate_test_visualizations(model, test_dataset, processor, text_tokenizer, text_encoder, vae, transport, freqs_cis, weight_dtype, args, accelerator, global_step, process_indix):
    """Generate visualizations for test cases using proper inference pipeline."""
    if test_dataset is None:
        return
        
    # Import pipeline components
    #from omnigen2.pipelines.omnigen2.pipeline_omnigen2_modify import OmniGen2Pipeline
    from qwen_image.qw_pipelines.pipeline_qwenimage_edit import QwenImageEditPipeline
    #from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler  
    #model.eval()
    with torch.no_grad():
        # Create scheduler for inference
        #scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler = FlowMatchEulerDiscreteScheduler(dynamic_time_shift=True)
        # Create pipeline
        pipeline = QwenImageEditPipeline(
            transformer=model,
            vae=vae,
            scheduler=scheduler,
            mllm=text_encoder,
            processor=processor
        )
        pipeline = pipeline.to(accelerator.device)
        # Rest of the function remains the same...
        # Sample a few test cases
        num_test_samples = 1 #min(args.val.get('num_test_visualization_samples', 1), len(test_dataset))
        test_indices = torch.randperm(len(test_dataset))[:num_test_samples]
        print(test_indices)
        for i, idx in enumerate(test_indices):
            test_item = test_dataset[int(idx)]
            
            # Prepare inputs for pipeline
            prompt = test_item['instruction']
            input_images = test_item['input_images'] if test_item['input_images'] is not None else None
            #print(prompt)
            #print(input_images)
            # Set target size
            target_width, target_height = test_item['target_img_size']
            generator = torch.Generator(device=accelerator.device).manual_seed(0)
            # Generate image using pipeline
            result = pipeline(
                prompt=prompt,
                image=input_images,
                height=target_height,
                width=target_width,
                max_sequence_length=1024,
                num_inference_steps=args.val.get('num_inference_steps', 50),
                #text_guidance_scale=args.val.get('text_guidance_scale', 5.0),
                #image_guidance_scale=args.val.get('image_guidance_scale', 2.0),
                #cfg_range=(0, 1.0),
                guidance_scale=1.0,
                true_cfg_scale=4.0,
                generator=generator,
                negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
                num_images_per_prompt = 1,
                #max_pixels=args.data.get('max_output_pixels', 1024 * 1024),
                #max_input_image_side_length=args.data.get('max_side_length', 2048),
                output_type="pil",  # Return as tensor for further processing
            )
            
            # Get generated image
            generated_image = result.images if isinstance(result.images, list) else result.images
            #print(to_tensor(generated_image).shape)
            # Create visualization
            print(len(generated_image))
            vis_images = [to_tensor(image[0])* 2 - 1 for image in generated_image]
            
            if input_images is not None:
                # Convert PIL images to tensors for concatenation
                input_tensors = []
                for img in input_images:
                    img_tensor = to_tensor(img).to(accelerator.device)
                    input_tensors.append(img_tensor* 2 - 1)
                    #print(img_tensor.shape)
                vis_images = input_tensors + vis_images
            
            # Concatenate images horizontally
            max_height = max(img.shape[-2] for img in vis_images)
            total_width = sum(img.shape[-1] for img in vis_images)
            
            canvas = torch.zeros((3, max_height, total_width), device=accelerator.device)
            current_x = 0
            
            for img in vis_images:
                h, w = img.shape[-2:]
                canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
                current_x += w
            
            # Save visualization
            save_path = os.path.join(args.output_dir, f"test_visualization_{global_step}_{i}_{process_indix}.png")
            to_pil_image(canvas).save(save_path)
            
            # Save instruction text
            with open(os.path.join(args.output_dir, f"test_instruction_{global_step}_{i}_{process_indix}.txt"), "w", encoding='utf-8') as f:
                f.write(f"Task type: multiple turn edit\n")
                f.write(f"Instruction: {prompt}\n")
                f.write(f"Target size: {test_item['target_img_size']}\n")
                f.write(f"Inference steps: {args.val.get('num_inference_steps', 28)}\n")
                f.write(f"Text guidance scale: {args.val.get('text_guidance_scale', 4.0)}\n")
                f.write(f"Image guidance scale: {args.val.get('image_guidance_scale', 1.0)}")
    del pipeline, vis_images, generated_image, input_tensors
    #model.train()

def fully_restore_model(accelerator, model):
    """
    完全还原模型，包括解包和解扁平化权重
    适用于 FSDP/ZeRO-3 训练场景
    """
    # 1. 解除分布式包装
    raw_model = accelerator.unwrap_model(model)
    
    # 2. 处理 FSDP 扁平化权重 - 关键步骤！
    print(accelerator.distributed_type)
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        # 方案A: 保存并重新加载模型（推荐）
        from accelerate.utils import save_fsdp_model, load_fsdp_model
        temp_dir = "./temp_fsdp_model"
        save_fsdp_model(accelerator, raw_model, output_dir=temp_dir)
        restored_model = load_fsdp_model(temp_dir, device_map="cpu")
        
        # 方案B: 手动解扁平化（备选）
        # 注意：此方法可能不适用于所有模型结构
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # if isinstance(raw_model, FSDP):
        #     with FSDP.summon_full_params(raw_model):
        #         restored_model = deepcopy(raw_model.module)
    
    else:
        # 非FSDP模型的常规处理
        restored_model = deepcopy(raw_model)
    
    # 3. 确保在CPU上且为FP32
    restored_model = restored_model.to("cpu").float()
    
    # 4. 重置模型状态
    restored_model.eval()
    for param in restored_model.parameters():
        param.requires_grad = False
    
    # 5. 清理临时文件
    if accelerator.distributed_type == accelerator.DistributedType.FSDP:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return restored_model


def apply_format(instruction):
    prompt=f"""<|im_start|>system
    You are a helpful assistant that describes and understands images.
    <|im_end|>
    <|im_start|>user<|vision_start|><|image_pad|><|vision_end|>
    First describe what you see in this image, and edit the image: {instruction} <|im_end|>
    """
    return prompt

def save_gradients_fsdp(model):
    """Save gradients for FSDP model"""
    gradients = {}
    for name, param in model.named_parameters():
        #print(param.grad)
        if param.grad is not None:
            print('check grad')
            # For FSDP, we need to gather the sharded gradients
            with torch.no_grad():
                full_grad = torch.zeros_like(param.grad)
                full_grad.copy_(param.grad)
                gradients[name] = full_grad
        #else:
        #    gradients[name] = None
    return gradients

def restore_gradients_fsdp(model, gradients):
    """Restore gradients for FSDP model"""
    #i = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            
            # Copy the saved gradient back
            param.grad = gradients[name].to(param.device)
            #i = i + 1
        else:
            param.grad = None

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

@staticmethod
# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Performs one GRPO step for flow matching models using Euler discretization.
    
    Args:
        model_output: Model prediction (velocity v_t)
        latents: Current latent state at time t
        eta: Noise scaling factor for stochastic sampling
        sigmas: Sigma/timestep schedule (from 1.0 to 0.0)
        index: Current step index
        prev_sample: Previous sample (if provided, used for computing log_prob only)
        generator: Random generator for sampling
    
    Returns:
        prev_sample: Next latent state
        pred_original_sample: Predicted original sample (x0)
        log_prob: Log probability of the transition
        prev_sample_mean: Mean of the transition distribution
        std_dev_t: Standard deviation of the transition
    """
    device = model_output.device
    dtype = model_output.dtype
    
    # Get current and next sigma values
    sigma = sigmas[index].to(device)
    sigma_next = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_next - sigma  # This is negative since we go from 1.0 to 0.0
    
    # For Flow Matching: x_{t+dt} = x_t + v_t * dt
    # Since model_output is the velocity prediction
    pred_original_sample = latents + model_output * dt
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    prev_sample_mean = latents*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    # Add stochastic noise if eta > 0
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, 
            generator=generator, 
            device=device, 
            dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
    
    # Sample next latent if not provided
    if prev_sample is None:
        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=dtype
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise
        else:
            prev_sample = prev_sample_mean
    
    # Compute log probability of the transition
    #if eta > 0 and std_dev_t > 0:
        # Gaussian log probability: log p(x) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
    log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2 + 1e-8))
            - torch.log(std_dev_t + 1e-8)
            - 0.5 * torch.log(2 * torch.as_tensor(math.pi, device=device, dtype=dtype))
        )
    
    
    # Mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # Predict original sample x0 using the flow matching relationship
    # In flow matching: x_t = (1-t)*x_0 + t*x_1 where x_1 is noise
    # So: x_0 = (x_t - t*x_1) / (1-t)
    # But for simplicity, we can approximate x0 from the velocity
    #pred_original_sample = latents + sigma * model_output
    
    return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t

'''
def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)
'''

def gather_tensor(tensor, accelerator):

    return accelerator.gather(tensor)
def create_grid_with_prompt(input_image, output_images, prompt_text, save_path, grid_columns=2, font_size=20):
    """
    Combine input image, output images, and prompt text into a single grid image.
    
    Args:
        input_image (PIL.Image): The original input image.
        output_images (list[PIL.Image]): List of generated output images.
        prompt_text (str): The instruction/prompt used for generation.
        save_path (str): Path to save the grid image.
        grid_columns (int): Number of columns in the grid.
        font_size (int): Font size for the prompt text.
    """
    images = input_image + output_images
    widths, heights = zip(*(img.size for img in images))
    print(widths, heights)
    # Calculate grid dimensions
    max_width = max(widths)
    max_height = max(heights)
    grid_rows = (len(images) + grid_columns - 1) // grid_columns
    
    # Create a blank canvas (extra space for text at the bottom)
    text_height = 0  # Space for prompt text
    grid = Image.new(
        "RGB",
        (max_width * grid_columns, max_height * grid_rows + text_height),
        color="white"
    )
    with open('file.txt', 'a+') as file:
        file.write(prompt_text)
        file.write('\n')
    # Paste images into the grid

    for i, img in enumerate(images):
        row = i // grid_columns
        col = i % grid_columns
        grid.paste(img, (col * max_width, row * max_height))
    
    '''
    # Add prompt text at the bottom
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Use Arial if available
    except:
        font = ImageFont.load_default()  # Fallback to default font
    
    # Wrap long prompts into multiple lines
    max_text_width = max_width * grid_columns - 20  # Padding
    lines = []
    current_line = ""
    for word in prompt_text.split():
        test_line = current_line + " " + word if current_line else word
        if draw.textlength(test_line, font=font) <= max_text_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    # Draw each line of text
    y_text = max_height * grid_rows + 10  # Start below images
    for line in lines:
        draw.text((10, y_text), line, fill="black", font=font)
        y_text += font_size + 5  # Adjust line spacing
    '''
    grid.save(save_path)
    print(f"Grid image with prompt saved to {save_path}")

from safetensors import safe_open

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

def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, 'logs'))
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=10800))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.train.mixed_precision,
        log_with=OmegaConf.to_object(args.logger.log_with),
        project_config=accelerator_project_config,
        kwargs_handlers=[init_process_group_kwargs]
    )

    setup_logging(args, accelerator)
    
    # Reproducibility
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    # Set performance flags
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    ema_decay = args.train.get('ema_decay', 0)
    
    
    model = QwenImageTransformer2DModel.from_pretrained(
        args.model.pretrained_model_path, 
        subfolder="transformer",
        #empty_init=False,
        #device_map="cpu",
        #n_dc_layers=8,
        #token_length=64,
        #total_length=128,
    )
    #print(model.device)
    model.locate_pos = model.locate_pos.to_empty(device=model.device)
    #model.dc_tokens = nn.init.normal_(nn.Parameter(torch.randn(8, 64, 3584).to('cpu')), mean=0, std=0.02)#model.dc_tokens.new_empty(size=(8, 64, 3584), device=model.device) #nn.init.normal_(nn.Parameter(torch.randn(8, 64, 3584).to('cpu')), mean=0, std=0.02) #
    model.set_dynamic_tokens(n_dc_layers=10, token_length=64)
    print(model.dc_tokens.device)
    model.train()
    #print(f"Model device before prepare: {next(model.parameters()).device}")
    #state_dict = load_large_safetensors(args.model.pretrained_model_path+'/transformer')
    #model.load_state_dict(state_dict, strict=False)
    #model.dc_tokens = model.dc_tokens.to_empty(device=accelerator.device)
    # model = OmniGen2Transformer2DModel(**args.model.arch_opt)
    # model.train()

    # if args.model.get("pretrained_model_path", None) is not None:
    #     logger.info(f"Loading model parameters from: {args.model.pretrained_model_path}")
    #     state_dict = torch.load(args.model.pretrained_model_path, map_location="cpu")
    #     missing, unexpect = model.load_state_dict(state_dict, strict=False)
    #     logger.info(
    #         f"missed parameters: {missing}",
    #     )
    #     logger.info(f"unexpected parameters: {unexpect}")

    if ema_decay != 0:
        model_ema = deepcopy(model)
        model_ema._requires_grad = False
    
    #model_ema = deepcopy(model)
    #model_ema._requires_grad = False
    #model.requires_grad = False

    #text_tokenizer = Qwen2Tokenizer.from_pretrained(args.model.pretrained_model_path, subfolder="tokenizer")
    #text_tokenizer.padding_side = "right"

    #if accelerator.is_main_process:
    #    text_tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    '''
    text_encoder = TextEncoder.from_pretrained(
        #args.model.pretrained_text_encoder_model_name_or_path,
        args.model.pretrained_model_path,
        subfolder="text_encoder",
        #torch_dtype=weight_dtype,
    )

    processor = Qwen2VLProcessor.from_pretrained(
        args.model.pretrained_model_path,
        subfolder="processor",
        #tokenizer=text_tokenizer,
    )
    '''
    '''
    if args.model.get('resize_token_embeddings', False):
        text_encoder.resize_token_embeddings(len(text_tokenizer))

    if accelerator.is_main_process:
        text_encoder.save_pretrained(os.path.join(args.output_dir, 'text_encoder'))
    

    log_model_info("text_encoder", text_encoder)
    '''
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.model.pretrained_model_path,
        subfolder=args.model.get("vae_subfolder", "vae"),
        device_map="cpu"
    )
   
    image_processor = VaeImageProcessor(vae_scale_factor=2 * 2 ** len(vae.temperal_downsample))
    #print(vae.temperal_downsample)
    #logger.info(vae)
    #logger.info("***** Move vae, text_encoder to device and cast to weight_dtype *****")
    # Move vae, unet, text_encoder and controlnet_ema to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    #text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    #model.requires_grad_(False)
    #text_encoder.requires_grad_(False)
    #vae.requires_grad_(False)
    #processor = processor.to(accelerator.device, dtype=weight_dtype)
    args.train.lora_ft = args.train.get('lora_ft', False)
    if args.train.lora_ft:
        model.requires_grad_(False)
        model.dc_tokens.requires_grad_(True)
        model.locate_pos.requires_grad_(True)
        #target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        target_modules = ["to_k", "to_v", "to_out.0"]
        # now we will add new LoRA weights the transformer layers
        lora_config = LoraConfig(
            r=args.train.lora_rank,
            lora_alpha=args.train.lora_rank,
            lora_dropout=args.train.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.add_adapter(lora_config)
        model.locate_pos.requires_grad_(True)
        model.dc_tokens.requires_grad_(True)
    if args.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    log_model_info("transformer", model)
    for name, param in model.named_parameters():
        if param.requires_grad:
        
    
            print(f" {name}")
    
    # Optimizer creation
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    #print(trainable_params)
    optimizer = optimizer_class(
        trainable_params,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )   
    #model.requires_grad_(False)

    logger.info("***** Prepare dataset *****")

    with accelerator.main_process_first():
        train_dataset = Qwen2TrainRlDataset(
            args.data.data_path,
            use_chat_template=args.data.use_chat_template,
            prompt_dropout_prob=args.data.get('prompt_dropout_prob', 0.0),
            ref_img_dropout_prob=args.data.get('ref_img_dropout_prob', 0.0),
            max_input_pixels=OmegaConf.to_object(args.data.get('max_input_pixels', 1024 * 1024)),
            max_output_pixels=args.data.get('max_output_pixels', 1024 * 1024),
            max_side_length=args.data.get('max_side_length', 2048),
        )
    
    logger.info(f"Number of training samples: {len(train_dataset)}")

    if args.seed is not None and args.get("workder_specific_seed", False):
        from omnigen2.utils.reproducibility import worker_init_fn

        worker_init_fn = partial(
            worker_init_fn,
            num_processes=AcceleratorState().num_processes,
            num_workers=args.train.dataloader_num_workers,
            process_index=AcceleratorState().process_index,
            seed=args.seed,
            same_seed_per_epoch=args.get("same_seed_per_epoch", False),
        )
    else:
        worker_init_fn = None

    logger.info("***** Prepare dataLoader *****")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        num_workers=args.train.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        collate_fn=OmniGen2Collator(max_token_len=args.data.maximum_text_tokens)
    )

    logger.info(f"{args.train.batch_size=} {args.train.gradient_accumulation_steps=} {accelerator.num_processes=} {args.train.global_batch_size=}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train.lr_scheduler == 'timm_cosine':
        from omnigen2.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    elif args.train.lr_scheduler == 'timm_constant_with_warmup':
        from omnigen2.optim.scheduler.step_lr import StepLRScheduler

        lr_scheduler = StepLRScheduler(
            optimizer=optimizer,
            decay_t=1,
            decay_rate=1,
            warmup_t=args.train.warmup_t,
            warmup_lr_init=args.train.warmup_lr_init,
            warmup_prefix=args.train.warmup_prefix,
            t_in_epochs=args.train.t_in_epochs,
        )
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps,
            num_training_steps=args.train.max_train_steps,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")
    scheduler = FlowMatchEulerDiscreteScheduler(base_image_seq_len= 256,
        base_shift=0.5,
        invert_sigmas= False,
        max_image_seq_len= 8192,
        max_shift =0.9,
        num_train_timesteps= 1000,
        shift=1.0,
        shift_terminal= 0.02,
        stochastic_sampling= False,
        time_shift_type= "exponential",
        use_beta_sigmas= False,
        use_dynamic_shifting= True,
        use_exponential_sigmas= False,
        use_karras_sigmas= False)
    
    if args.train.ema_decay != 0:
        model, model_ema, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, model_ema, optimizer, train_dataloader, lr_scheduler
        )
        #model_ema = EMAModel(model_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(model)), model_config=model_ema.config)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        #model, optimizer = accelerator.prepare(model, optimizer)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
        
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            #path = os.path.basename(args.resume_from_checkpoint)
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    #if accelerator.is_main_process:
    #    for tracker in accelerator.trackers:
    #        if tracker.name == "wandb":
    #            logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")
    
    def decode_vae_image(latents, height, width, output_type):
        latents = _unpack_latents(latents, height, width, 8)
        latents = latents.to(vae.dtype)
        latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
        latents = latents / latents_std + latents_mean
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]
        image_pt = image_processor.postprocess(image, output_type='pt')
        image_pil = image_processor.postprocess(image.detach(), output_type='pil')

        return image_pt, image_pil

    def encode_vae(img):
        print(img.shape)
        z0 = vae.encode(img.unsqueeze(2).to(dtype=vae.dtype)).latent_dist.sample()
        z0 = z0.to(dtype=weight_dtype)
        return z0

    num_inference_steps = 20

    for epoch in range(first_epoch, args.train.num_train_epochs):
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            break
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Number of bins, for loss recording
           
            batch_instruction = batch['instruction']
            input_latents = batch['input_latent']
            input_images = batch['input_images']
            batch_size = len(batch_instruction)

            if args.use_group:
                def repeat_tensor(tensor):
                    if tensor is None:
                        return None
                    return torch.repeat_interleave(tensor, args.num_generations, dim=0)

                batch['encoder_hidden'] = repeat_tensor(batch['encoder_hidden'])
                batch['encoder_hidden_mask'] = repeat_tensor(batch['encoder_hidden_mask'])
                input_latents = repeat_tensor(input_latents)
                input_images = repeat_tensor(input_images)
                batch['think_instruction'] = repeat_tensor(batch['think_instruction'])
                if 'input_image_path' in batch:
                    batch['input_image_path'] = repeat_tensor(batch['input_image_path'])

                if isinstance(batch_instruction, str):
                    batch_instruction = [batch_instruction] * args.num_generations
                elif isinstance(batch_instruction, list):
                    batch_instruction = [item for item in batch_instruction for _ in range(args.num_generations)]
                else:
                    raise ValueError(f"Unsupported batch_instruction type: {type(batch_instruction)}")

                batch_size = len(batch_instruction)
            think_instruction = batch['think_instruction']
            print(f"Processing batch of size: {batch_size}")
            print(f"Input latents shape: {input_latents[0].shape}")

            # 5. Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            all_image_latents = []
            all_latents = []
            all_log_probs=[]
            all_prev_samples = []
            all_img_shapes = []
            all_txt_seq_lens = []
            all_image_seq_lens = []
            all_input_data = []
            image_pils = []
            for i in range(batch_size):
                image_seq_len = input_latents[i].shape[1]//2
                mu = calculate_shift(
                    image_seq_len,
                    scheduler.config.get("base_image_seq_len", 256),
                    scheduler.config.get("max_image_seq_len", 4096),
                    scheduler.config.get("base_shift", 0.5),
                    scheduler.config.get("max_shift", 1.15),
                )
                timesteps, num_inference_steps = retrieve_timesteps(
                    scheduler,
                    num_inference_steps,
                    model.device,
                    sigmas=sigmas,
                    mu=mu,
                )
                image_latents = input_latents[i][:, image_seq_len:]
                latents = input_latents[i][:, :image_seq_len]
                all_image_seq_lens.append(image_seq_len)
                all_image_latents.append(image_latents)
                #all_latents.append(latents)
                image_size = input_images[i][0].size
                #print(image_size)
                calculated_width, calculated_height, _  = calculate_dimensions(512*512, image_size[0] / image_size[1])
                img_shapes = [
                    [
                        (1, calculated_height // 8 // 2, calculated_width // 8 // 2),
                        (1, calculated_height // 8 // 2, calculated_width // 8 // 2)]
                ] * args.train.batch_size
                #print(img_shapes)
                all_img_shapes.append(img_shapes)
                txt_seq_lens = batch['encoder_hidden_mask'][i].sum(dim=1).tolist()
                all_txt_seq_lens.append(txt_seq_lens)
            
            #image_latents = torch.cat(all_image_latents, dim=0)
            #latents = torch.cat(all_latents, dim=0)
            
                scheduler.set_begin_index(0)
                
                # GRPO hyperparameters
                eta = args.train.get('grpo_eta', 0.1)
                clip_range = args.train.get('grpo_clip_range', 1e-4)
                adv_clip_max = args.train.get('grpo_adv_clip_max', 5.0)
                kl_beta = args.train.get('grpo_kl_beta', 0.01)
                timestep_fraction = args.train.get('grpo_timestep_fraction', 1.0)
                
                # Phase 1: Reference sampling (collect trajectories with log_probs)
                all_latents_list = []
                all_log_probs_list = []
                all_prev_sample_mean_ref_list = []
                sigma_schedule = scheduler.sigmas #torch.linspace(1.0, 0.0, len(timesteps) + 1)
                latents_ref = latents.clone()
                print("Phase 1: Reference sampling...", accelerator.process_index)
                with torch.no_grad():
                    for j, t in enumerate(timesteps):
                        timestep = t.expand(1).to(latents.dtype)
                        next_timestep = timesteps[j + 1] if j + 1 < len(timesteps) else torch.tensor(0.0, device=timestep.device, dtype=timestep.dtype)
                        next_timestep = next_timestep.expand(1).to(latents.dtype)
                        batch_encoder_hidden = batch['encoder_hidden'][i]#torch.cat([batch['encoder_hidden'][j] for j in range(batch_size)], dim=0)
                        batch_encoder_mask = batch['encoder_hidden_mask'][i]#torch.cat([batch['encoder_hidden_mask'][j] for j in range(batch_size)], dim=0)
                        latent_model_input = torch.cat([latents_ref, image_latents], dim=1)
                        #print(batch_encoder_hidden.shape)
                        
                        model_kwargs = dict(
                            encoder_hidden_states=batch_encoder_hidden,
                            encoder_hidden_states_mask=batch_encoder_mask,
                            txt_seq_lens=txt_seq_lens,
                            img_shapes=img_shapes,
                        )
                        
                        #print(latent_model_input.shape)
                        # Forward pass
                        pred_latents = model(hidden_states=latent_model_input, timestep=timestep/1000, return_dict=False, **model_kwargs)
                        model_output = pred_latents[0][:, :image_seq_len]
                        
                        # GRPO step with sampling
                        latents_ref, pred_original, log_prob, prev_sample_mean, std_dev = flow_grpo_step(
                            model_output=model_output.to(torch.float32),
                            latents=latents_ref.to(torch.float32),
                            eta=eta,
                            sigmas=sigma_schedule,
                            index=j,
                            prev_sample=None,
                            generator=None,
                        )
                        latents_ref = latents_ref.to(weight_dtype)
                        
                        all_latents_list.append(latents_ref.clone())
                        all_log_probs_list.append(log_prob)
                        all_prev_sample_mean_ref_list.append(prev_sample_mean)
            
                # Stack collected data
                #all_latents_sample = torch.stack(all_latents_list, dim=1)  # (batch, num_steps+1, seq_len, hidden_dim)
                #all_log_probs_sample = torch.stack(all_log_probs_list, dim=1)  # (batch, num_steps)
                #all_prev_sample_mean_ref_sample = torch.stack(all_prev_sample_mean_ref_list, dim=1)  # (batch, num_steps, ...)
                all_latents.append(torch.stack(all_latents_list, dim=1))
                all_log_probs.append(torch.stack(all_log_probs_list, dim=1))
                all_prev_samples.append(torch.stack(all_prev_sample_mean_ref_list, dim=1))
            # Phase 2: Compute rewards
            
                print("Phase 2: Computing rewards...")
                image, image_pil = decode_vae_image(latents_ref, calculated_height, calculated_width, 'pt')
                image_pil[0].save(args.output_dir + f'/output/test_{step}_{accelerator.process_index}_batch_{i}.png')
                image_pils.append(image_pil)
                all_input_data.append({
                    "images": [args.output_dir + f'/output/test_{step}_{accelerator.process_index}_batch_{i}.png', batch['input_image_path'][i]],
                    "problem": template.format(prompt=think_instruction[i])
                })
            
            
            all_response = evaluate_batch(all_input_data, api_url='http://localhost:8080')
            alignment_reward, style_reward, consistency_reward, dim_reward = extract_normalized_rewards([response['model_output'] for response in all_response]) 
            
            #print(all_response)
            alignment_reward = torch.cat(alignment_reward, dim=0)
            style_reward = torch.cat(style_reward, dim=0)
            consistency_reward = torch.cat(consistency_reward, dim=0)
            print(alignment_reward, style_reward, consistency_reward)

            alignment_advantages = (alignment_reward - alignment_reward.mean())/(alignment_reward.std()+1e-8)
            style_advantages = (style_reward - style_reward.mean())/(style_reward.std()+1e-8)
            consistency_advantages = (consistency_reward - consistency_reward.mean())/(consistency_reward.std()+1e-8)
            advantages =  0.7*style_advantages + 1.4*alignment_advantages + 1.0*consistency_advantages
            #print(advantages, style_advantages, consistency_advantages, alignment_advantages)
            advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
            
            #advantages = 2.5
            #print(f"Reward: {reward.item():.4f}, Advantage: {advantages.item():.4f}")
            samples = {
                "latents": [all_latents[i][:, :-1] for i in range(batch_size)],  # each entry is the latent before timestep t
                "next_latents": [all_latents[i][:, 1:] for i in range(batch_size)],  # each entry is the latent after timestep t
                "log_probs": [all_log_probs[i][:, :-1] for i in range(batch_size)],
                "prev_sample_mean_ref": [all_prev_samples[i][:, :-1] for i in range(batch_size)],
                #"alignment_reward": alignment_reward.to(torch.float32),
                #"style_reward": style_reward.to(torch.float32),
                #"consistency_reward": consistency_reward.to(torch.float32),
                "encoder_hidden_states": batch['encoder_hidden'],
               
            }
            
            list_keys = ["latents", "next_latents", "log_probs", "prev_sample_mean_ref", "encoder_hidden_states"]
            num_samples = len(samples[list_keys[0]])  # Should be 2 based on the data structure
            
            samples_batched_list = []
            for i in range(num_samples):
                sample_dict = {}
                
                # Handle list-type data (take i-th element from each list)
                for key in list_keys:
                    if key in samples:
                        sample_dict[key] = samples[key][i]
        
                samples_batched_list.append(sample_dict)
                
            #train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
            # Phase 3: GRPO training
            #print("Phase 3: GRPO training...")
            model.train()
            accelerator.wait_for_everyone()
            for ind, sample in list(enumerate(samples_batched_list)):
                with accelerator.accumulate(model):
                    # Prepare timesteps for training
                    train_timesteps = int(len(timesteps) * timestep_fraction)
                    # Randomly select which timesteps to train on
                    #train_indices = torch.randperm(len(timesteps))[:train_timesteps].to(accelerator.device)
                    
                    total_loss = 0.0
                    total_kl_loss = 0.0
                    rand_start = random.randint(0, len(timesteps)-6)
                    for train_idx in range(rand_start, rand_start+5): #indices:
                        #if t not in time_bp:
                        #    continue
                        i = train_idx #int(train_idx.item())
                        t = timesteps[i]
                        timestep = t.expand(1).to(latents.dtype)
                        
                        # Get latent at this step
                        if i == rand_start:
                            current_latent = sample["latents"][:, i]#all_latents[:, i]
                    
                        next_latent = sample["next_latents"][:,i] #all_latents[:, i + 1]
                  
                        ref_log_prob = sample["log_probs"][:,i] #all_log_probs[:, i]
                        ref_prev_sample_mean = sample['prev_sample_mean_ref'][:,i] #all_prev_samples[:, i]
                        #print(current_latent.shape, next_latent.shape)
                        model_kwargs = dict(
                            encoder_hidden_states=batch['encoder_hidden'][ind],
                            encoder_hidden_states_mask=batch['encoder_hidden_mask'][ind],
                            txt_seq_lens=all_txt_seq_lens[ind],
                            img_shapes=all_img_shapes[ind],
                        )
                        latent_model_input = torch.cat([current_latent, all_image_latents[ind]], dim=1)
                        
                        if i == rand_start:
                            pred_latents = checkpoint.checkpoint(model, latent_model_input.detach(), batch['encoder_hidden'][ind], batch['encoder_hidden_mask'][ind], timestep/1000, all_img_shapes[ind], all_txt_seq_lens[ind], use_reentrant=False)
                            model_output = pred_latents[0][:, :all_image_seq_lens[ind]]   
                        else:
                            with torch.no_grad():
                                pred_latents = model(hidden_states=latent_model_input, timestep=timestep/1000, return_dict=False, **model_kwargs)
                                model_output = pred_latents[0][:, :all_image_seq_lens[ind]] 
                

                        _, _, new_log_prob, new_prev_sample_mean, new_std_dev = flow_grpo_step(
                                model_output=model_output, #.to(torch.float32),
                                latents=current_latent, #.to(torch.float32),
                                eta=eta,
                                sigmas=sigma_schedule,
                                index=i,
                                prev_sample=next_latent, #.to(torch.float32),
                                generator=None,
                            )
                        
                        # Compute PPO-style clipped loss
                        ratio = torch.exp(new_log_prob - ref_log_prob)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                        ppo_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        current_latent = scheduler.step(model_output, t, current_latent, return_dict=False)[0]
                
                        # Compute KL divergence loss
                        kl_loss = ((new_prev_sample_mean - ref_prev_sample_mean) ** 2).mean() / (2 * new_std_dev ** 2 + 1e-8)
                        
                        # Combined loss
                        loss = (ppo_loss + kl_beta * kl_loss) / train_timesteps
                        
                        total_loss += loss#.item()
                        total_kl_loss += kl_loss.item()
                        #accelerator.backward(total_loss)
                    print(f'GRPO training complete - Loss: {total_loss.item():.4f}, PPO: {ppo_loss}, KL Loss: {total_kl_loss:.4f}, advantages:{advantages}')
                    
                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, args.train.max_grad_norm)
                    
                    optimizer.step()
                    if 'timm' in args.train.lr_scheduler:
                        lr_scheduler.step(global_step)
                    else:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)
                    
                #image_pil[0].save(f'outputs/test_{step}_process_{accelerator.process_index}_sample_{ind}.png')
                create_grid_with_prompt(input_images[ind], image_pils[ind], batch_instruction[0], args.output_dir + f'/output/test_{step}_process_{accelerator.process_index}_sample_{ind}.png')
            
            global_step += 1

            if (step+1)%10==0:
                #if accelerator.is_main_process:
                os.makedirs(args.output_dir+f"/checkpoint-{global_step}/", exist_ok=True)
                accelerator.wait_for_everyone()
                accelerator.save_state(args.output_dir+f"/checkpoint-{global_step}/")
               
                if accelerator.is_main_process:
                    if args.logger.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        
                        if len(checkpoints) >= args.logger.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                logger.info(f"Saving checkpoints at step {step}")
                
                with torch.no_grad():
                #    #
                    with FSDP.summon_full_params(model, writeback=False, offload_to_cpu=True):
                        if accelerator.is_main_process:
                            unwrapped = accelerator.unwrap_model(model).to(weight_dtype)
                            unwrapped.save_pretrained(args.output_dir, safe_serialization=True)
                            del unwrapped
                     
            if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                break
            
    accelerator.wait_for_everyone()
    save_path = os.path.join(args.output_dir, f"multi_turn_model.safetensors")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)