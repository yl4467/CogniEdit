from typing import Optional, Union, List

import os
import random
import yaml
import glob
from PIL import Image
import math
import torch
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from datasets import load_dataset, concatenate_datasets
from diffusers.image_processor import VaeImageProcessor
#from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None

def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

def _get_qwen_prompt_embeds(
    prompt: Union[str, List[str]] = None,
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    processor=None,
    text_encoder=None,

):
    #device = device or self._execution_device
    #dtype = dtype or self.text_encoder.dtype
    print(prompt)
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

    template = prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    drop_idx = 64
    txt = [template.format(base_img_prompt + e) for e in prompt]

    model_inputs = processor(
        text=txt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)
    print(model_inputs)
    if torch.isnan(model_inputs.input_ids).any() or torch.isnan(model_inputs.pixel_values).any():
        print("Warning: NaN in input")
    outputs = text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )
    #print(outputs.hidden_states[-1])
    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(hidden_states, model_inputs.attention_mask)
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


def encode_prompt(
    prompt: Union[str, List[str]],
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
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
    #device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(prompt, image, device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

    return prompt_embeds, prompt_embeds_mask

class Qwen2TrainRlDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

    def __init__(
        self,
        config_path: str,
        use_chat_template: bool,
        max_input_pixels: Optional[Union[int, List[int]]] = None,
        max_output_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
    ):
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels

        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob

        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.use_chat_template = use_chat_template
        self.image_processor = VaeImageProcessor(vae_scale_factor=img_scale_num)

        data = self._collect_annotations(self.config)

        self.data = data
        #self.tokenizer = tokenizer
        
        #self.processor = processor
        #self.text_encoder = text_encoder
        
    def _collect_annotations(self, config):
        total_samples = 0
        total_ratio = 0
        json_datasets = []
        for data in config['data']:
            data_path, data_type = data['path'], data.get("type", "default")
            if os.path.isdir(data_path):
                jsonl_files = list(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)) + list(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
                json_dataset = load_dataset('json', data_files=jsonl_files, cache_dir=None)['train']
            else:
                data_ext = os.path.splitext(data_path)[-1]
                if data_ext in [".json", ".jsonl"]:
                    json_dataset = load_dataset('json', data_files=data_path, cache_dir=None)['train']
                elif data_ext in [".yml", ".yaml"]:
                    with open(data_path, "r") as f:
                        sub_config = yaml.load(f, Loader=yaml.FullLoader)
                        json_dataset = self._collect_annotations(sub_config)
                else:
                    raise NotImplementedError(
                        f'Unknown data file extension: "{data_ext}". '
                        f"Currently, .json, .jsonl .yml .yaml are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
            total_ratio += data['ratio']
            total_samples += len(json_dataset)
            json_datasets.append(json_dataset)
        
        for json_dataset in json_datasets:
            target_size = int(len(json_dataset) * data['ratio'] / total_ratio) # normalize the ratio
            if target_size <= len(json_dataset):
                # Random selection without replacement
                indices = random.sample(range(len(json_dataset)), target_size)
            else:
                # Oversample with replacement
                indices = random.choices(range(len(json_dataset)), k=target_size)
            json_dataset = json_dataset.select(indices)
            
        json_dataset = concatenate_datasets(json_datasets)
        return json_dataset
    
    def clean_data_item(self, data_item):
        #task_type = data_item['task_type']
        prefixs = ["The image portrays ", "The image depicts ", "The image captures ", "The image highlights ", "The image shows ", "这张图片展示了"]
        if "text_to_image" in task_type or "t2i" in task_type:
            if random.random() < 0.5:
                for p in prefixs:
                    if p in data_item['instruction']:
                        data_item['instruction'] = data_item['instruction'].replace(p, "")
                        break
        return data_item
    
    def apply_chat_template(self, instruction, system_prompt):
        if self.use_chat_template:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction
    
    def process_item(self, data_item):
        
        #instructions = sorted(instructions, key=lambda x: int(x.split("_")[1]))
        assert data_item['instruction'] is not None
        #data_item = self.clean_data_item(data_item)
        if "source_image" in data_item and data_item["source_image"] is not None:
            if data_item["source_image"].startswith("images"):
                root_path = "/home/ma-user/work/yanli/test_yl/Datasets/"
            else:
                root_path = "/home/ma-user/work/yanli/test_yl/Omni_Pos_Data/"
            instruction = data_item['instruction']
            think_instruction = data_item['object'][0]
            caption = data_item['info'][0]
            input_image = [Image.open(root_path + data_item["source_image"]).convert("RGB")]
            input_image_path = root_path  + data_item["source_image"]
            input_id = os.path.basename(data_item['source_image'])[:-4]
            root_path = root_path + "input_data_rl_512"
            input_latents = torch.load(os.path.join(root_path, input_id + '_input_latent.pt'),  map_location='cpu')
            encoder_hidden = torch.load(os.path.join(root_path, input_id + '_prompt_embed.pt'), map_location='cpu')
            encoder_hidden_mask = torch.load(os.path.join(root_path, input_id + '_prompt_embed_mask.pt'),  map_location='cpu')


        '''
        output_images = []
        output_image_paths = []
        if 'input_images' in data_item and data_item['input_images'] is not None:
            input_images_path = data_item['input_images']
            input_images = []

            max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)
        elif "source_image" in data_item and data_item["source_image"] is not None:
            root_path = "/home/ma-user/work/yanli/test_yl/Datasets/multi_turn_editing/images/data"
            input_images_path = [os.path.join(root_path, data_item["source_image"])]
            #print(f"input_images_path: {input_images_path}")
            input_images = [Image.open(input_images_path[0]).convert("RGB")]
            input_images_pil = [Image.open(input_images_path[0]).convert("RGB")]
            #print(f"input_images: {input_images}")
            max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            image_size = input_images[0].size if isinstance(input_images, list) else input_images.size
            calculated_width, calculated_height, _ = calculate_dimensions(512*512, image_size[0] / image_size[1])
            
            input_images[0] = self.image_processor.preprocess(input_images[0], calculated_width, calculated_height)
            #print(data_item)
            key_items=data_item.keys()
            instructions = [d for d in key_items if d.startswith("instruction")]
            prompts = []
            output_images_pil = []
            #print(instructions)
            for i in range(1, len(instructions)+1):
                drop_prompt = random.random() < self.prompt_dropout_prob
                drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
                #print(data_item[f'instruction_{i}'])
                if data_item[f'edit_image_{i}']==None:
                    prompts.append(None)
                    output_image_paths.append(None)
                    output_images.append(None)
                else:
                
                    instruction = data_item[f'instruction_{i}'] #self.apply_chat_template(data_item[f'instruction_{i}'], self.SYSTEM_PROMPT)

                    prompts.append(instruction)
                    output_image_path = os.path.join(root_path, data_item[f'edit_image_{i}'])
                    output_image_paths.append(output_image_path)
                    output_image = Image.open(output_image_path).convert("RGB")
                
                    output_image = self.image_processor.preprocess(output_image, calculated_width, calculated_height)
                    #print(calculated_width)
                    output_images.append(output_image)
        else:
            input_images_path, input_images,  input_images_pil = None, None, None
            #output_image_path = data_item['output_image']    
        '''
        data = {
            #'task_type': data_item['task_type'],
            'instruction': instruction,
            'caption': caption,
            'think_instruction': think_instruction,
            'input_latent': input_latents,
            'encoder_hidden': encoder_hidden,
            'encoder_hidden_mask': encoder_hidden_mask,
            'input_image_path': input_image_path,
            'input_images': input_image,
            #'input_images_pil': input_images_pil,
            #'output_image': output_images,
            #'output_image_path': output_image_paths,
        }
        return data

    def __getitem__(self, index):
        #current_index = index
        #data_item = self.data[current_index]
        #return self.process_item(data_item)
        
        max_retries = 12 
        current_index = index
        for attempt in range(max_retries):
            try:
                data_item = self.data[current_index]
                return self.process_item(data_item)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Try a different index for the next attempt
                    current_index = random.randint(0, len(self.data) - 1)
                    continue
        
    def __len__(self):
        return len(self.data)

class OmniGen2TestMultiDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."

    def __init__(
        self,
        config_path: str,
        use_chat_template: bool,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        align_res: bool = True
    ):
        
        self.max_pixels = max_pixels
        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.align_res = align_res

        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.use_chat_template = use_chat_template
        self.image_processor = VaeImageProcessor(vae_scale_factor=img_scale_num)

        data = self._collect_annotations(self.config)

        self.data = data
        
    def _collect_annotations(self, config):
        json_datasets = []
        for data in config['data']:
            data_path, data_type = data['path'], data.get("type", "default")
            if os.path.isdir(data_path):
                jsonl_files = list(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)) + list(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
                json_dataset = load_dataset('json', data_files=jsonl_files, cache_dir=None)['train']
            else:
                data_ext = os.path.splitext(data_path)[-1]
                if data_ext in [".json", ".jsonl"]:
                    json_dataset = load_dataset('json', data_files=data_path, cache_dir=None)['train']
                elif data_ext in [".yml", ".yaml"]:
                    with open(data_path, "r") as f:
                        sub_config = yaml.load(f, Loader=yaml.FullLoader)
                        json_dataset = self._collect_annotations(sub_config)
                else:
                    raise NotImplementedError(
                        f'Unknown data file extension: "{data_ext}". '
                        f"Currently, .json, .jsonl .yml .yaml are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
            json_datasets.append(json_dataset)
        
        json_dataset = concatenate_datasets(json_datasets)
        return json_dataset
    
    def apply_chat_template(self, instruction, system_prompt):
        if self.use_chat_template:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction
    
    def process_item(self, data_item):
        assert data_item['instruction_1'] is not None
        if 'input_images' in data_item and data_item['input_images'] is not None:
            input_images_path = data_item['input_images']
            input_images = []

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_images.append(input_image)
        
        elif "source_image" in data_item and data_item["source_image"] is not None:
            output_images = []
            output_image_paths = []
            root_path = "/home/ma-user/work/yanli/test_yl/Datasets/multi_turn_editing/images/data"
            input_images_path = [os.path.join(root_path, data_item["source_image"])]
            #print(f"input_images_path: {input_images_path}")
            input_images = [Image.open(input_images_path[0]).convert("RGB")]
            #input_images_pil = [Image.open(input_images_path[0]).convert("RGB")]
            #print(f"input_images: {input_images}")
            #max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            #input_images[0] = self.image_processor.preprocess(input_images[0], max_pixels=max_input_pixels, max_side_length=self.max_side_length)
            #print(data_item)
            key_items=data_item.keys()
            instructions = [d for d in key_items if d.startswith("instruction")]
            prompts = []
            #print(instructions)
            for i in range(1, len(instructions)+1):
                '''
                drop_prompt = random.random() < self.prompt_dropout_prob
                drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
                #print(data_item[f'instruction_{i}'])
                if data_item[f'edit_image_{i}']==None:
                    continue
                if drop_prompt:
                    instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
                else:
                    instruction = self.apply_chat_template(data_item[f'instruction_{i}'], self.SYSTEM_PROMPT)
                '''
                if data_item[f'edit_image_{i}']==None:
                    continue
                instruction = data_item[f'instruction_{i}']
                prompts.append(instruction)
                output_image_path = os.path.join(root_path, data_item[f'edit_image_{i}'])
                output_image_paths.append(output_image_path)
                output_image = Image.open(output_image_path).convert("RGB")
                #output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)
                output_images.append(output_image)
            #max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels
            #input_images[0] = self.image_processor.preprocess(input_images[0], max_pixels=max_input_pixels, max_side_length=self.max_side_length)
            #output_image_path = os.path.join(root_path, data_item['target_image'])
        else:
            input_images_path, input_images = None, None

        if input_images is not None and len(input_images) == 1 and self.align_res:
            target_img_size = (input_images[0].width, input_images[0].height)
        else:
            target_img_size = data_item["target_img_size"]

        w, h = target_img_size
        cur_pixels = w * h
        ratio = min(1, (self.max_pixels / cur_pixels) ** 0.5)

        target_img_size = (int(w * ratio) // self.img_scale_num * self.img_scale_num, int(h * ratio) // self.img_scale_num * self.img_scale_num)

        data = {
            'instruction': prompts,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'target_img_size': target_img_size,
            'output_images': output_images,
        }
        return data

    def __getitem__(self, index):
        data_item = self.data[index]
        return self.process_item(data_item)
        
    def __len__(self):
        return len(self.data)

class OmniGen2Collator():
    def __init__(self,max_token_len):

        self.max_token_len = max_token_len

    def __call__(self, batch):
        #task_type = [data['task_type'] for data in batch]
        instruction = [data['instruction'] for data in batch]
        caption = [data['caption'] for data in batch]
        think_instruction = [data['think_instruction'] for data in batch]
        input_latents =  [data['input_latent'] for data in batch]
        encoder_hidden =  [data['encoder_hidden'] for data in batch]
        encoder_hidden_mask =  [data['encoder_hidden_mask'] for data in batch]
        input_image_path = [data['input_image_path'] for data in batch]
        input_images = [data['input_images'] for data in batch]
        #output_image = [data['output_image'] for data in batch]
        #output_image_path = [data['output_image_path'] for data in batch]
        #input_images_pil = [data['input_images_pil'] for data in batch]

        '''
        text_inputs = self.tokenizer(
            instruction[0][0],
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )
        '''
        #prompt_embeds, prompt_embeds_mask = encode_prompt(
        #    instruction[0][0], 
        #    input_images_pil[0][0])
        
        data = {
            #"task_type": task_type,
            "instruction": instruction,
            'caption': caption,
            'think_instruction': think_instruction,
            'input_latent': input_latents,
            'encoder_hidden': encoder_hidden,
            'encoder_hidden_mask': encoder_hidden_mask,
            #"text_ids": text_inputs.input_ids,
            #"text_mask": text_inputs.attention_mask,
            "input_images": input_images, 
            #"input_images_pil": input_images_pil,
            "input_image_path": input_image_path,
            #"output_image": output_image,
            #"output_image_path": output_image_path,
        }
        return data
