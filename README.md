# CogniEdit: Dense Gradient Flow Optimization for Fine-Grained Image Editing (CVPR 26)

## Overview
![CogniEdit Pipeline](assets/cogniedit_pipeline.png)

### Data Preprocess
1. We use two datasets: [COCO 2017](https://cocodataset.org/#home) and [SEED-Data-Edit](https://huggingface.co/datasets/haotian-liu/SEED-Data-Edit).

2. For the COCO 2017 dataset, we retain a single mask per image to designate the target region, which helps the model enhance its localization ability. You can refer to the code in [`preprocess_coco.py`](preprocess_coco.py) for constructing the masked images.

3. Next, we select relevant image pairs from both datasets. The IDs of the selected image pairs are stored in JSONL files located at `data_configs/train/example/edit/jsonls/`.

4. Then to accelerate the training, we pre-generated the "prompt_embeds", "prompt_embeds_mask", "input_latent_path", you may refer to `preprocess_rl/preprocess_data.py` to process you own data.

## Experiment Setup

1. Install the requirement packages
```bash
conda create -n cogni python=3.11
conda activate cogni
pip install -r requirements.txt
```

2. Install vLLM (for UnifiedReward-based rewards)
```bash
conda create -n vllm
conda activate vllm
pip install "vllm>=0.11.0"
pip install qwen-vl-utils==0.0.14
```

## Download models

1. Download the qwen-image-edit from [https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2509](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2509)

2. Download the reward model with `huggingface-cli download CodeGoat24/UnifiedReward-Think-qwen3vl-8b`

3. Download the thinking models from [https://huggingface.co/cythu/PeBR_R1/tree/main/PeBR_R1_7B](https://huggingface.co/cythu/PeBR_R1/tree/main/PeBR_R1_7B)

## Training
1. First start the reward model server
```bash
bash vllm_utils/vllm_server_UnifiedReward_Edit.sh
```
2. Modify the model path and data path to your own path in `options/ft_qwen_rl.yml`

3. run the training scripts

```bash
bash scripts/train/ft_qwen_rl_v1.sh
```

## Inference

After training your own model, you can use the inference scripts to generate editing samples:
```bash
bash example_edit_test.sh
```

## Credit

We gratefully acknowledge the contribution and inspiration from the following projects:

- [Pref-GRPO](https://github.com/THUDM/Pref-GRPO)
- [OmniGen2](https://github.com/alpha-vl/OmniGen2)
- [Qwen-Image-Edit](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2509)

Thank you to the developers and communities of these excellent works.

