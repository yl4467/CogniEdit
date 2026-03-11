# !/bin/bash
MODEL_PATH="/workspace/CogniEdit/models/Qwen-Image-Edit-2509"
TRANSFORMER_PATH="/workspace/CogniEdit/output/checkpoint-latest"
DATASET_PATH="/workspace/CogniEdit/data_configs/train/example/edit/edit_test.yml"  # HuggingFace dataset name or local path
TEXT_ENCODER_PATH="/workspace/CogniEdit/models/PeBR_R1/PeBR_R1_7B"

# Optional parameters
OUTPUT_DIR="/workspace/CogniEdit/output/results"
NUM_INFERENCE_STEPS=50
SEED=0
DTYPE="bf16"

CUDA_VISIBLE_DEVICES='0' python inference_r1.py \
    --model_path ${MODEL_PATH} \
    --transformer_path ${TRANSFORMER_PATH} \
    --dataset_path ${DATASET_PATH} \
    --text_encoder_path ${TEXT_ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --num_inference_step ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --dtype ${DTYPE} \
    --use_raw_image \
    --max_samples 100 