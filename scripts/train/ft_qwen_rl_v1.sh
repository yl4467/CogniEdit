# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

#source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
#conda activate py3.11+pytorch2.6+cu124

debug=false
RANK=0
MASTER_ADDR=1
MASTER_PORT=29600
WORLD_SIZE=1

# 处理命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)
            RANK="${1#*=}"
            shift
            ;;
        --master_addr=*)
            MASTER_ADDR="${1#*=}"
            shift
            ;;
        --master_port=*)
            MASTER_PORT="${1#*=}"
            shift
            ;;
        --world_size=*)
            WORLD_SIZE="${1#*=}"
            shift
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

num_processes=$(($WORLD_SIZE * 2))

echo "num_processes: $num_processes"
#export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:128"
export WANDB_MODE=disabled
export ACCELERATE_FSDP_REDUCE_BUCKET_SIZE=10000000000
export ACCELERATE_FSDP_ALLGATHER_BUCKET_SIZE=10000000000
export ACCELERATE_FSDP_GRADIENT_PREDIVIDE_FACTOR=1.0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_HIGH_PRIORITY=True
export CUDA_LAUNCH_BLOCKING=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
experiment_name=ft_qwen_rl

CUDA_VISIBLE_DEVICES=2,3 accelerate launch  \
--machine_rank=$RANK \
--main_process_ip=$MASTER_ADDR \
--main_process_port=$MASTER_PORT \
--num_machines=$WORLD_SIZE \
--num_processes=$num_processes \
--use_fsdp \
--fsdp_offload_params True \
--fsdp_sharding_strategy FULL_SHARD \
--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
--fsdp_transformer_layer_cls_to_wrap QwenImageTransformerBlock \
--fsdp_state_dict_type FULL_STATE_DICT \
--fsdp_forward_prefetch false \
--fsdp_use_orig_params True \
--fsdp_cpu_ram_efficient_loading True \
--fsdp_sync_module_states True \
train_qwen_grpo_v1.py --config options/${experiment_name}.yml