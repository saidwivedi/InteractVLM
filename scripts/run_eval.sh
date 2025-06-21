#!/bin/bash

# Set environment variables
export PATH=$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export HF_HOME='.cache/huggingface'
# export HF_HOME='/is/cluster/sdwivedi/.cache/huggingface'


# generate random number between 20000 and 25000
MASTER_PORT=$(( ( RANDOM % 5000 )  + 20000 ))
VISION_PRETRAINED="../pretrained_models/sam_vit_h_4b8939.pth"


get_gpu_memory() {
    # Query GPU memory info using nvidia-smi
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo $gpu_memory
}
gpu_memory=$(get_gpu_memory)
gpu_memory_in_gb=$(($gpu_memory / 1024))


# Function to set configuration based on numerical argument
set_configuration() {
  case $1 in
    0)
      EXP_NAME="eval_damon_hcontact"
      VERSION="trained_models/DAMON_LLaVA-13B"
      LOG_WANDB="False"
      VAL_DATASET="damon_hcontact"
      DISP_SIZE="128"
      ;;
    *)
      echo "Unknown configuration number: $1"
      exit 1
      ;;
  esac
}

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Please provide a configuration number (e.g., 0, 1)."
  exit 1
fi

# Set the configuration based on the numerical argument
set_configuration $1


deepspeed \
  --master_port=$MASTER_PORT evaluate.py \
  --version=$VERSION \
  --log_wandb=$LOG_WANDB \
  --val_dataset=$VAL_DATASET \
  --disp_size=$DISP_SIZE \