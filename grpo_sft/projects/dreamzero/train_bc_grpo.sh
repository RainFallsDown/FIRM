#!/bin/bash

# BC + GRPO Training Script for Haimiandian Dataset

export PYTHONPATH=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/bc-grpo:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Activate conda environment
source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate sam3d-objects

# Log directory
LOG_DIR="${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/haimiandian_bc_grpo"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting BC+GRPO training..."
echo "Log file: $LOG_FILE"

# Training command
torchrun --nproc_per_node=4 --master_port=29505   groot/vla/experiment/experiment.py   data=dreamzero/haimiandian   train_architecture=lora   wandb_project=haimiandian_bc_grpo   tokenizer_path=./checkpoints/umt5-xxl   dit_version=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B   text_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth   image_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   vae_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B/Wan2.1_VAE.pth   training_args.output_dir=$LOG_DIR   training_args.run_name=haimiandian_bc_grpo   training_args.report_to=none   training_args.per_device_train_batch_size=1   training_args.gradient_accumulation_steps=4   training_args.max_steps=5000   training_args.save_steps=500   training_args.logging_steps=10   training_args.bf16=true   training_args.dataloader_num_workers=1   training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json   use_grpo=true   grpo_weight=0.1   grpo_reward_scale=1.0   grpo_reward_shaper_scale=0.01   > "$LOG_FILE" 2>&1

echo "Training completed. Check log: $LOG_FILE"
