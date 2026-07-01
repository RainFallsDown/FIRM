#!/bin/bash

# BC + GRPO Training with Wan2.1 (Single GPU Test)

export PYTHONPATH=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/bc-grpo:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Activate conda environment
source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate sam3d-objects

# Log directory
LOG_DIR="${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/haimiandian_bc_grpo_wan21"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting BC+GRPO training with Wan2.1 (Single GPU)..."
echo "Log file: $LOG_FILE"

# Training command - Single GPU with aggressive optimization
torchrun --nproc_per_node=1 --master_port=29506   groot/vla/experiment/experiment.py   data=dreamzero/haimiandian   train_architecture=lora   wandb_project=haimiandian_bc_grpo_test   tokenizer_path=./checkpoints/umt5-xxl   dit_version=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P   text_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth   image_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   vae_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth   training_args.output_dir=$LOG_DIR   training_args.run_name=haimiandian_bc_grpo_wan21   training_args.report_to=none   training_args.per_device_train_batch_size=1   training_args.gradient_accumulation_steps=8   training_args.max_steps=100   training_args.save_steps=50   training_args.logging_steps=5   training_args.bf16=true   training_args.dataloader_num_workers=1   training_args.deepspeed=groot/vla/configs/deepspeed/zero3.json   use_grpo=true   grpo_weight=0.1   grpo_reward_scale=1.0   grpo_reward_shaper_scale=0.01   > "$LOG_FILE" 2>&1

echo "Training completed. Check log: $LOG_FILE"
