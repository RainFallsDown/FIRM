#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

cd ${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero
source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate sam3d-objects

LOG_DIR="${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/haimiandian_lora"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node=4 --master_port=29503   groot/vla/experiment/experiment.py   data=dreamzero/haimiandian   train_architecture=lora   wandb_project=haimiandian_bc   tokenizer_path=./checkpoints/umt5-xxl   text_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth   image_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth   vae_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth   training_args.output_dir=${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/haimiandian_lora   training_args.report_to=none   training_args.per_device_train_batch_size=1   > "$LOG_FILE" 2>&1

echo "Training completed. Log file: $LOG_FILE"
