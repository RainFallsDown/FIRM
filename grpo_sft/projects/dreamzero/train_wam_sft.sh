#!/bin/bash
# WAM SFT on FIRM dataset - BC only (no GRPO)

export PYTHONPATH=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/bc-grpo-wam-20260428:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate sam3d-objects

LOG_DIR="${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/wam_firm_sft_20260428"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "WAM SFT on FIRM Dataset" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Model: Wan2.2-TI2V-5B" | tee -a "$LOG_FILE"
echo "Dataset: FIRM (1,718 samples)" | tee -a "$LOG_FILE"
echo "Training: BC only (no GRPO)" | tee -a "$LOG_FILE"
echo "Hardware: 4×H100 80GB" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

torchrun --nproc_per_node=4 --master_port=29508 \
  groot/vla/experiment/experiment.py \
  data=dreamzero/haimiandian \
  train_architecture=lora \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
  wandb_project=wam_firm_sft \
  tokenizer_path=./checkpoints/umt5-xxl \
  dit_version=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B \
  text_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
  image_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  vae_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  training_args.output_dir=$LOG_DIR \
  training_args.run_name=wam_firm_sft \
  training_args.report_to=none \
  training_args.per_device_train_batch_size=1 \
  training_args.gradient_accumulation_steps=8 \
  training_args.max_steps=5000 \
  training_args.save_steps=500 \
  training_args.logging_steps=10 \
  training_args.bf16=true \
  training_args.dataloader_num_workers=4 \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero3.json \
  use_grpo=false \
  >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "========================================" | tee -a "$LOG_FILE"
echo "Training finished at: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
