#!/usr/bin/env bash

CODE_DIR="${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero-grpo-pro"
SFT_CKPT="${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/wam_firm_sft_20260428/checkpoint-3500"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-/path/to/dreamzero}"
OUT_DIR="${OUT_DIR:-$DREAMZERO_ROOT/outputs/wam_firm_grpo_lingbot_from_sft3500_3500_${RUN_ID}}"
TRAIN_LOG="$OUT_DIR/training.log"
PIPELINE_LOG="$OUT_DIR/pipeline.log"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
COMPARE_DIR="$OUT_DIR/comparison_zero_sft_grpo_n${EVAL_SAMPLES}"
COMPARE_LOG="$COMPARE_DIR/evaluate.log"
GRPO_CKPT="$OUT_DIR/checkpoint-3500"

mkdir -p "$OUT_DIR" "$COMPARE_DIR"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$PIPELINE_LOG"
}

log "DreamZero GRPO 3500-step pipeline starting"
log "code_dir=$CODE_DIR"
log "sft_ckpt=$SFT_CKPT"
log "out_dir=$OUT_DIR"
log "eval_samples=$EVAL_SAMPLES"

cd "$CODE_DIR" || exit 1

export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate sam3d-objects

log "Running preflight checks"
${CONDA_ROOT:-/path/to/miniconda3}/envs/sam3d-objects/bin/python test_dream_grpo_lingbot_parity.py >> "$PIPELINE_LOG" 2>&1
PREFLIGHT_EC=$?
if [ "$PREFLIGHT_EC" -ne 0 ]; then
  log "Preflight failed with exit code $PREFLIGHT_EC"
  exit "$PREFLIGHT_EC"
fi

log "Training started"
torchrun --nproc_per_node=4 --master_port=29562 \
  groot/vla/experiment/experiment.py \
  data=dreamzero/haimiandian \
  train_architecture=lora \
  model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
  wandb_project=wam_firm_grpo_lingbot_sft3500 \
  tokenizer_path=./checkpoints/umt5-xxl \
  dit_version=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B \
  text_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
  image_encoder_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  vae_pretrained_path=${DREAMZERO_ROOT:-/path/to/dreamzero}/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  pretrained_model_path="$SFT_CKPT" \
  training_args.output_dir="$OUT_DIR" \
  training_args.run_name=wam_firm_grpo_lingbot_sft3500_3500 \
  training_args.report_to=none \
  training_args.per_device_train_batch_size=1 \
  training_args.gradient_accumulation_steps=2 \
  training_args.max_steps=3500 \
  training_args.save_steps=500 \
  training_args.logging_steps=10 \
  training_args.bf16=true \
  training_args.dataloader_num_workers=4 \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero3.json \
  use_grpo=true \
  grpo_weight=0.1 \
  grpo_reward_scale=1.0 \
  grpo_reward_shaper_scale=0.01 \
  grpo_buffer_size=32 \
  grpo_buffer_warmup_steps=4 \
  grpo_advantage_alpha=1.0 \
  grpo_weight_clamp_min=0.5 \
  grpo_weight_clamp_max=2.0 \
  > "$TRAIN_LOG" 2>&1
TRAIN_EC=$?
log "Training finished with exit code $TRAIN_EC"

if [ "$TRAIN_EC" -ne 0 ]; then
  log "Skip comparison because training failed"
  exit "$TRAIN_EC"
fi

if [ ! -d "$GRPO_CKPT" ]; then
  GRPO_CKPT="$(find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
fi

if [ ! -d "$GRPO_CKPT" ]; then
  log "No GRPO checkpoint found under $OUT_DIR"
  exit 2
fi

log "Comparison started with grpo_ckpt=$GRPO_CKPT"
export CUDA_VISIBLE_DEVICES=0
${CONDA_ROOT:-/path/to/miniconda3}/envs/sam3d-objects/bin/python \
  SFT_test/evaluate_zero_sft_grpo.py \
  --sft-model "$SFT_CKPT" \
  --grpo-model "$GRPO_CKPT" \
  --output-dir "$COMPARE_DIR" \
  --num-samples "$EVAL_SAMPLES" \
  --models zero_shot_same_config sft grpo \
  > "$COMPARE_LOG" 2>&1
COMPARE_EC=$?
log "Comparison finished with exit code $COMPARE_EC"

if [ "$COMPARE_EC" -eq 0 ]; then
  log "Comparison summary: $COMPARE_DIR/summary.md"
  log "Comparison metrics: $COMPARE_DIR/metrics.json"
fi

exit "$COMPARE_EC"
