#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dataset_common.sh
source "$SCRIPT_DIR/dataset_common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/run_datatest_chain.sh DATASET_KEY... [--dry-run]

Environment overrides:
  CUDA_VISIBLE_DEVICES=0
  SFT_STEPS=3500
  GRPO_STEPS=3500
  ACT_BATCH_SIZE=16
  ACT_NUM_WORKERS=8
  ACT_SAVE_FREQ=500
  ACT_LOG_FREQ=20
  EVAL_SAMPLES=80
  EVAL_SEED=20260505
  EVAL_DEVICE=cuda
EOF
}

DRY_RUN=0
dataset_keys=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      dataset_keys+=("$1")
      ;;
  esac
  shift
done

if [ "${#dataset_keys[@]}" -eq 0 ]; then
  usage >&2
  exit 2
fi

SFT_STEPS="${SFT_STEPS:-3500}"
GRPO_STEPS="${GRPO_STEPS:-3500}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-16}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-8}"
ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-500}"
ACT_LOG_FREQ="${ACT_LOG_FREQ:-20}"
EVAL_SAMPLES="${EVAL_SAMPLES:-80}"
EVAL_SEED="${EVAL_SEED:-20260505}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

step_dir_name() {
  printf '%06d' "$1"
}

latest_checkpoint_for_step() {
  local base="$1"
  local step="$2"
  find "$base" -path "*/checkpoints/$(step_dir_name "$step")/pretrained_model" -type d | sort | tail -1
}

run_dataset_chain() {
  local dataset_key="$1"
  prepare_dataset_runtime "$dataset_key"

  echo "CHAIN_DATASET=$dataset_key"
  echo "CHAIN_DATASET_ROOT=$DATASET_ROOT"
  echo "CHAIN_OUTPUT_GROUP=$OUTPUT_GROUP"

  if [ "$DRY_RUN" = "1" ]; then
    echo "Would run: ACT_STEPS=$SFT_STEPS bash scripts/train_act_bc_dataset.sh $dataset_key"
    echo "Would run: ACT_GRPO_STEPS=$GRPO_STEPS ACT_GRPO_PRETRAINED_PATH=<sft_ckpt> bash scripts/train_act_grpo_dataset.sh $dataset_key"
    echo "Would run: scripts/compare_act_zero_sft_grpo_offline.py for $dataset_key"
    echo "CHAIN_DRY_RUN_OK dataset=$dataset_key"
    return 0
  fi

  bash "$SCRIPT_DIR/preflight_dataset.sh" "$dataset_key"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ACT_STEPS="$SFT_STEPS" \
  ACT_BATCH_SIZE="$ACT_BATCH_SIZE" \
  ACT_NUM_WORKERS="$ACT_NUM_WORKERS" \
  ACT_SAVE_FREQ="$ACT_SAVE_FREQ" \
  ACT_LOG_FREQ="$ACT_LOG_FREQ" \
    bash "$SCRIPT_DIR/train_act_bc_dataset.sh" "$dataset_key"

  local sft_ckpt
  sft_ckpt="$(latest_checkpoint_for_step "$ACT_GRPO_ROOT/results/baselines/$OUTPUT_GROUP" "$SFT_STEPS")"
  ensure_path_exists "$sft_ckpt/model.safetensors" "SFT checkpoint model"
  echo "CHAIN_SFT_CKPT=$sft_ckpt"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ACT_GRPO_PRETRAINED_PATH="$sft_ckpt" \
  ACT_GRPO_STEPS="$GRPO_STEPS" \
  ACT_BATCH_SIZE="$ACT_BATCH_SIZE" \
  ACT_NUM_WORKERS="$ACT_NUM_WORKERS" \
  ACT_SAVE_FREQ="$ACT_SAVE_FREQ" \
  ACT_LOG_FREQ="$ACT_LOG_FREQ" \
    bash "$SCRIPT_DIR/train_act_grpo_dataset.sh" "$dataset_key"

  local grpo_ckpt
  grpo_ckpt="$(latest_checkpoint_for_step "$ACT_GRPO_ROOT/results/grpo/$OUTPUT_GROUP" "$GRPO_STEPS")"
  ensure_path_exists "$grpo_ckpt/model.safetensors" "GRPO checkpoint model"
  echo "CHAIN_GRPO_CKPT=$grpo_ckpt"

  local eval_dir
  eval_dir="$ACT_GRPO_ROOT/results/eval/$OUTPUT_GROUP/zero_sft_grpo_$(timestamp)"
  ensure_dir "$eval_dir"
  PYTHONPATH="$LEROBOT_DIR/src:${PYTHONPATH:-}" python "$SCRIPT_DIR/compare_act_zero_sft_grpo_offline.py" \
    --zero-shot zero="$sft_ckpt" \
    --checkpoint "sft${SFT_STEPS}=$sft_ckpt" \
    --checkpoint "grpo${GRPO_STEPS}=$grpo_ckpt" \
    --dataset-root "$DATASET_ROOT" \
    --num-samples "$EVAL_SAMPLES" \
    --seed "$EVAL_SEED" \
    --device "$EVAL_DEVICE" \
    --output-dir "$eval_dir"

  echo "CHAIN_EVAL_DIR=$eval_dir"
  echo "CHAIN_DONE dataset=$dataset_key"
}

for dataset_key in "${dataset_keys[@]}"; do
  run_dataset_chain "$dataset_key"
done

echo "DATATEST_CHAIN_DONE datasets=${dataset_keys[*]}"
