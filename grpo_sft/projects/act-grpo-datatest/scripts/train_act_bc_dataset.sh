#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dataset_common.sh
source "$SCRIPT_DIR/dataset_common.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/train_act_bc_dataset.sh DATASET_KEY [--dry-run] [--smoke]" >&2
  exit 2
fi

DATASET_KEY_ARG="$1"
shift
DRY_RUN=0
SMOKE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --smoke)
      SMOKE=1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

prepare_dataset_runtime "$DATASET_KEY_ARG"

ACT_STEPS="${ACT_STEPS:-3500}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-16}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-8}"
ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-500}"
ACT_LOG_FREQ="${ACT_LOG_FREQ:-20}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-30}"
ACT_N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-30}"
ACT_LR="${ACT_LR:-1e-5}"
ACT_PRETRAINED_BACKBONE_WEIGHTS="${ACT_PRETRAINED_BACKBONE_WEIGHTS:-null}"

if [ "$SMOKE" = "1" ]; then
  ACT_STEPS=1
  ACT_BATCH_SIZE=1
  ACT_NUM_WORKERS=0
  ACT_SAVE_FREQ=1
  ACT_LOG_FREQ=1
fi

RUN_NAME="act_bc_${DATASET_KEY}_$(timestamp)"
OUT_DIR="$ACT_GRPO_ROOT/results/baselines/$OUTPUT_GROUP/$RUN_NAME"
LOG_DIR="$ACT_GRPO_ROOT/logs/baselines/$OUTPUT_GROUP/$RUN_NAME"
LOG_FILE="$LOG_DIR/train.log"
ensure_dir "$LOG_DIR"

cd "$LEROBOT_DIR"
cmd=(
  env
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "PYTHONPATH=$LEROBOT_DIR/src:${PYTHONPATH:-}"
  "HF_HOME=$HF_HOME"
  "TORCH_HOME=$TORCH_HOME"
  "WANDB_MODE=$WANDB_MODE"
  lerobot-train
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.root=$DATASET_ROOT"
  "--dataset.video_backend=$VIDEO_BACKEND"
  "--dataset.image_transforms.enable=true"
  "--policy.type=act"
  "--policy.device=cuda"
  "--policy.push_to_hub=false"
  "--policy.chunk_size=$ACT_CHUNK_SIZE"
  "--policy.n_action_steps=$ACT_N_ACTION_STEPS"
  "--policy.optimizer_lr=$ACT_LR"
  "--policy.pretrained_backbone_weights=$ACT_PRETRAINED_BACKBONE_WEIGHTS"
  "--batch_size=$ACT_BATCH_SIZE"
  "--num_workers=$ACT_NUM_WORKERS"
  "--steps=$ACT_STEPS"
  "--save_freq=$ACT_SAVE_FREQ"
  "--log_freq=$ACT_LOG_FREQ"
  "--eval_freq=0"
  "--save_checkpoint=true"
  "--seed=$SEED"
  "--output_dir=$OUT_DIR"
  "--job_name=$RUN_NAME"
  "--wandb.enable=$WANDB_ENABLE"
  "--wandb.project=act_grpo_datatest_sft"
)

echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
print_command "${cmd[@]}"
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
