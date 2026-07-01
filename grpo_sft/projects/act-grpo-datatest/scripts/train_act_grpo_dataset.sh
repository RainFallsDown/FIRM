#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dataset_common.sh
source "$SCRIPT_DIR/dataset_common.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/train_act_grpo_dataset.sh DATASET_KEY [--dry-run] [--smoke]" >&2
  echo "For full runs set ACT_GRPO_PRETRAINED_PATH=/path/to/sft/checkpoint/pretrained_model" >&2
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

ACT_GRPO_STEPS="${ACT_GRPO_STEPS:-3500}"
ACT_BATCH_SIZE="${ACT_BATCH_SIZE:-16}"
ACT_NUM_WORKERS="${ACT_NUM_WORKERS:-8}"
ACT_SAVE_FREQ="${ACT_SAVE_FREQ:-500}"
ACT_LOG_FREQ="${ACT_LOG_FREQ:-20}"
ACT_CHUNK_SIZE="${ACT_CHUNK_SIZE:-30}"
ACT_N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-30}"
ACT_LR="${ACT_LR:-1e-5}"
ACT_PRETRAINED_BACKBONE_WEIGHTS="${ACT_PRETRAINED_BACKBONE_WEIGHTS:-null}"
ACT_GRPO_BETA="${ACT_GRPO_BETA:-1.0}"
ACT_GRPO_MIN_WEIGHT="${ACT_GRPO_MIN_WEIGHT:-0.5}"
ACT_GRPO_MAX_WEIGHT="${ACT_GRPO_MAX_WEIGHT:-2.0}"
ACT_GRPO_BC_REWARD_WEIGHT="${ACT_GRPO_BC_REWARD_WEIGHT:-0.45}"
ACT_GRPO_SMOOTH_REWARD_WEIGHT="${ACT_GRPO_SMOOTH_REWARD_WEIGHT:-0.25}"
ACT_GRPO_ACCEL_REWARD_WEIGHT="${ACT_GRPO_ACCEL_REWARD_WEIGHT:-0.20}"
ACT_GRPO_GRIPPER_REWARD_WEIGHT="${ACT_GRPO_GRIPPER_REWARD_WEIGHT:-0.10}"
ACT_GRPO_PRETRAINED_PATH="${ACT_GRPO_PRETRAINED_PATH:-}"

if [ -n "$ACT_GRPO_PRETRAINED_PATH" ]; then
  case "$ACT_GRPO_PRETRAINED_PATH" in
    /*)
      ;;
    *)
      ACT_GRPO_PRETRAINED_PATH="$ACT_GRPO_ROOT/$ACT_GRPO_PRETRAINED_PATH"
      ;;
  esac
  if [ "${ACT_GRPO_PRETRAINED_PATH##*/}" != "pretrained_model" ]; then
    ACT_GRPO_PRETRAINED_PATH="$ACT_GRPO_PRETRAINED_PATH/pretrained_model"
  fi
fi

if [ "$SMOKE" = "1" ]; then
  ACT_GRPO_STEPS=1
  ACT_BATCH_SIZE=2
  ACT_NUM_WORKERS=0
  ACT_SAVE_FREQ=1
  ACT_LOG_FREQ=1
fi

if [ "$DRY_RUN" = "0" ] && [ "$SMOKE" = "0" ]; then
  ensure_path_exists "$ACT_GRPO_PRETRAINED_PATH/model.safetensors" "SFT pretrained model"
fi

RUN_NAME="act_grpo_${DATASET_KEY}_$(timestamp)"
OUT_DIR="$ACT_GRPO_ROOT/results/grpo/$OUTPUT_GROUP/$RUN_NAME"
LOG_DIR="$ACT_GRPO_ROOT/logs/grpo/$OUTPUT_GROUP/$RUN_NAME"
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
  "--steps=$ACT_GRPO_STEPS"
  "--save_freq=$ACT_SAVE_FREQ"
  "--log_freq=$ACT_LOG_FREQ"
  "--eval_freq=0"
  "--save_checkpoint=true"
  "--seed=$SEED"
  "--output_dir=$OUT_DIR"
  "--job_name=$RUN_NAME"
  "--wandb.enable=$WANDB_ENABLE"
  "--wandb.project=act_grpo_datatest_grpo"
  "--use_grpo=true"
  "--grpo_beta=$ACT_GRPO_BETA"
  "--grpo_min_weight=$ACT_GRPO_MIN_WEIGHT"
  "--grpo_max_weight=$ACT_GRPO_MAX_WEIGHT"
  "--grpo_bc_reward_weight=$ACT_GRPO_BC_REWARD_WEIGHT"
  "--grpo_smooth_reward_weight=$ACT_GRPO_SMOOTH_REWARD_WEIGHT"
  "--grpo_accel_reward_weight=$ACT_GRPO_ACCEL_REWARD_WEIGHT"
  "--grpo_gripper_reward_weight=$ACT_GRPO_GRIPPER_REWARD_WEIGHT"
)

if [ -n "$ACT_GRPO_PRETRAINED_PATH" ]; then
  cmd+=("--policy.pretrained_path=$ACT_GRPO_PRETRAINED_PATH")
fi

echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
print_command "${cmd[@]}"
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
