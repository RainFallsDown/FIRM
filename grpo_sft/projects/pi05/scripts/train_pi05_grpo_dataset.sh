#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dataset_common.sh
source "$SCRIPT_DIR/dataset_common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/train_pi05_grpo_dataset.sh DATASET_KEY [--dry-run] [--smoke]

First PI05-GRPO stage: offline batch intrinsic reward-weighted BC.
This script reuses the existing LeRobot --use_grpo training-loop path.

Environment overrides:
  CUDA_VISIBLE_DEVICES=2
  PI05_GRPO_PRETRAINED_PATH=/path/to/pi05_sft/checkpoints/003500/pretrained_model
  PI05_GRPO_INIT_STEP=3500
  PI05_GRPO_STEPS=3500
  PI05_GRPO_BATCH_SIZE=2
  PI05_GRPO_NUM_WORKERS=0
  PI05_GRPO_SAVE_FREQ=500
  PI05_GRPO_LOG_FREQ=20
  PI05_GRPO_BETA=1.0
  PI05_GRPO_MIN_WEIGHT=0.5
  PI05_GRPO_MAX_WEIGHT=2.0
EOF
}

if [ "$#" -lt 1 ]; then
  usage >&2
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

prepare_dataset_runtime "$DATASET_KEY_ARG"

PI05_GRPO_INIT_STEP="${PI05_GRPO_INIT_STEP:-3500}"
PI05_GRPO_STEPS="${PI05_GRPO_STEPS:-3500}"
PI05_GRPO_BATCH_SIZE="${PI05_GRPO_BATCH_SIZE:-2}"
PI05_GRPO_NUM_WORKERS="${PI05_GRPO_NUM_WORKERS:-0}"
PI05_GRPO_SAVE_FREQ="${PI05_GRPO_SAVE_FREQ:-500}"
PI05_GRPO_LOG_FREQ="${PI05_GRPO_LOG_FREQ:-20}"
PI05_GRPO_CHUNK_SIZE="${PI05_GRPO_CHUNK_SIZE:-30}"
PI05_GRPO_N_ACTION_STEPS="${PI05_GRPO_N_ACTION_STEPS:-30}"
PI05_GRPO_DTYPE="${PI05_GRPO_DTYPE:-bfloat16}"
PI05_GRPO_LR="${PI05_GRPO_LR:-2.5e-5}"
PI05_GRPO_TOKENIZER_MAX_LENGTH="${PI05_GRPO_TOKENIZER_MAX_LENGTH:-200}"
PI05_GRPO_GRADIENT_CHECKPOINTING="${PI05_GRPO_GRADIENT_CHECKPOINTING:-true}"
PI05_GRPO_COMPILE_MODEL="${PI05_GRPO_COMPILE_MODEL:-false}"
PI05_GRPO_FREEZE_VISION_ENCODER="${PI05_GRPO_FREEZE_VISION_ENCODER:-true}"
PI05_GRPO_TRAIN_EXPERT_ONLY="${PI05_GRPO_TRAIN_EXPERT_ONLY:-true}"

PI05_GRPO_BETA="${PI05_GRPO_BETA:-1.0}"
PI05_GRPO_MIN_WEIGHT="${PI05_GRPO_MIN_WEIGHT:-0.5}"
PI05_GRPO_MAX_WEIGHT="${PI05_GRPO_MAX_WEIGHT:-2.0}"
PI05_GRPO_BC_REWARD_WEIGHT="${PI05_GRPO_BC_REWARD_WEIGHT:-0.45}"
PI05_GRPO_SMOOTH_REWARD_WEIGHT="${PI05_GRPO_SMOOTH_REWARD_WEIGHT:-0.25}"
PI05_GRPO_ACCEL_REWARD_WEIGHT="${PI05_GRPO_ACCEL_REWARD_WEIGHT:-0.20}"
PI05_GRPO_GRIPPER_REWARD_WEIGHT="${PI05_GRPO_GRIPPER_REWARD_WEIGHT:-0.10}"

step_dir_name() {
  printf "%06d" "$1"
}

latest_pi05_sft_checkpoint() {
  local step_name="$1"
  find "$ACT_GRPO_ROOT/results/baselines/$OUTPUT_GROUP" \
    -path "*/checkpoints/$step_name/pretrained_model" \
    -type d 2>/dev/null | sort | tail -1
}

PI05_GRPO_PRETRAINED_PATH="${PI05_GRPO_PRETRAINED_PATH:-}"
if [ -z "$PI05_GRPO_PRETRAINED_PATH" ]; then
  PI05_GRPO_PRETRAINED_PATH="$(latest_pi05_sft_checkpoint "$(step_dir_name "$PI05_GRPO_INIT_STEP")")"
fi

if [ -n "$PI05_GRPO_PRETRAINED_PATH" ]; then
  case "$PI05_GRPO_PRETRAINED_PATH" in
    /*)
      ;;
    *)
      PI05_GRPO_PRETRAINED_PATH="$ACT_GRPO_ROOT/$PI05_GRPO_PRETRAINED_PATH"
      ;;
  esac
  if [ "${PI05_GRPO_PRETRAINED_PATH##*/}" != "pretrained_model" ]; then
    PI05_GRPO_PRETRAINED_PATH="$PI05_GRPO_PRETRAINED_PATH/pretrained_model"
  fi
fi

if [ "$SMOKE" = "1" ]; then
  PI05_GRPO_STEPS=1
  PI05_GRPO_BATCH_SIZE=2
  PI05_GRPO_NUM_WORKERS=0
  PI05_GRPO_SAVE_FREQ=1
  PI05_GRPO_LOG_FREQ=1
fi

if [ "$DRY_RUN" = "0" ]; then
  ensure_path_exists "$PI05_GRPO_PRETRAINED_PATH/model.safetensors" "PI05 SFT pretrained model"
fi

RUN_NAME="pi05_grpo_${DATASET_KEY}_$(timestamp)"
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
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
  "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}"
  "WANDB_MODE=$WANDB_MODE"
  lerobot-train
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.root=$DATASET_ROOT"
  "--dataset.video_backend=$VIDEO_BACKEND"
  "--dataset.image_transforms.enable=true"
  "--policy.type=pi05"
  "--policy.device=cuda"
  "--policy.push_to_hub=false"
  "--policy.dtype=$PI05_GRPO_DTYPE"
  "--policy.gradient_checkpointing=$PI05_GRPO_GRADIENT_CHECKPOINTING"
  "--policy.compile_model=$PI05_GRPO_COMPILE_MODEL"
  "--policy.freeze_vision_encoder=$PI05_GRPO_FREEZE_VISION_ENCODER"
  "--policy.train_expert_only=$PI05_GRPO_TRAIN_EXPERT_ONLY"
  "--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}"
  "--policy.optimizer_lr=$PI05_GRPO_LR"
  "--policy.chunk_size=$PI05_GRPO_CHUNK_SIZE"
  "--policy.n_action_steps=$PI05_GRPO_N_ACTION_STEPS"
  "--policy.tokenizer_max_length=$PI05_GRPO_TOKENIZER_MAX_LENGTH"
  "--policy.pretrained_path=$PI05_GRPO_PRETRAINED_PATH"
  "--batch_size=$PI05_GRPO_BATCH_SIZE"
  "--num_workers=$PI05_GRPO_NUM_WORKERS"
  "--steps=$PI05_GRPO_STEPS"
  "--save_freq=$PI05_GRPO_SAVE_FREQ"
  "--log_freq=$PI05_GRPO_LOG_FREQ"
  "--eval_freq=0"
  "--save_checkpoint=true"
  "--seed=$SEED"
  "--output_dir=$OUT_DIR"
  "--job_name=$RUN_NAME"
  "--wandb.enable=$WANDB_ENABLE"
  "--wandb.project=pi05_grpo_datatest"
  "--use_grpo=true"
  "--grpo_beta=$PI05_GRPO_BETA"
  "--grpo_min_weight=$PI05_GRPO_MIN_WEIGHT"
  "--grpo_max_weight=$PI05_GRPO_MAX_WEIGHT"
  "--grpo_bc_reward_weight=$PI05_GRPO_BC_REWARD_WEIGHT"
  "--grpo_smooth_reward_weight=$PI05_GRPO_SMOOTH_REWARD_WEIGHT"
  "--grpo_accel_reward_weight=$PI05_GRPO_ACCEL_REWARD_WEIGHT"
  "--grpo_gripper_reward_weight=$PI05_GRPO_GRIPPER_REWARD_WEIGHT"
)

echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "PI05_GRPO_PRETRAINED_PATH=$PI05_GRPO_PRETRAINED_PATH"
print_command "${cmd[@]}"
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
