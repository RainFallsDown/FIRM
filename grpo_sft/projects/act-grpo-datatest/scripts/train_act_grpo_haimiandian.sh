#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=haimiandian_baseline_common.sh
source "$SCRIPT_DIR/haimiandian_baseline_common.sh"

DRY_RUN=0
SMOKE=0

usage() {
  cat <<'EOF'
Usage: scripts/train_act_grpo_haimiandian.sh [--dry-run] [--smoke]

Environment overrides:
  CUDA_VISIBLE_DEVICES=0
  ACT_GRPO_STEPS=3500
  ACT_BATCH_SIZE=16
  ACT_NUM_WORKERS=8
  ACT_GRPO_BETA=1.0
  ACT_GRPO_MIN_WEIGHT=0.5
  ACT_GRPO_MAX_WEIGHT=2.0
  ACT_GRPO_PRETRAINED_PATH=/path/to/act-bc/checkpoint
EOF
}

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

prepare_baseline_runtime

ACT_GRPO_STEPS="${ACT_GRPO_STEPS:-3500}"
ACT_GRPO_BETA="${ACT_GRPO_BETA:-1.0}"
ACT_GRPO_MIN_WEIGHT="${ACT_GRPO_MIN_WEIGHT:-0.5}"
ACT_GRPO_MAX_WEIGHT="${ACT_GRPO_MAX_WEIGHT:-2.0}"
ACT_GRPO_BC_REWARD_WEIGHT="${ACT_GRPO_BC_REWARD_WEIGHT:-0.45}"
ACT_GRPO_SMOOTH_REWARD_WEIGHT="${ACT_GRPO_SMOOTH_REWARD_WEIGHT:-0.25}"
ACT_GRPO_ACCEL_REWARD_WEIGHT="${ACT_GRPO_ACCEL_REWARD_WEIGHT:-0.20}"
ACT_GRPO_GRIPPER_REWARD_WEIGHT="${ACT_GRPO_GRIPPER_REWARD_WEIGHT:-0.10}"
ACT_GRPO_PRETRAINED_PATH="${ACT_GRPO_PRETRAINED_PATH:-}"

if [ "$SMOKE" = "1" ]; then
  ACT_GRPO_STEPS=1
  ACT_BATCH_SIZE=2
  ACT_NUM_WORKERS=0
  ACT_SAVE_FREQ=1
  ACT_LOG_FREQ=1
fi

RUN_NAME="act_grpo_$(timestamp)"
OUT_DIR="$ACT_GRPO_ROOT/results/grpo/haimiandian/$RUN_NAME"
LOG_DIR="$ACT_GRPO_ROOT/logs/grpo/haimiandian/$RUN_NAME"
LOG_FILE="$LOG_DIR/train.log"
ensure_dir "$LOG_DIR"

cd "$LEROBOT_DIR"

cmd=(
  env
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "PYTHONPATH=$LEROBOT_DIR/src:${PYTHONPATH:-}"
  "HF_HOME=$HF_HOME"
  "TORCH_HOME=$TORCH_HOME"
  "WANDB_MODE=${WANDB_MODE:-disabled}"
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
  "--wandb.project=act_grpo_haimiandian"
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

if [ "$DRY_RUN" = "1" ]; then
  echo "OUT_DIR=$OUT_DIR"
  echo "LOG_FILE=$LOG_FILE"
  run_or_dry_run 1 "${cmd[@]}"
  exit 0
fi

echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
print_command "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
