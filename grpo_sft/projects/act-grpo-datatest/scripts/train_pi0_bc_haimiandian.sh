#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=haimiandian_baseline_common.sh
source "$SCRIPT_DIR/haimiandian_baseline_common.sh"

DRY_RUN=0
SMOKE=0

usage() {
  cat <<'EOF'
Usage: scripts/train_pi0_bc_haimiandian.sh [--dry-run] [--smoke] [--policy pi0|pi05]

Default policy is PI_POLICY=pi0.

Important pretrained model handling:
  PI0_PRETRAINED_PATH=lerobot/pi0_base
  PI05_PRETRAINED_PATH=lerobot/pi05_base

Because the remote machine may not reach HuggingFace reliably, actual runs require one of:
  1. set PI0_PRETRAINED_PATH or PI05_PRETRAINED_PATH to a local model directory;
  2. have the HuggingFace model already cached;
  3. use ALLOW_REMOTE_PRETRAINED=1 to permit network download;
  4. use ALLOW_RANDOM_INIT=1 and an empty pretrained path for smoke tests only.
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
    --policy)
      shift
      PI_POLICY="${1:-}"
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

case "$PI_POLICY" in
  pi0|pi05)
    ;;
  *)
    echo "PI_POLICY must be pi0 or pi05, got: $PI_POLICY" >&2
    exit 2
    ;;
esac

prepare_baseline_runtime

if [ "$SMOKE" = "1" ]; then
  PI_STEPS=1
  PI_BATCH_SIZE=1
  PI_NUM_WORKERS=0
  PI_SAVE_FREQ=1
  PI_COMPILE_MODEL=false
fi

PI_PRETRAINED_PATH="$(resolve_pi_pretrained_path "$PI_POLICY")"
if [ "$DRY_RUN" != "1" ]; then
  validate_pi_pretrained_for_run "$PI_PRETRAINED_PATH"
fi

RUN_NAME="${PI_POLICY}_bc_$(timestamp)"
OUT_DIR="$OUTPUT_BASE/$RUN_NAME"
LOG_DIR="$LOG_BASE/$RUN_NAME"
LOG_FILE="$LOG_DIR/train.log"
ensure_dir "$LOG_DIR"

cd "$LEROBOT_DIR"

cmd=(
  env
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "PYTHONPATH=$LEROBOT_DIR/src:${PYTHONPATH:-}"
  "HF_HOME=$HF_HOME"
  "TORCH_HOME=$TORCH_HOME"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
  "WANDB_MODE=${WANDB_MODE:-disabled}"
  lerobot-train
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.root=$DATASET_ROOT"
  "--dataset.video_backend=$VIDEO_BACKEND"
  "--dataset.image_transforms.enable=true"
  "--policy.type=$PI_POLICY"
  "--policy.device=cuda"
  "--policy.push_to_hub=false"
  "--policy.dtype=$PI_DTYPE"
  "--policy.gradient_checkpointing=$PI_GRADIENT_CHECKPOINTING"
  "--policy.compile_model=$PI_COMPILE_MODEL"
  "--policy.freeze_vision_encoder=$PI_FREEZE_VISION_ENCODER"
  "--policy.train_expert_only=$PI_TRAIN_EXPERT_ONLY"
  "--policy.normalization_mapping={\"ACTION\":\"MEAN_STD\",\"STATE\":\"MEAN_STD\",\"VISUAL\":\"IDENTITY\"}"
  "--policy.optimizer_lr=$PI_LR"
  "--policy.chunk_size=$PI_CHUNK_SIZE"
  "--policy.n_action_steps=$PI_N_ACTION_STEPS"
  "--policy.tokenizer_max_length=$PI_TOKENIZER_MAX_LENGTH"
  "--batch_size=$PI_BATCH_SIZE"
  "--num_workers=$PI_NUM_WORKERS"
  "--steps=$PI_STEPS"
  "--save_freq=$PI_SAVE_FREQ"
  "--log_freq=$PI_LOG_FREQ"
  "--eval_freq=0"
  "--save_checkpoint=true"
  "--seed=$SEED"
  "--output_dir=$OUT_DIR"
  "--job_name=$RUN_NAME"
  "--wandb.enable=$WANDB_ENABLE"
  "--wandb.project=$WANDB_PROJECT"
)

if [ -n "$PI_PRETRAINED_PATH" ]; then
  cmd+=("--policy.pretrained_path=$PI_PRETRAINED_PATH")
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "OUT_DIR=$OUT_DIR"
  echo "LOG_FILE=$LOG_FILE"
  echo "PI_PRETRAINED_PATH=${PI_PRETRAINED_PATH:-<random-init>}"
  run_or_dry_run 1 "${cmd[@]}"
  exit 0
fi

echo "OUT_DIR=$OUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "PI_PRETRAINED_PATH=${PI_PRETRAINED_PATH:-<random-init>}"
print_command "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
