#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACT_GRPO_ROOT="${ACT_GRPO_ROOT:-$(cd "$COMMON_DIR/.." && pwd)}"
FIRM_ROOT="${FIRM_ROOT:-$(cd "$ACT_GRPO_ROOT/../../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/path/to/miniconda3}"
ACT_GRPO_ENV="${ACT_GRPO_ENV:-$CONDA_ROOT/envs/act-grpo}"
LEROBOT_DIR="${LEROBOT_DIR:-$FIRM_ROOT}"
HF_HOME="${HF_HOME:-$ACT_GRPO_ROOT/hf_cache}"
TORCH_HOME="${TORCH_HOME:-$ACT_GRPO_ROOT/torch_cache}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
SEED="${SEED:-1000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

print_command() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

ensure_dir() {
  mkdir -p "$1"
}

ensure_path_exists() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "Missing $label: $path" >&2
    exit 2
  fi
}

load_dataset_config() {
  local dataset_key="$1"
  local config_path="$ACT_GRPO_ROOT/configs/datasets/${dataset_key}.env"
  ensure_path_exists "$config_path" "dataset config"
  # shellcheck source=/dev/null
  source "$config_path"
  export DATASET_KEY DATASET_REPO_ID DATASET_ROOT OUTPUT_GROUP
}

activate_act_grpo() {
  ensure_path_exists "$ACT_GRPO_ROOT/env/activate_act_grpo.sh" "activation script"
  # shellcheck source=/dev/null
  source "$ACT_GRPO_ROOT/env/activate_act_grpo.sh"
}

setup_lerobot_runtime() {
  ensure_path_exists "$LEROBOT_DIR/src/lerobot" "LeRobot source"
  export PYTHONPATH="$LEROBOT_DIR/src:${PYTHONPATH:-}"
  export ACT_GRPO_ROOT FIRM_ROOT ACT_GRPO_ENV LEROBOT_DIR HF_HOME TORCH_HOME VIDEO_BACKEND
  export WANDB_MODE WANDB_ENABLE CUDA_VISIBLE_DEVICES
  ensure_dir "$HF_HOME"
  ensure_dir "$TORCH_HOME"
}

require_lerobot_v3_dataset() {
  ensure_path_exists "$DATASET_ROOT/meta/info.json" "dataset info.json"
  ensure_path_exists "$DATASET_ROOT/meta/stats.json" "dataset stats.json"
  ensure_path_exists "$DATASET_ROOT/meta/tasks.parquet" "dataset tasks.parquet"
  ensure_path_exists "$DATASET_ROOT/data" "dataset data directory"
  ensure_path_exists "$DATASET_ROOT/videos" "dataset videos directory"
}

prepare_dataset_runtime() {
  local dataset_key="$1"
  load_dataset_config "$dataset_key"
  activate_act_grpo
  setup_lerobot_runtime
  require_lerobot_v3_dataset
}
