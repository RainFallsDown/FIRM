#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ACT_GRPO_ROOT="$(cd "$COMMON_DIR/.." && pwd)"
BASELINE_CONFIG="${BASELINE_CONFIG:-$LOCAL_ACT_GRPO_ROOT/configs/baselines/haimiandian_baseline.env}"

if [ -f "$BASELINE_CONFIG" ]; then
  # shellcheck source=/dev/null
  source "$BASELINE_CONFIG"
else
  echo "Missing baseline config: $BASELINE_CONFIG" >&2
  exit 2
fi

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

activate_act_grpo() {
  ensure_path_exists "$ACT_GRPO_ROOT/env/activate_act_grpo.sh" "activation script"
  # shellcheck source=/dev/null
  source "$ACT_GRPO_ROOT/env/activate_act_grpo.sh"
}

setup_lerobot_copy() {
  ensure_path_exists "$LEROBOT_DIR/src/lerobot" "copied LeRobot source"
  export PYTHONPATH="$LEROBOT_DIR/src:${PYTHONPATH:-}"
  export ACT_GRPO_ROOT
  export ACT_GRPO_ENV
  export LEROBOT_DIR
  export DATASET_ROOT
  export DATASET_REPO_ID
  export OUTPUT_BASE
  export LOG_BASE
  export VIDEO_BACKEND
  export CUDA_VISIBLE_DEVICES
  export HF_HOME
  export TORCH_HOME="${TORCH_HOME:-$ACT_GRPO_ROOT/torch_cache}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  ensure_dir "$OUTPUT_BASE"
  ensure_dir "$LOG_BASE"
  ensure_dir "$HF_HOME"
  ensure_dir "$TORCH_HOME"
}

require_baseline_inputs() {
  ensure_path_exists "$DATASET_ROOT/meta/info.json" "haimiandian dataset info"
  ensure_path_exists "$DATASET_ROOT/meta/episodes.jsonl" "haimiandian episodes"
  ensure_path_exists "$DATASET_ROOT/data" "haimiandian data directory"
  ensure_path_exists "$DATASET_ROOT/videos" "haimiandian videos directory"
}

run_or_dry_run() {
  local dry_run="$1"
  shift
  print_command "$@"
  if [ "$dry_run" = "1" ]; then
    return 0
  fi
  "$@"
}

repo_cache_name() {
  local repo_id="$1"
  printf 'models--%s\n' "${repo_id//\//--}"
}

find_hf_model_cache() {
  local repo_id="$1"
  local cache_name
  cache_name="$(repo_cache_name "$repo_id")"
  local candidates=(
    "$HF_HOME/hub/$cache_name"
    "$ACT_GRPO_ROOT/hf_cache/hub/$cache_name"
    "$HOME/.cache/huggingface/hub/$cache_name"
    "${CACHE_ROOT:-/path/to/cache}/huggingface/hub/$cache_name"
    "/root/.cache/huggingface/hub/$cache_name"
  )
  local path
  for path in "${candidates[@]}"; do
    if [ -d "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

resolve_pi_pretrained_path() {
  local policy="$1"
  if [ "$policy" = "pi05" ]; then
    printf '%s\n' "$PI05_PRETRAINED_PATH"
  else
    printf '%s\n' "$PI0_PRETRAINED_PATH"
  fi
}

validate_pi_pretrained_for_run() {
  local pretrained_path="$1"
  if [ -z "$pretrained_path" ]; then
    if [ "$ALLOW_RANDOM_INIT" = "1" ]; then
      echo "PI baseline will use random initialization because ALLOW_RANDOM_INIT=1."
      return 0
    fi
    echo "PI_PRETRAINED_PATH is empty. Set ALLOW_RANDOM_INIT=1 only for smoke tests." >&2
    exit 2
  fi

  if [ -d "$pretrained_path" ]; then
    return 0
  fi

  if find_hf_model_cache "$pretrained_path" >/dev/null; then
    export HF_HUB_OFFLINE=1
    return 0
  fi

  if [ "$ALLOW_REMOTE_PRETRAINED" = "1" ]; then
    echo "No local cache found for $pretrained_path; ALLOW_REMOTE_PRETRAINED=1 so LeRobot may try network download."
    return 0
  fi

  cat >&2 <<EOF
Missing local pretrained model for $pretrained_path.

Current remote network is unreliable/offline. Copy the PI0/PI05 model into a local directory under:
  $ACT_GRPO_ROOT/resources/models/

Then run with, for example:
  PI0_PRETRAINED_PATH=$ACT_GRPO_ROOT/resources/models/pi0_base scripts/train_pi0_bc_haimiandian.sh

For a non-meaningful smoke test only, use:
  ALLOW_RANDOM_INIT=1 PI0_PRETRAINED_PATH= scripts/train_pi0_bc_haimiandian.sh --smoke
EOF
  exit 2
}

prepare_baseline_runtime() {
  activate_act_grpo
  setup_lerobot_copy
  require_baseline_inputs
}
