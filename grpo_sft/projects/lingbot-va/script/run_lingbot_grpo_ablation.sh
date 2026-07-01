#!/usr/bin/env bash
set -euo pipefail

USE_GRPO=${1:?use_grpo true/false}
REWARD_SCALE=${2:?reward scale}
ALPHA=${3:?advantage alpha}
BUFFER_SIZE=${4:?buffer size}
NUM_STEPS=${5:?num steps}
OUTPUT_NAME=${6:?output name}

ROOT=${LINGBOT_ROOT:-/path/to/lingbot-va}
CODE=$ROOT/lingbot-va
OUT_ROOT=$ROOT/outputs/haimiandian_lingbot_grpo_ablation_20260503
OUT=$OUT_ROOT/$OUTPUT_NAME
LOG=$OUT/logs/nohup_grpo.log

# Default: latest haimiandian SFT checkpoint (override with LINGBOT_GRPO_RESUME_FROM).
DEFAULT_GRPO_RESUME="$ROOT/outputs/haimiandian_sft_resume1000_to_eq5000_20260503_030524/sft/checkpoints/checkpoint_step_3500"
export LINGBOT_GRPO_RESUME_FROM="${LINGBOT_GRPO_RESUME_FROM:-$DEFAULT_GRPO_RESUME}"

mkdir -p "$OUT/logs"
set +u
source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate lingbot-va-torch29
set -u
cd "$CODE"

export PYTHONPATH="$CODE:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
# Do not disable NCCL monitoring: watchdog + flight recorder help catch hangs.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Heartbeat / op watchdog (seconds). Full FSDP steps + /share I/O can exceed 10m default.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
# torch.distributed default PG timeout (must match util.py LINGBOT_DIST_TIMEOUT_SEC intent)
export LINGBOT_DIST_TIMEOUT_SEC="${LINGBOT_DIST_TIMEOUT_SEC:-7200}"
# NCCL collective timeout (seconds; library-dependent, safe upper bound for long steps)
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
# If you still see NCCL hangs on a single node, try: export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/lingbot_triton_cache_${OUTPUT_NAME}"

export LINGBOT_GRPO_USE_GRPO="$USE_GRPO"
export LINGBOT_GRPO_REWARD_SCALE="$REWARD_SCALE"
export LINGBOT_GRPO_ALPHA="$ALPHA"
export LINGBOT_GRPO_BUFFER_SIZE="$BUFFER_SIZE"
export LINGBOT_GRPO_NUM_STEPS="$NUM_STEPS"
export LINGBOT_GRPO_SAVE_INTERVAL="${LINGBOT_GRPO_SAVE_INTERVAL:-500}"
export LINGBOT_GRPO_GRAD_ACCUM="${LINGBOT_GRPO_GRAD_ACCUM:-2}"
export LINGBOT_GRPO_LOAD_WORKER="${LINGBOT_GRPO_LOAD_WORKER:-4}"
export LINGBOT_GRPO_WARMUP_STEPS="${LINGBOT_GRPO_WARMUP_STEPS:-4}"
export LINGBOT_GRPO_CLAMP_MIN="${LINGBOT_GRPO_CLAMP_MIN:-0.5}"
export LINGBOT_GRPO_CLAMP_MAX="${LINGBOT_GRPO_CLAMP_MAX:-2.0}"

CONFIG_NAME="${LINGBOT_GRPO_CONFIG_NAME:-haimiandian_grpo_ablation_base}"

cat > "$OUT/logs/launch_info.txt" <<EOF
start_time=$(date -Iseconds)
output_name=$OUTPUT_NAME
config_name=$CONFIG_NAME
use_grpo=$USE_GRPO
reward_scale=$REWARD_SCALE
alpha=$ALPHA
buffer_size=$BUFFER_SIZE
num_steps=$NUM_STEPS
save_interval=$LINGBOT_GRPO_SAVE_INTERVAL
resume_from=$LINGBOT_GRPO_RESUME_FROM
vae=${LINGBOT_ROOT:-/path/to/lingbot-va}/models/lingbot-va-base/vae
EOF

echo "[ablation] start $(date -Iseconds)" > "$LOG"
# Prefer LINGBOT_GRPO_MASTER_PORT: generic MASTER_PORT is often injected by schedulers and may collide (EADDRINUSE).
# Default 29552 matches the successful 2026-05-03 haimiandian_grpo_smoke run (see outputs/.../nohup_grpo.log).
NGPU=4 MASTER_PORT="${LINGBOT_GRPO_MASTER_PORT:-29552}" CONFIG_NAME="$CONFIG_NAME" SAVE_ROOT="$OUT/grpo" PYTHON_BIN=python \
  bash script/run_haimiandian_posttrain.sh >> "$LOG" 2>&1
echo "[ablation] exit 0 $(date -Iseconds)" >> "$LOG"
