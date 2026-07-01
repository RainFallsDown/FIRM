#!/usr/bin/env bash
# Replay the successful GRPO smoke path (config haimiandian_grpo_smoke, 20 steps in cfg).
# Default resume: latest haimiandian SFT checkpoint_step_3500 (override with LINGBOT_GRPO_RESUME_FROM).
set -euo pipefail

ROOT=${LINGBOT_ROOT:-/path/to/lingbot-va}
CODE=$ROOT/lingbot-va
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/outputs/haimiandian_grpo_smoke_verify_${STAMP}"
LOG="$OUT/logs/nohup_grpo.log"

DEFAULT_RESUME="$ROOT/outputs/haimiandian_sft_resume1000_to_eq5000_20260503_030524/sft/checkpoints/checkpoint_step_3500"
export LINGBOT_GRPO_RESUME_FROM="${LINGBOT_GRPO_RESUME_FROM:-$DEFAULT_RESUME}"

mkdir -p "$OUT/logs"
set +u
source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate lingbot-va-torch29
set -u
cd "$CODE"

export PYTHONPATH="$CODE:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cat > "$OUT/logs/launch_info.txt" <<EOF
start_time=$(date -Iseconds)
config=haimiandian_grpo_smoke
resume_from=$LINGBOT_GRPO_RESUME_FROM
save_root=$OUT/grpo
EOF

echo "[grpo] start $(date -Iseconds)" > "$LOG"
# Same master port family as passing run: outputs/haimiandian_grpo_smoke_20260503_021347 (29552)
NGPU=4 MASTER_PORT="${LINGBOT_GRPO_MASTER_PORT:-29552}" CONFIG_NAME=haimiandian_grpo_smoke SAVE_ROOT="$OUT/grpo" PYTHON_BIN=python \
  bash script/run_haimiandian_posttrain.sh >> "$LOG" 2>&1
echo "[grpo] exit 0 $(date -Iseconds)" >> "$LOG"
echo "OK: logs at $LOG"
