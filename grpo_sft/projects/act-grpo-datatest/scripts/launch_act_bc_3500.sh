#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACT_GRPO_ROOT:-/path/to/act-grpo}"
TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$ROOT/logs/launch/act_bc_3500_${TS}.log"

mkdir -p "$ROOT/logs/launch"
cd "$ROOT"

nohup bash -lc '
cd ${ACT_GRPO_ROOT:-/path/to/act-grpo}
CUDA_VISIBLE_DEVICES=0 \
ACT_STEPS=3500 \
ACT_BATCH_SIZE=16 \
ACT_NUM_WORKERS=8 \
ACT_SAVE_FREQ=500 \
ACT_LOG_FREQ=50 \
bash scripts/train_act_bc_haimiandian.sh
' > "$LAUNCH_LOG" 2>&1 &

PID="$!"
echo "pid=$PID"
echo "launch_log=$LAUNCH_LOG"
sleep 2
if ps -p "$PID" >/dev/null 2>&1; then
  echo "LAUNCH_OK"
else
  echo "LAUNCH_FAILED"
  tail -80 "$LAUNCH_LOG" || true
  exit 1
fi
