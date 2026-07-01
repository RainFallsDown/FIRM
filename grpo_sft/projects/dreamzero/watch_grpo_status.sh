#!/usr/bin/env bash

OUT="${1:?usage: watch_grpo_status.sh OUT_DIR}"
INTERVAL="${INTERVAL:-600}"
LOG="$OUT/status_watch.log"
TRAIN_LOG="$OUT/training.log"
PIPELINE_PID_FILE="$OUT/pipeline.pid"

snapshot() {
  {
    echo "===== $(date -Iseconds) ====="

    echo "[pipeline]"
    PID="$(cat "$PIPELINE_PID_FILE" 2>/dev/null || true)"
    if [ -n "$PID" ]; then
      ps -p "$PID" -o pid=,etime=,cmd= || echo "pipeline process not found"
    else
      echo "pipeline pid file missing"
    fi

    echo "[training]"
    pgrep -af "torchru[n].*$OUT" || echo "torchrun not found"
    pgrep -af "experimen[t].py.*$OUT" | head -8 || true

    echo "[progress]"
    grep -ao "[0-9][0-9]*/3500" "$TRAIN_LOG" 2>/dev/null | tail -1 || true

    echo "[last_loss]"
    grep "loss" "$TRAIN_LOG" 2>/dev/null | tail -6 || true

    echo "[last_grpo]"
    grep "grpo_reward" "$TRAIN_LOG" 2>/dev/null | tail -6 || true

    echo "[checkpoints]"
    ls -d "$OUT"/checkpoint-* 2>/dev/null | tail -10 || true

    echo "[gpu]"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

    echo "[comparison]"
    if [ -f "$OUT/comparison_zero_sft_grpo_n100/summary.md" ]; then
      echo "summary=$OUT/comparison_zero_sft_grpo_n100/summary.md"
      sed -n '1,40p' "$OUT/comparison_zero_sft_grpo_n100/summary.md" 2>/dev/null || true
    else
      pgrep -af "evaluate_zero_sft_grpo.py.*$OUT" || echo "comparison not started"
    fi
    echo
  } >> "$LOG"
}

while true; do
  snapshot

  PID="$(cat "$PIPELINE_PID_FILE" 2>/dev/null || true)"
  PIPELINE_ALIVE=0
  if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
    PIPELINE_ALIVE=1
  fi
  TRAIN_ALIVE=0
  if pgrep -f "torchru[n].*$OUT" >/dev/null 2>&1; then
    TRAIN_ALIVE=1
  fi
  COMPARE_ALIVE=0
  if pgrep -f "evaluate_zero_sft_grpo.py.*$OUT" >/dev/null 2>&1; then
    COMPARE_ALIVE=1
  fi

  if [ "$PIPELINE_ALIVE" -eq 0 ] && [ "$TRAIN_ALIVE" -eq 0 ] && [ "$COMPARE_ALIVE" -eq 0 ]; then
    echo "===== $(date -Iseconds) watcher exiting: pipeline/train/compare all stopped =====" >> "$LOG"
    exit 0
  fi

  sleep "$INTERVAL"
done
