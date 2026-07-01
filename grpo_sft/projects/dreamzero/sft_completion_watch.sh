#!/usr/bin/env bash
set -u

OUT="${DREAMZERO_ROOT:-/path/to/dreamzero}/outputs/wam_firm_sft_20260428"
LOG="$OUT/monitor/sft_completion_watch.log"

mkdir -p "$OUT/monitor"
echo "watch_start $(date '+%F %T')" > "$LOG"

while pgrep -f "torchrun.*wam_firm_sft" >/dev/null || pgrep -f "experiment.py.*wam_firm_sft" >/dev/null; do
  STEP="$(tail -n 200 "$OUT/loss_log.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | tail -1 | grep -o '[0-9]*' || true)"
  CKPT="$(find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1)"
  echo "$(date '+%F %T') step=${STEP:-unknown} latest_ckpt=${CKPT:-none}" >> "$LOG"
  sleep 600
done

echo "watch_end $(date '+%F %T')" >> "$LOG"
find "$OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -1 > "$OUT/monitor/final_checkpoint.txt"
