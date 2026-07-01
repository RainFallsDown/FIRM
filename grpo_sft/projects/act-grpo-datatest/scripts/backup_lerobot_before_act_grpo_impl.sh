#!/usr/bin/env bash
set -euo pipefail

# Deprecated helper from the original isolated remote workspace. The integrated
# FIRM repository uses the root src/lerobot package instead of resources/lerobot.
ROOT="${ACT_GRPO_ROOT:-/path/to/act-grpo}"
SRC="$ROOT/resources/lerobot"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/backups/lerobot_before_act_grpo_impl_${TS}"
LOG="$ROOT/logs/backup_lerobot_before_act_grpo_impl_${TS}.log"

mkdir -p "$ROOT/backups" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== backup lerobot before act-grpo implementation ==="
echo "timestamp=$TS"
echo "source=$SRC"
echo "backup=$BACKUP"
date -Iseconds

if [ ! -d "$SRC" ]; then
  echo "missing source: $SRC" >&2
  exit 2
fi

cd "$SRC"
echo "=== git status ==="
git status --short || true
echo "=== git rev ==="
git rev-parse HEAD || true

cp -a --reflink=auto "$SRC" "$BACKUP"

echo "=== backup verification ==="
du -sh "$SRC" "$BACKUP"
test -f "$BACKUP/src/lerobot/policies/act/modeling_act.py"
test -f "$BACKUP/src/lerobot/scripts/lerobot_train.py"
test -f "$BACKUP/src/lerobot/configs/train.py"

echo "BACKUP_OK backup=$BACKUP log=$LOG"
