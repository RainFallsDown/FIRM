#!/usr/bin/env bash
set -euo pipefail

# Deprecated helper from the original isolated remote workspace. The integrated
# FIRM repository keeps LeRobot at the repository root under src/lerobot.
ROOT="${ACT_GRPO_ROOT:-/path/to/act-grpo}"
CONDA="${CONDA_ROOT:-/path/to/miniconda3}/bin/conda"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/setup_${TS}.log"

mkdir -p "$ROOT" "$ROOT/resources" "$ROOT/backups" "$ROOT/logs" "$ROOT/scripts" "$ROOT/results" "$ROOT/plans" "$ROOT/notes" "$ROOT/env"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== act-grpo setup start ==="
echo "timestamp=$TS"
echo "host=$(hostname)"
echo "root=$ROOT"
date -Iseconds

cat > "$ROOT/README.md" <<EOF
# act-grpo isolated workspace

Created: $TS
Root: $ROOT

Purpose: isolated workspace for ACT-GRPO and PI0/PI05-GRPO experiments.

Rules:
- Modify copied resources under resources/ only.
- Do not edit original repositories or datasets directly.
- Keep ACT-GRPO and PI0/PI05-GRPO plans, scripts, logs, notes, and results under this workspace.
- Timestamped backups of copied resources are stored under backups/.
EOF

cat > "$ROOT/MANIFEST.txt" <<EOF
created_at=$TS
root=$ROOT
host=$(hostname)
source_lerobot=${FIRM_ROOT:-/path/to/FIRM}
source_lingbot_va=${LINGBOT_ROOT:-/path/to/lingbot-va}/lingbot-va
source_bc_grpo=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/bc-grpo
source_haimiandian=${LINGBOT_DATA_ROOT:-/path/to/lingbot-data}/haimiandian_50
source_lingbot_model=${LINGBOT_ROOT:-/path/to/lingbot-va}/models/lingbot-va-base
conda_env=${CONDA_ROOT:-/path/to/miniconda3}/envs/act-grpo
EOF

echo "=== disk before copy ==="
df -h "${WORKSPACE_ROOT:-/path/to/workspace}" || true

copy_resource() {
  local name="$1"
  local src="$2"
  local dst="$ROOT/resources/$name"

  if [ ! -e "$src" ]; then
    echo "MISSING_SOURCE name=$name src=$src"
    return 1
  fi

  if [ -e "$dst" ]; then
    echo "SKIP_EXISTING name=$name dst=$dst"
  else
    echo "COPY_START name=$name src=$src dst=$dst"
    cp -a --reflink=auto "$src" "$dst"
    echo "COPY_DONE name=$name"
  fi

  du -sh "$dst"
}

echo "=== copy mutable code resources ==="
copy_resource "lerobot" "${FIRM_ROOT:-/path/to/FIRM}"
copy_resource "lingbot-va" "${LINGBOT_ROOT:-/path/to/lingbot-va}/lingbot-va"
copy_resource "bc-grpo-code" "${DREAMZERO_ROOT:-/path/to/dreamzero}/code/bc-grpo"

echo "=== copy data and model resources ==="
copy_resource "haimiandian_50" "${LINGBOT_DATA_ROOT:-/path/to/lingbot-data}/haimiandian_50"
copy_resource "lingbot-va-base" "${LINGBOT_ROOT:-/path/to/lingbot-va}/models/lingbot-va-base"

echo "=== write resource map ==="
cat > "$ROOT/resources/RESOURCE_MAP.txt" <<EOF
lerobot=$ROOT/resources/lerobot
lingbot_va=$ROOT/resources/lingbot-va
bc_grpo_code=$ROOT/resources/bc-grpo-code
haimiandian_50=$ROOT/resources/haimiandian_50
lingbot_va_base=$ROOT/resources/lingbot-va-base
EOF
cat "$ROOT/resources/RESOURCE_MAP.txt"

echo "=== conda env setup ==="
if [ ! -x "$CONDA" ]; then
  echo "MISSING_CONDA $CONDA"
  exit 1
fi

if "$CONDA" env list | awk '{print $1}' | grep -qx "act-grpo"; then
  echo "CONDA_ENV_EXISTS act-grpo"
else
  echo "CONDA_ENV_CREATE_START clone=lerobot name=act-grpo"
  "$CONDA" create -y -n act-grpo --clone lerobot
  echo "CONDA_ENV_CREATE_DONE act-grpo"
fi

"$CONDA" env list | grep -E '(^act-grpo[[:space:]]|/act-grpo$)' || true

echo "=== backup copied resources ==="
BACKUP_DIR="$ROOT/backups/resources_${TS}"
if [ -e "$BACKUP_DIR" ]; then
  echo "BACKUP_EXISTS $BACKUP_DIR"
else
  mkdir -p "$BACKUP_DIR"
  for name in lerobot lingbot-va bc-grpo-code haimiandian_50 lingbot-va-base RESOURCE_MAP.txt; do
    if [ -e "$ROOT/resources/$name" ]; then
      echo "BACKUP_COPY_START $name"
      cp -a --reflink=auto "$ROOT/resources/$name" "$BACKUP_DIR/$name"
      echo "BACKUP_COPY_DONE $name"
    fi
  done
fi

echo "=== final verification ==="
echo "root dirs:"
find "$ROOT" -maxdepth 2 -type d | sort
echo "resource sizes:"
du -sh "$ROOT/resources"/* 2>/dev/null || true
echo "backup sizes:"
du -sh "$BACKUP_DIR"/* 2>/dev/null || true
echo "conda env:"
"$CONDA" run -n act-grpo python -V

echo "=== disk after setup ==="
df -h "${WORKSPACE_ROOT:-/path/to/workspace}" || true
date -Iseconds
echo "SETUP_DONE root=$ROOT backup=$BACKUP_DIR log=$LOG_FILE"
