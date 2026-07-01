#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACT_GRPO_ROOT:-/path/to/act-grpo}"
CONDA="${CONDA_ROOT:-/path/to/miniconda3}/bin/conda"
ENV_DIR="${CONDA_ROOT:-/path/to/miniconda3}/envs/act-grpo"
SRC_ENV="${CONDA_ROOT:-/path/to/miniconda3}/envs/lerobot"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/resume_setup_${TS}.log"

mkdir -p "$ROOT/env" "$ROOT/backups" "$ROOT/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== resume act-grpo setup ==="
echo "timestamp=$TS"
echo "host=$(hostname)"
date -Iseconds

env_registered() {
  "$CONDA" env list | awk '{print $1}' | grep -qx "act-grpo"
}

echo "=== current env state ==="
"$CONDA" env list | grep -E '(^act-grpo[[:space:]]|^lerobot[[:space:]])' || true
ls -ld "$ENV_DIR" 2>/dev/null || true

if [ -d "$ENV_DIR" ] && ! env_registered; then
  MOVED="$ROOT/env/partial_act-grpo_${TS}"
  mv "$ENV_DIR" "$MOVED"
  echo "MOVED_PARTIAL_ENV $MOVED"
fi

if env_registered; then
  echo "CONDA_ENV_EXISTS act-grpo"
else
  echo "CONDA_OFFLINE_CLONE_START"
  if CONDA_OFFLINE=true CONDA_NOTICES=false "$CONDA" create -y --offline -n act-grpo --clone lerobot; then
    echo "CONDA_OFFLINE_CLONE_DONE"
  else
    echo "CONDA_OFFLINE_CLONE_FAILED_FALLBACK_COPY"
    if [ -d "$ENV_DIR" ]; then
      FAILED="$ROOT/env/failed_offline_clone_act-grpo_${TS}"
      mv "$ENV_DIR" "$FAILED"
      echo "MOVED_FAILED_OFFLINE_CLONE $FAILED"
    fi
    echo "COPY_ENV_START src=$SRC_ENV dst=$ENV_DIR"
    cp -a --reflink=auto "$SRC_ENV" "$ENV_DIR"
    echo "COPY_ENV_DONE"
  fi
fi

echo "=== conda env verification ==="
"$CONDA" env list | grep -E '(^act-grpo[[:space:]]|/act-grpo$)' || true
"$ENV_DIR/bin/python" -V
"$ENV_DIR/bin/python" - <<'PY'
import sys
print("sys.prefix", sys.prefix)
try:
    import torch
    print("torch", torch.__version__)
except Exception as exc:
    print("torch_import_error", repr(exc))
PY

cat > "$ROOT/env/activate_act_grpo.sh" <<'EOF'
#!/usr/bin/env bash
source ${CONDA_ROOT:-/path/to/miniconda3}/etc/profile.d/conda.sh
conda activate act-grpo
EOF
chmod +x "$ROOT/env/activate_act_grpo.sh"

echo "=== backup copied resources ==="
BACKUP_DIR="$ROOT/backups/resources_${TS}"
mkdir -p "$BACKUP_DIR"
for name in lerobot lingbot-va bc-grpo-code haimiandian_50 lingbot-va-base RESOURCE_MAP.txt; do
  if [ -e "$ROOT/resources/$name" ]; then
    echo "BACKUP_COPY_START $name"
    cp -a --reflink=auto "$ROOT/resources/$name" "$BACKUP_DIR/$name"
    echo "BACKUP_COPY_DONE $name"
  else
    echo "BACKUP_SKIP_MISSING $name"
  fi
done

echo "=== final verification ==="
echo "resource sizes:"
du -sh "$ROOT/resources"/* 2>/dev/null || true
echo "backup dir:"
echo "$BACKUP_DIR"
du -sh "$BACKUP_DIR"/* 2>/dev/null || true
echo "workspace dirs:"
find "$ROOT" -maxdepth 2 -type d | sort
echo "disk:"
df -h "${WORKSPACE_ROOT:-/path/to/workspace}" || true
date -Iseconds
echo "RESUME_SETUP_DONE root=$ROOT env=$ENV_DIR backup=$BACKUP_DIR log=$LOG_FILE"
