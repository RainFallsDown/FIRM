#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACT_GRPO_ROOT:-/path/to/act-grpo}"
CONDA="${CONDA_ROOT:-/path/to/miniconda3}/bin/conda"
ENV_DIR="${CONDA_ROOT:-/path/to/miniconda3}/envs/act-grpo"

echo "=== env ==="
"$CONDA" env list | grep -E '(^act-grpo[[:space:]]|^lerobot[[:space:]])' || true
"$ENV_DIR/bin/python" -V
"$ENV_DIR/bin/python" - <<'PY'
import sys
print("sys.prefix", sys.prefix)
try:
    import torch
    print("torch", torch.__version__)
except Exception as exc:
    print("torch_import_error", repr(exc))
    raise
PY

echo "=== activation ==="
ls -l "$ROOT/env/activate_act_grpo.sh"
source "$ROOT/env/activate_act_grpo.sh"
echo "which_python=$(command -v python)"
python -V
python - <<'PY'
import sys
print("activated_prefix", sys.prefix)
PY

echo "=== resources ==="
du -sh "$ROOT/resources"/*

echo "=== backups ==="
find "$ROOT/backups" -maxdepth 1 -type d -name 'resources_*' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -5
latest="$(find "$ROOT/backups" -maxdepth 1 -type d -name 'resources_*' | sort | tail -1)"
echo "latest_backup=$latest"
du -sh "$latest"/*

echo "=== logs ==="
ls -lt "$ROOT/logs" | head -5
