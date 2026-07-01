#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=haimiandian_baseline_common.sh
source "$SCRIPT_DIR/haimiandian_baseline_common.sh"

PI_MODEL_REPO="${PI_MODEL_REPO:-lerobot/pi0_base}"
PI_MODEL_DIR="${PI_MODEL_DIR:-$ACT_GRPO_ROOT/resources/models/pi0_base}"
FORCE_DOWNLOAD=0

usage() {
  cat <<'EOF'
Usage: scripts/download_pi0_resources.sh [--force]

Defaults:
  PI_MODEL_REPO=lerobot/pi0_base
  PI_MODEL_DIR=${ACT_GRPO_ROOT:-/path/to/act-grpo}/resources/models/pi0_base

Optional environment:
  HF_TOKEN=...                       # if the repo requires authentication
  HF_ENDPOINT=https://hf-mirror.com  # force one endpoint
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      FORCE_DOWNLOAD=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

activate_act_grpo
setup_lerobot_copy
ensure_dir "$(dirname "$PI_MODEL_DIR")"
ensure_dir "$PI_MODEL_DIR"

export PI_MODEL_REPO
export PI_MODEL_DIR
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

validate_model_dir() {
  python - <<'PY'
import json
import os
from pathlib import Path

target = Path(os.environ["PI_MODEL_DIR"])
required = ["config.json", "model.safetensors"]
missing = [name for name in required if not (target / name).exists()]
if missing:
    raise SystemExit(f"missing_required_files={missing}")

cfg = json.loads((target / "config.json").read_text())
print("model_dir", target)
print("policy_type", cfg.get("type"))
print("model_size_bytes", (target / "model.safetensors").stat().st_size)
print("PI0_RESOURCE_OK")
PY
}

if [ "$FORCE_DOWNLOAD" = "0" ] && [ -f "$PI_MODEL_DIR/model.safetensors" ] && [ -f "$PI_MODEL_DIR/config.json" ]; then
  echo "PI0 model already exists. Use --force to refresh."
  validate_model_dir
  exit 0
fi

endpoints=()
if [ -n "${HF_ENDPOINT:-}" ]; then
  endpoints+=("$HF_ENDPOINT")
else
  endpoints+=("https://hf-mirror.com")
  endpoints+=("https://huggingface.co")
fi

download_once() {
  local endpoint="$1"
  export HF_ENDPOINT="$endpoint"
  echo "DOWNLOAD_TRY endpoint=$HF_ENDPOINT repo=$PI_MODEL_REPO target=$PI_MODEL_DIR"
  python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["PI_MODEL_REPO"]
target = Path(os.environ["PI_MODEL_DIR"])
target.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target,
    force_download=os.environ.get("FORCE_DOWNLOAD", "0") == "1",
)

print("SNAPSHOT_DOWNLOAD_DONE", target)
PY
}

last_status=1
for endpoint in "${endpoints[@]}"; do
  if FORCE_DOWNLOAD="$FORCE_DOWNLOAD" download_once "$endpoint"; then
    validate_model_dir
    echo "DOWNLOAD_DONE endpoint=$endpoint target=$PI_MODEL_DIR"
    exit 0
  else
    last_status=$?
    echo "DOWNLOAD_FAILED endpoint=$endpoint status=$last_status" >&2
  fi
done

cat >&2 <<EOF
PI0 download failed for all endpoints.

Tried:
$(printf '  %s\n' "${endpoints[@]}")

You can retry with:
  cd $ACT_GRPO_ROOT
  HF_ENDPOINT=https://hf-mirror.com bash scripts/download_pi0_resources.sh --force
EOF
exit "$last_status"
