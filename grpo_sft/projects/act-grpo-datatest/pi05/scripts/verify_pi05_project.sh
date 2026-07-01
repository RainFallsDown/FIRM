#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACT_GRPO_ROOT="${ACT_GRPO_ROOT:-$(cd "$PI05_PROJECT_ROOT/.." && pwd)}"
MODEL_DIR="$PI05_PROJECT_ROOT/models/pi05_base"
TOKENIZER_DIR="$PI05_PROJECT_ROOT/tokenizer/paligemma-3b-pt-224"
EXPECTED_TOKENIZER_PATH="$TOKENIZER_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_path() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    echo "Missing $label: $path" >&2
    exit 2
  fi
}

require_path "$MODEL_DIR/config.json" "PI05 config"
require_path "$MODEL_DIR/model.safetensors" "PI05 model"
require_path "$MODEL_DIR/policy_preprocessor.json" "PI05 preprocessor"
require_path "$MODEL_DIR/policy_postprocessor.json" "PI05 postprocessor"
require_path "$TOKENIZER_DIR/tokenizer.json" "Paligemma tokenizer.json"
require_path "$TOKENIZER_DIR/tokenizer.model" "Paligemma tokenizer.model"
require_path "$PI05_PROJECT_ROOT/MANIFEST.sha256" "PI05 manifest"

echo "PI05_PROJECT_ROOT=$PI05_PROJECT_ROOT"
echo "ACT_GRPO_ROOT=$ACT_GRPO_ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "TOKENIZER_DIR=$TOKENIZER_DIR"

cd "$PI05_PROJECT_ROOT"
sha256sum -c MANIFEST.sha256

"$PYTHON_BIN" - "$MODEL_DIR" "$EXPECTED_TOKENIZER_PATH" <<'PY'
import json
import pathlib
import struct
import sys

model_dir = pathlib.Path(sys.argv[1])
expected_tokenizer = sys.argv[2]

for name in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
    json.loads((model_dir / name).read_text(encoding="utf-8"))

with (model_dir / "model.safetensors").open("rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_len))

tensors = [name for name in header if name != "__metadata__"]
if len(tensors) != 812:
    raise SystemExit(f"unexpected_tensor_count={len(tensors)}")

preprocessor = json.loads((model_dir / "policy_preprocessor.json").read_text(encoding="utf-8"))
tokenizer_paths = [
    step.get("config", {}).get("tokenizer_name")
    for step in preprocessor.get("steps", [])
    if step.get("registry_name") == "tokenizer_processor"
]
if tokenizer_paths != [expected_tokenizer]:
    raise SystemExit(f"unexpected_tokenizer_path={tokenizer_paths}, expected={expected_tokenizer}")

config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
if config.get("type") != "pi05":
    raise SystemExit(f"unexpected_policy_type={config.get('type')}")

print("SAFETENSORS_HEADER_OK")
print("tensor_count", len(tensors))
print("model_size_bytes", (model_dir / "model.safetensors").stat().st_size)
print("policy_type", config.get("type"))
PY

echo "PI05_PROJECT_OK"
