#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=haimiandian_baseline_common.sh
source "$SCRIPT_DIR/haimiandian_baseline_common.sh"

prepare_baseline_runtime

echo "=== runtime ==="
echo "ACT_GRPO_ROOT=$ACT_GRPO_ROOT"
echo "LEROBOT_DIR=$LEROBOT_DIR"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "OUTPUT_BASE=$OUTPUT_BASE"
echo "LOG_BASE=$LOG_BASE"
echo "HF_HOME=$HF_HOME"
echo "VIDEO_BACKEND=$VIDEO_BACKEND"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["DATASET_ROOT"])
lerobot_dir = Path(os.environ["LEROBOT_DIR"])
repo_id = os.environ["DATASET_REPO_ID"]
video_backend = os.environ["VIDEO_BACKEND"]

print("python", os.sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))

import lerobot

print("lerobot_import", Path(lerobot.__file__).resolve())
expected = (lerobot_dir / "src" / "lerobot").resolve()
actual = Path(lerobot.__file__).resolve().parent
if actual != expected:
    raise SystemExit(f"LeRobot import path mismatch: {actual} != {expected}")

for policy in ("act", "pi0", "pi05"):
    policy_dir = lerobot_dir / "src" / "lerobot" / "policies" / policy
    print(f"policy_{policy}_exists", policy_dir.is_dir())
    if not policy_dir.is_dir():
        raise SystemExit(f"Missing policy directory: {policy_dir}")

info = json.loads((root / "meta" / "info.json").read_text())
features = info["features"]
required_features = [
    "observation.state",
    "action",
    "observation.images.head.color",
    "observation.images.hand.right.color",
    "observation.images.hand.left.color",
]
missing = [key for key in required_features if key not in features]
if missing:
    raise SystemExit(f"Missing required dataset features: {missing}")

print("total_episodes", info["total_episodes"])
print("total_frames", info["total_frames"])
print("fps", info["fps"])
print("action_shape", features["action"]["shape"])
print("state_shape", features["observation.state"]["shape"])

if info["total_episodes"] != 50:
    raise SystemExit(f"Expected 50 episodes, got {info['total_episodes']}")
if features["action"]["shape"] != [16]:
    raise SystemExit(f"Expected 16D action, got {features['action']['shape']}")
if features["observation.state"]["shape"] != [16]:
    raise SystemExit(f"Expected 16D state, got {features['observation.state']['shape']}")

try:
    import torchcodec  # noqa: F401
    print("torchcodec_available", True)
except Exception as exc:
    print("torchcodec_available", False, repr(exc))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id=repo_id,
    root=root,
    episodes=[0],
    video_backend=video_backend,
)
print("dataset_len_episode0", len(dataset))
item = dataset[0]
for key in required_features:
    value = item[key]
    shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
    print("sample", key, shape)
print("sample_task", item["task"])
print("PREFLIGHT_OK")
PY
