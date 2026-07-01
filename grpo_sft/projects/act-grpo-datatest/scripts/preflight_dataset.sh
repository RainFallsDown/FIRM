#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dataset_common.sh
source "$SCRIPT_DIR/dataset_common.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/preflight_dataset.sh DATASET_KEY [--dataloader]" >&2
  exit 2
fi

DATASET_KEY_ARG="$1"
shift
DATALOADER=0
if [ "${1:-}" = "--dataloader" ]; then
  DATALOADER=1
fi

prepare_dataset_runtime "$DATASET_KEY_ARG"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
info = json.loads((root / "meta" / "info.json").read_text())
features = info["features"]

missing = [key for key in ("action", "observation.state") if key not in features]
if missing:
    raise SystemExit(f"missing_features={missing}")

action_shape = features["action"].get("shape")
state_shape = features["observation.state"].get("shape")
if action_shape != [16]:
    raise SystemExit(f"unexpected_action_shape={action_shape}")
if state_shape != [16]:
    raise SystemExit(f"unexpected_state_shape={state_shape}")

image_keys = sorted(key for key in features if key.startswith("observation.images."))
if not image_keys:
    raise SystemExit("missing_observation_images")

print(f"DATASET_KEY={os.environ['DATASET_KEY']}")
print(f"DATASET_REPO_ID={os.environ['DATASET_REPO_ID']}")
print(f"DATASET_ROOT={root}")
print(f"total_episodes={info.get('total_episodes')}")
print(f"total_frames={info.get('total_frames')}")
print(f"fps={info.get('fps')}")
print("image_keys=" + ",".join(image_keys))
print("METADATA_PREFLIGHT_OK")
PY

if [ "$DATALOADER" = "1" ]; then
  python - <<'PY'
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ["DATASET_ROOT"])
dataset = LeRobotDataset(
    repo_id=os.environ["DATASET_REPO_ID"],
    root=root,
    episodes=[0],
    video_backend=os.environ["VIDEO_BACKEND"],
)
if len(dataset) == 0:
    raise SystemExit("episode_zero_has_no_frames")
item = dataset[0]
required = ["action", "observation.state"]
missing = [key for key in required if key not in item]
if missing:
    raise SystemExit(f"missing_sample_keys={missing}")
print(f"dataloader_len_episode0={len(dataset)}")
print("sample_keys=" + ",".join(sorted(item.keys())))
print("DATALOADER_PREFLIGHT_OK")
PY
fi
