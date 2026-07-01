#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=haimiandian_baseline_common.sh
source "$SCRIPT_DIR/haimiandian_baseline_common.sh"

activate_act_grpo
setup_lerobot_copy
ensure_path_exists "$DATASET_ROOT/meta/info.json" "haimiandian info.json"
ensure_path_exists "$DATASET_ROOT/data" "haimiandian data directory"

python - <<'PY'
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root = Path(os.environ["DATASET_ROOT"])
info_path = root / "meta" / "info.json"
data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
if not data_files:
    raise SystemExit(f"No parquet data files found under {root / 'data'}")

schema = pq.read_schema(data_files[0])
if "embodiment_id" not in schema.names:
    print("embodiment_id_absent_in_parquet")
    print("REPAIR_NOT_NEEDED")
    raise SystemExit(0)

field = schema.field("embodiment_id")
if pa.types.is_integer(field.type):
    dtype = str(field.type)
elif pa.types.is_string(field.type):
    dtype = "string"
else:
    raise SystemExit(f"Unsupported embodiment_id type: {field.type}")

info = json.loads(info_path.read_text())
features = info.setdefault("features", {})
current = features.get("embodiment_id")
desired = {"dtype": dtype, "shape": [1], "names": None}
desired_paths = {
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
}

changes = []
if current != desired:
    features["embodiment_id"] = desired
    changes.append(f"embodiment_id dtype={dtype} shape=[1]")

for key, value in desired_paths.items():
    if info.get(key) != value:
        info[key] = value
        changes.append(f"{key}={value}")

if not changes:
    print("REPAIR_NOT_NEEDED")
    raise SystemExit(0)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = info_path.with_name(f"info.json.before_baseline_repair_{ts}")
shutil.copy2(info_path, backup_path)

info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

print(f"backup={backup_path}")
for change in changes:
    print(f"changed={change}")
print("REPAIR_OK")
PY
