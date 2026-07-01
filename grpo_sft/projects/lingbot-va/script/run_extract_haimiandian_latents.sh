#!/usr/bin/bash

set -euo pipefail
set -x

PYTHON_BIN=${PYTHON_BIN:-"python"}
DATASET_PATH=${DATASET_PATH:-"${LINGBOT_DATA_ROOT:-/path/to/lingbot-data}/haimiandian_50"}
MODEL_PATH=${MODEL_PATH:-"${LINGBOT_ROOT:-/path/to/lingbot-va}/models/lingbot-va-base"}
DEVICE=${DEVICE:-"cuda:0"}
TARGET_FPS=${TARGET_FPS:-"10"}

"${PYTHON_BIN}" script/extract_wan22_latents.py \
    --dataset-path "${DATASET_PATH}" \
    --model-path "${MODEL_PATH}" \
    --device "${DEVICE}" \
    --target-fps "${TARGET_FPS}" \
    "$@"
