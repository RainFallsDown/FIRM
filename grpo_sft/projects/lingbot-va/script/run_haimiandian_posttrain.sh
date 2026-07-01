#!/usr/bin/bash

set -euo pipefail
set -x

umask 007

NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29521"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"haimiandian_train"}
SAVE_ROOT=${SAVE_ROOT:-"${LINGBOT_ROOT:-/path/to/lingbot-va}/outputs/haimiandian_sft_$(date +%Y%m%d_%H%M%S)"}
PYTHON_BIN=${PYTHON_BIN:-"python"}

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# Default: only rank-0 logs (clean). Set LINGBOT_SHOW_ALL_RANK_LOGS=1 to omit --local-ranks-filter (debug rank>0 crashes).
TORCHRUN_EXTRA=(--nproc_per_node="${NGPU}" --master_port "${MASTER_PORT}" --tee 3)
if [ -n "${LINGBOT_SHOW_ALL_RANK_LOGS:-}" ]; then
  :
else
  TORCHRUN_EXTRA+=(--local-ranks-filter "${LOG_RANK}")
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON_BIN}" -m torch.distributed.run \
    "${TORCHRUN_EXTRA[@]}" \
    -m wan_va.train \
    --config-name "${CONFIG_NAME}" \
    --save-root "${SAVE_ROOT}"
