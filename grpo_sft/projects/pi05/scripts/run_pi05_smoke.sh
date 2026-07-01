#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACT_GRPO_ROOT="${ACT_GRPO_ROOT:-$(cd "$PI05_PROJECT_ROOT/.." && pwd)}"

# shellcheck source=../configs/pi05.env
source "$PI05_PROJECT_ROOT/configs/pi05.env"

export ACT_GRPO_ROOT
export PI05_PROJECT_ROOT
export PI_POLICY=pi05
export PI05_PRETRAINED_PATH
export HF_HOME
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export TOKENIZERS_PARALLELISM
export PI_TOKENIZER_MAX_LENGTH

cd "$ACT_GRPO_ROOT"

exec bash scripts/train_pi0_bc_haimiandian.sh --policy pi05 --smoke "$@"
