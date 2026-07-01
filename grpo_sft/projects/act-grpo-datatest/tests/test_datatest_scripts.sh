#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

dataset_keys=(
  jiaodai
  box_task_id_15
  mouse_1
  mouse_2
  mouse_3
  tianqing_mixed
)

required_files=(
  "$ROOT/scripts/dataset_common.sh"
  "$ROOT/scripts/preflight_dataset.sh"
  "$ROOT/scripts/train_act_bc_dataset.sh"
  "$ROOT/scripts/train_act_grpo_dataset.sh"
  "$ROOT/scripts/run_datatest_chain.sh"
  "$ROOT/scripts/compare_act_zero_sft_grpo_offline.py"
  "$ROOT/scripts/build_mixed_lerobot_dataset.py"
)

for key in "${dataset_keys[@]}"; do
  required_files+=("$ROOT/configs/datasets/${key}.env")
done

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "missing required file: $file" >&2
    exit 1
  fi
done

for script in \
  "$ROOT/scripts/dataset_common.sh" \
  "$ROOT/scripts/preflight_dataset.sh" \
  "$ROOT/scripts/train_act_bc_dataset.sh" \
  "$ROOT/scripts/train_act_grpo_dataset.sh" \
  "$ROOT/scripts/run_datatest_chain.sh"; do
  bash -n "$script"
done

for key in "${dataset_keys[@]}"; do
  (
    # shellcheck source=/dev/null
    source "$ROOT/configs/datasets/${key}.env"
    test "$DATASET_KEY" = "$key"
    test -n "$DATASET_REPO_ID"
    test -e "$DATASET_ROOT/meta/info.json"
    test -e "$DATASET_ROOT/meta/stats.json"
    test -e "$DATASET_ROOT/meta/tasks.parquet"
  )
done

"$ROOT/scripts/train_act_bc_dataset.sh" jiaodai --dry-run | grep -q -- "--dataset.repo_id=tianqing/jiaodai"
"$ROOT/scripts/train_act_bc_dataset.sh" jiaodai --dry-run | grep -q -- "--dataset.root=${ACT_GRPO_DATA_ROOT:-/path/to/act-grpo-datatest}/resources/datasets/jiaodai"
"$ROOT/scripts/train_act_grpo_dataset.sh" jiaodai --dry-run | grep -q -- "--use_grpo=true"
"$ROOT/scripts/train_act_grpo_dataset.sh" jiaodai --dry-run | grep -q -- "--grpo_beta=1.0"
"$ROOT/scripts/run_datatest_chain.sh" jiaodai --dry-run | grep -q -- "CHAIN_DRY_RUN_OK dataset=jiaodai"
"$ROOT/scripts/build_mixed_lerobot_dataset.py" --dry-run | grep -q -- "MIXED_DATASET_DRY_RUN_OK"
"$ROOT/scripts/train_act_bc_dataset.sh" tianqing_mixed --dry-run | grep -q -- "--dataset.repo_id=tianqing/tianqing_mixed"
"$ROOT/scripts/run_datatest_chain.sh" tianqing_mixed --dry-run | grep -q -- "CHAIN_DRY_RUN_OK dataset=tianqing_mixed"

relative_ckpt="results/baselines/jiaodai/fake/checkpoints/000001/pretrained_model"
ACT_GRPO_PRETRAINED_PATH="$relative_ckpt" \
  "$ROOT/scripts/train_act_grpo_dataset.sh" jiaodai --dry-run \
  | grep -q -- "--policy.pretrained_path=$ROOT/$relative_ckpt"

if grep -R "/path/to/outputs" \
  "$ROOT/scripts/dataset_common.sh" \
  "$ROOT/scripts/preflight_dataset.sh" \
  "$ROOT/scripts/train_act_bc_dataset.sh" \
  "$ROOT/scripts/train_act_grpo_dataset.sh"; then
  echo "datatest scripts must write under ${ACT_GRPO_DATA_ROOT:-/path/to/act-grpo-datatest}" >&2
  exit 1
fi

echo "DATATEST_SCRIPT_TEST_OK"
