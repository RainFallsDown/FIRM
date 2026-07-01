#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

required_files=(
  "$ROOT/configs/baselines/haimiandian_baseline.env"
  "$ROOT/scripts/haimiandian_baseline_common.sh"
  "$ROOT/scripts/repair_haimiandian_metadata.sh"
  "$ROOT/scripts/preflight_haimiandian_baseline.sh"
  "$ROOT/scripts/train_act_bc_haimiandian.sh"
  "$ROOT/scripts/train_pi0_bc_haimiandian.sh"
)

for file in "${required_files[@]}"; do
  test -f "$file"
done

for script in "$ROOT"/scripts/*haimiandian*.sh; do
  bash -n "$script"
done

"$ROOT/scripts/train_act_bc_haimiandian.sh" --dry-run | grep -q "policy.type=act"
"$ROOT/scripts/train_pi0_bc_haimiandian.sh" --dry-run | grep -Eq "policy.type=(pi0|pi05)"

if grep "/path/to/outputs" "${required_files[@]}"; then
  echo "baseline files must write under ${ACT_GRPO_ROOT:-/path/to/act-grpo}, not /path/to/outputs" >&2
  exit 1
fi

if grep "${FIRM_ROOT:-/path/to/FIRM}" "${required_files[@]}"; then
  echo "baseline files must use the configured FIRM root LeRobot source" >&2
  exit 1
fi

echo "BASELINE_SCRIPT_TEST_OK"
