#!/usr/bin/env bash
set -euo pipefail

cd ${LINGBOT_ROOT:-/path/to/lingbot-va}/lingbot-va

OUT_ROOT=${LINGBOT_ROOT:-/path/to/lingbot-va}/outputs/haimiandian_lingbot_grpo_ablation_20260503

run_one() {
  local use_grpo=$1
  local reward_scale=$2
  local alpha=$3
  local buffer_size=$4
  local steps=$5
  local name=$6
  local log="$OUT_ROOT/$name/logs/nohup_grpo.log"

  if [[ -f "$log" ]] && grep -q "\[ablation\] exit 0" "$log"; then
    echo "[run_all] skip completed $name $(date -Iseconds)"
    return 0
  fi

  echo "[run_all] start $name $(date -Iseconds)"
  # Avoid scheduler-injected MASTER_PORT; align with run_lingbot_grpo_ablation.sh default.
  LINGBOT_GRPO_MASTER_PORT="${LINGBOT_GRPO_MASTER_PORT:-29552}" bash script/run_lingbot_grpo_ablation.sh \
    "$use_grpo" "$reward_scale" "$alpha" "$buffer_size" "$steps" "$name"
  echo "[run_all] done $name $(date -Iseconds)"
}

run_one false 0.0 1.0 32 100 lb_hmd_bc_only_s100
run_one true 1.0 1.0 32 100 lb_hmd_grpo_w1_a1_b32_s100
run_one true 0.5 1.0 32 100 lb_hmd_grpo_w0p5_a1_b32_s100
run_one true 2.0 1.0 32 100 lb_hmd_grpo_w2_a1_b32_s100
run_one true 1.0 0.5 32 100 lb_hmd_grpo_w1_a0p5_b32_s100
run_one true 1.0 1.0 16 100 lb_hmd_grpo_w1_a1_b16_s100
