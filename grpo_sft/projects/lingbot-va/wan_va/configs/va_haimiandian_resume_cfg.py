# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from copy import deepcopy

from .va_haimiandian_train_cfg import va_haimiandian_train_cfg

RESUME_FROM_STEP_500 = "${LINGBOT_ROOT:-/path/to/lingbot-va}/outputs/haimiandian_full_20260501_194155/sft/checkpoints/checkpoint_step_500"
RESUME_FROM_STEP_1000 = "${LINGBOT_ROOT:-/path/to/lingbot-va}/outputs/haimiandian_sft_resume4500_20260502_143105/sft/checkpoints/checkpoint_step_1000"
# Latest haimiandian SFT run (1000 -> 3500 extra steps, ends at global step 3500).
RESUME_FROM_SFT_EQ5000_STEP_3500 = "${LINGBOT_ROOT:-/path/to/lingbot-va}/outputs/haimiandian_sft_resume1000_to_eq5000_20260503_030524/sft/checkpoints/checkpoint_step_3500"

va_haimiandian_train_resume_short_cfg = deepcopy(va_haimiandian_train_cfg)
va_haimiandian_train_resume_short_cfg.__name__ = "Config: VA haimiandian train resume short debug"
va_haimiandian_train_resume_short_cfg.resume_from = RESUME_FROM_STEP_500
va_haimiandian_train_resume_short_cfg.num_steps = 220
va_haimiandian_train_resume_short_cfg.save_interval = 100

va_haimiandian_train_resume4500_cfg = deepcopy(va_haimiandian_train_cfg)
va_haimiandian_train_resume4500_cfg.__name__ = "Config: VA haimiandian train resume from step 500 for 4500 steps"
va_haimiandian_train_resume4500_cfg.resume_from = RESUME_FROM_STEP_500
va_haimiandian_train_resume4500_cfg.num_steps = 4500
va_haimiandian_train_resume4500_cfg.save_interval = 500

va_haimiandian_train_resume1000_to5000_cfg = deepcopy(va_haimiandian_train_cfg)
va_haimiandian_train_resume1000_to5000_cfg.__name__ = "Config: VA haimiandian train resume from run step 1000 to equivalent 5000"
va_haimiandian_train_resume1000_to5000_cfg.resume_from = RESUME_FROM_STEP_1000
va_haimiandian_train_resume1000_to5000_cfg.num_steps = 3500
va_haimiandian_train_resume1000_to5000_cfg.save_interval = 500

va_haimiandian_grpo_smoke_cfg = deepcopy(va_haimiandian_train_cfg)
va_haimiandian_grpo_smoke_cfg.__name__ = "Config: VA haimiandian GRPO smoke test from step 1000"
va_haimiandian_grpo_smoke_cfg.resume_from = os.environ.get(
    "LINGBOT_GRPO_RESUME_FROM", RESUME_FROM_SFT_EQ5000_STEP_3500
)
va_haimiandian_grpo_smoke_cfg.num_steps = 20
va_haimiandian_grpo_smoke_cfg.save_interval = 1000
va_haimiandian_grpo_smoke_cfg.gradient_accumulation_steps = 2
va_haimiandian_grpo_smoke_cfg.load_worker = 4
va_haimiandian_grpo_smoke_cfg.use_grpo = True
va_haimiandian_grpo_smoke_cfg.grpo_reward_scale = 1.0
va_haimiandian_grpo_smoke_cfg.grpo_buffer_size = 32
va_haimiandian_grpo_smoke_cfg.grpo_buffer_warmup_steps = 4
va_haimiandian_grpo_smoke_cfg.grpo_advantage_alpha = 1.0
va_haimiandian_grpo_smoke_cfg.grpo_weight_clamp_min = 0.5
va_haimiandian_grpo_smoke_cfg.grpo_weight_clamp_max = 2.0

va_haimiandian_grpo_ablation_base_cfg = deepcopy(va_haimiandian_train_cfg)
va_haimiandian_grpo_ablation_base_cfg.__name__ = "Config: VA haimiandian GRPO ablation base"
va_haimiandian_grpo_ablation_base_cfg.resume_from = os.environ.get(
    "LINGBOT_GRPO_RESUME_FROM", RESUME_FROM_SFT_EQ5000_STEP_3500
)
va_haimiandian_grpo_ablation_base_cfg.num_steps = int(os.environ.get("LINGBOT_GRPO_NUM_STEPS", "100"))
va_haimiandian_grpo_ablation_base_cfg.save_interval = int(os.environ.get("LINGBOT_GRPO_SAVE_INTERVAL", "1000"))
va_haimiandian_grpo_ablation_base_cfg.gradient_accumulation_steps = int(os.environ.get("LINGBOT_GRPO_GRAD_ACCUM", "2"))
va_haimiandian_grpo_ablation_base_cfg.load_worker = int(os.environ.get("LINGBOT_GRPO_LOAD_WORKER", "4"))
va_haimiandian_grpo_ablation_base_cfg.use_grpo = os.environ.get("LINGBOT_GRPO_USE_GRPO", "true").lower() == "true"
va_haimiandian_grpo_ablation_base_cfg.grpo_reward_scale = float(os.environ.get("LINGBOT_GRPO_REWARD_SCALE", "1.0"))
va_haimiandian_grpo_ablation_base_cfg.grpo_buffer_size = int(os.environ.get("LINGBOT_GRPO_BUFFER_SIZE", "32"))
va_haimiandian_grpo_ablation_base_cfg.grpo_buffer_warmup_steps = int(os.environ.get("LINGBOT_GRPO_WARMUP_STEPS", "4"))
va_haimiandian_grpo_ablation_base_cfg.grpo_advantage_alpha = float(os.environ.get("LINGBOT_GRPO_ALPHA", "1.0"))
va_haimiandian_grpo_ablation_base_cfg.grpo_weight_clamp_min = float(os.environ.get("LINGBOT_GRPO_CLAMP_MIN", "0.5"))
va_haimiandian_grpo_ablation_base_cfg.grpo_weight_clamp_max = float(os.environ.get("LINGBOT_GRPO_CLAMP_MAX", "2.0"))
