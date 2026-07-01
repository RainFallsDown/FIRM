# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from easydict import EasyDict

from .va_haimiandian_cfg import va_haimiandian_cfg

va_haimiandian_train_cfg = EasyDict(__name__="Config: VA haimiandian train")
va_haimiandian_train_cfg.update(va_haimiandian_cfg)

va_haimiandian_train_cfg.dataset_path = "${LINGBOT_DATA_ROOT:-/path/to/lingbot-data}/haimiandian_50"
va_haimiandian_train_cfg.empty_emb_path = os.path.join(va_haimiandian_train_cfg.dataset_path, "empty_emb.pt")
va_haimiandian_train_cfg.enable_wandb = False
va_haimiandian_train_cfg.load_worker = 8
va_haimiandian_train_cfg.save_interval = 500
va_haimiandian_train_cfg.gc_interval = 25
va_haimiandian_train_cfg.cfg_prob = 0.1

va_haimiandian_train_cfg.learning_rate = 1e-5
va_haimiandian_train_cfg.beta1 = 0.9
va_haimiandian_train_cfg.beta2 = 0.95
va_haimiandian_train_cfg.weight_decay = 0.1
va_haimiandian_train_cfg.warmup_steps = 10
va_haimiandian_train_cfg.batch_size = 1
va_haimiandian_train_cfg.gradient_accumulation_steps = 8
va_haimiandian_train_cfg.num_steps = 5000
