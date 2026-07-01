# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_haimiandian_cfg = EasyDict(__name__="Config: VA haimiandian")
va_haimiandian_cfg.update(va_shared_cfg)

va_haimiandian_cfg.wan22_pretrained_model_name_or_path = "${LINGBOT_ROOT:-/path/to/lingbot-va}/models/lingbot-va-base"

va_haimiandian_cfg.attn_window = 72
va_haimiandian_cfg.frame_chunk_size = 2
va_haimiandian_cfg.env_type = "haimiandian"
va_haimiandian_cfg.latent_layout = "tshape"
va_haimiandian_cfg.action_pose_mode = "raw"

va_haimiandian_cfg.height = 256
va_haimiandian_cfg.width = 320
va_haimiandian_cfg.action_dim = 30
va_haimiandian_cfg.action_per_frame = 12
va_haimiandian_cfg.obs_cam_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
va_haimiandian_cfg.guidance_scale = 5
va_haimiandian_cfg.action_guidance_scale = 1

va_haimiandian_cfg.num_inference_steps = 25
va_haimiandian_cfg.video_exec_step = -1
va_haimiandian_cfg.action_num_inference_steps = 50

va_haimiandian_cfg.snr_shift = 5.0
va_haimiandian_cfg.action_snr_shift = 1.0

va_haimiandian_cfg.used_action_channel_ids = list(range(16))
inverse_used_action_channel_ids = [len(va_haimiandian_cfg.used_action_channel_ids)] * va_haimiandian_cfg.action_dim
for i, j in enumerate(va_haimiandian_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_haimiandian_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_haimiandian_cfg.action_norm_method = "quantiles"
va_haimiandian_cfg.norm_stat = {
    "q01": [
        0.2453520784205324,
        -1.0465188792872373,
        -1.3922108523797359,
        0.005150128545675745,
        -1.7881207987652246,
        -1.5530121345944838,
        1.5156665273850545,
        -1.6488608444857449,
        1.6920946065878433e-05,
        -0.5009206806296218,
        0.17276724181559464,
        -0.008092475771087343,
        -1.3527717650446942,
        -2.6149857646530674,
        -1.000000013351432e-10,
        -1.000000013351432e-10,
    ]
    + [0.0] * 14,
    "q99": [
        1.816772507335845,
        -0.009255857523523751,
        0.5047004324750973,
        1.6216860116831804,
        -0.1153327593036983,
        0.13997574927032613,
        2.8797321002958496,
        -0.24510334537413286,
        0.5643549489958063,
        0.6558848855045961,
        1.474234162090559,
        1.8503997629411006,
        -0.48193809664144754,
        -1.2584156229340544,
        0.9999885686417964,
        0.9999899307429788,
    ]
    + [1.0] * 14,
}
