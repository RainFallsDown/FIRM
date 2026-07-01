from pathlib import Path


def patch_registry() -> None:
    p = Path("wan_va/configs/__init__.py")
    s = p.read_text()
    if "va_haimiandian_train_cfg" not in s:
        s = s.replace(
            "from .va_libero_i2va import va_libero_i2va_cfg\n",
            "from .va_libero_i2va import va_libero_i2va_cfg\n"
            "from .va_haimiandian_cfg import va_haimiandian_cfg\n"
            "from .va_haimiandian_train_cfg import va_haimiandian_train_cfg\n",
        )
        s = s.replace(
            "    'libero_i2av': va_libero_i2va_cfg,\n}",
            "    'libero_i2av': va_libero_i2va_cfg,\n"
            "    'haimiandian': va_haimiandian_cfg,\n"
            "    'haimiandian_train': va_haimiandian_train_cfg,\n"
            "}",
        )
    p.write_text(s)


def patch_dataset() -> None:
    p = Path("wan_va/dataset/lerobot_latent_dataset.py")
    s = p.read_text()
    old = """        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)
"""
    new = """        latent_layout = getattr(
            self.config,
            'latent_layout',
            'tshape' if self.config.env_type == 'robotwin_tshape' else 'concat_width',
        )
        if latent_layout == 'tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)
"""
    if old in s:
        s = s.replace(old, new)

    old = """        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
"""
    new = """        action_pose_mode = getattr(
            self.config,
            'action_pose_mode',
            'relative' if self.config.env_type == 'robotwin_tshape' else 'raw',
        )
        if action_pose_mode == 'relative': ## TODO support get_relative_pose for other dataset, currently only support robotwin
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
"""
    if old in s:
        s = s.replace(old, new)
    p.write_text(s)


def patch_server() -> None:
    p = Path("wan_va/wan_va_server.py")
    s = p.read_text()
    old = """        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
"""
    new = """        self.env_type = job_config.env_type
        self.latent_layout = getattr(
            job_config,
            'latent_layout',
            'tshape' if self.env_type == 'robotwin_tshape' else 'concat_width',
        )
        self.streaming_vae_half = None
        if self.latent_layout == 'tshape':
"""
    if old in s:
        s = s.replace(old, new)
    s = s.replace(
        "            if self.env_type == 'robotwin_tshape':\n",
        "            if self.latent_layout == 'tshape':\n",
    )
    s = s.replace(
        "        if self.env_type == 'robotwin_tshape':\n            videos_high",
        "        if self.latent_layout == 'tshape':\n            videos_high",
    )
    p.write_text(s)


def patch_flash_attention_import() -> None:
    p = Path("wan_va/modules/model.py")
    s = p.read_text()
    old = """try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func
"""
    new = """try:
    from flash_attn_interface import flash_attn_func
except Exception:
    try:
        from flash_attn import flash_attn_func
    except Exception:
        flash_attn_func = None
"""
    if old in s:
        s = s.replace(old, new)
    old = """        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
"""
    new = """        elif attn_mode == 'flashattn':
            if flash_attn_func is None:
                raise ImportError("flash_attn is required when attn_mode='flashattn'")
            self.attn_op = flash_attn_func
"""
    if old in s:
        s = s.replace(old, new)
    p.write_text(s)


patch_registry()
patch_dataset()
patch_server()
patch_flash_attention_import()
