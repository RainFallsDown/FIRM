# FIRM SFT/GRPO

[English](README.md) | [中文](README_CN.md)

FIRM SFT/GRPO provides the training code used for supervised fine-tuning (SFT) and offline GRPO-style optimization across four robot policy lines: DreamZero, LingBot-VA, ACT, and Pi0.5. The project focuses on vision-action modeling for embodied manipulation and covers the main workflow from LeRobot data loading, video latent preparation, SFT training, GRPO reward weighting, and offline evaluation.

The repository keeps each upstream training framework in its own subdirectory while providing a unified top-level guide for environment setup, model preparation, dataset layout, and launch commands.

This copy is integrated under the main FIRM repository at `grpo_sft/`. The duplicate `projects/act-grpo-datatest/resources/lerobot` snapshot from the original archive was removed; use the root `src/lerobot/` package instead.

## Repository Layout

```text
firm/
  README.md
  README_CN.md
  ENVIRONMENT_REQUIREMENTS.txt
  requirements.txt
  requirements/
    dreamzero.txt
    lingbot-va.txt
    act-pi05.txt
  projects/
    dreamzero/
      train_wam_sft.sh
      run_grpo3500_compare.sh
      groot/vla/grpo_simple.py
      groot/vla/experiment/
      groot/vla/configs/
      SFT_test/
    lingbot-va/
      script/
      wan_va/train.py
      wan_va/grpo.py
      wan_va/configs/
      wan_va/dataset/
      evaluation/
    act-grpo-datatest/
      scripts/
      configs/
      pi05/
    pi05/
      scripts/
      configs/
      models/pi05_base/
  scripts/
    check_release.ps1
```

`projects/dreamzero`, `projects/lingbot-va`, and `projects/act-grpo-datatest` preserve the original relative project layout where practical. `projects/pi05` stores Pi0.5-specific configs and script copies. ACT/Pi0.5 training should use the root FIRM `src/lerobot/` package instead of a nested LeRobot copy.

## Components

### DreamZero

DreamZero is located at `projects/dreamzero`. It contains the Wan2.2-TI2V-5B based WAM/FIRM training path.

Main entry points:

- SFT: `train_wam_sft.sh`
- GRPO: `run_grpo3500_compare.sh`

The SFT stage performs behavior-cloning style supervised fine-tuning on the FIRM/Haimiandian dataset. The current script uses LoRA training, DeepSpeed ZeRO-3, four-GPU distributed execution, and 5000 training steps. The GRPO stage resumes from an SFT checkpoint, runs 3500 additional steps by default, and then calls `SFT_test/evaluate_zero_sft_grpo.py` to compare zero-shot, SFT, and GRPO policies.

DreamZero GRPO is implemented in:

```text
projects/dreamzero/groot/vla/grpo_simple.py
projects/dreamzero/groot/vla/experiment/base.py
```

The current implementation aligns DreamZero with the LingBot-VA action-only intrinsic reward. The reward combines first-order action smoothness, second-order jerk, action magnitude regularization, and gripper consistency. `GRPOWeightBuffer` maintains a sliding reward history, converts the current reward into an advantage, and maps it to a clipped exponential loss weight.

### LingBot-VA

LingBot-VA is located at `projects/lingbot-va`. It contains video-action latent extraction, SFT/post-training, GRPO ablation training, and evaluation utilities.

Main files:

```text
script/run_extract_haimiandian_latents.sh
script/run_haimiandian_posttrain.sh
script/run_lingbot_grpo_ablation.sh
wan_va/train.py
wan_va/grpo.py
wan_va/dataset/lerobot_latent_dataset.py
```

LingBot-VA GRPO does not run online rollouts. Instead, it computes an intrinsic reward from each training batch and reweights the original training loss. Reward computation and the sliding weight buffer are implemented in `wan_va/grpo.py`; the training loop calls them from `wan_va/train.py`.

### ACT

ACT is located at `projects/act-grpo-datatest`. It extends the LeRobot training entry point with ACT SFT and ACT GRPO scripts:

```text
scripts/train_act_bc_dataset.sh
scripts/train_act_grpo_dataset.sh
../src/lerobot/utils/act_grpo.py
../src/lerobot/configs/train.py
../src/lerobot/scripts/lerobot_train.py
```

ACT-GRPO is implemented as offline batch intrinsic reward-weighted BC. It does not require online interaction. For each batch, it combines BC quality, action smoothness, acceleration, and gripper consistency into an intrinsic reward, converts rewards into batch z-score advantages, and uses them as sample weights. The default reward weights are:

```text
bc_reward_weight=0.45
smooth_reward_weight=0.25
accel_reward_weight=0.20
gripper_reward_weight=0.10
```

### Pi0.5

Pi0.5 reuses the ACT/LeRobot `--use_grpo` path.

Core code:

```text
../src/lerobot/policies/pi05/
projects/act-grpo-datatest/scripts/train_pi05_grpo_dataset.sh
```

The `projects/pi05` directory stores Pi0.5 configs, base policy config files, and script copies for reference. For training, use `projects/act-grpo-datatest` so that the same dataset configuration, shared shell utilities, and LeRobot source are used.

## Environment Setup

See [ENVIRONMENT_REQUIREMENTS.txt](ENVIRONMENT_REQUIREMENTS.txt) for the full hardware and software requirements.

The three training stacks use incompatible PyTorch and Transformers versions, so use separate Python environments:

```text
DreamZero:   Python 3.11, torch==2.8.0, transformers==4.51.3
LingBot-VA:  Python 3.10, torch==2.9.0, transformers>=4.55.4
ACT/Pi0.5:   Python 3.10, torch>=2.2.1,<2.8.0, local LeRobot 0.4.3 tree
```

Install with the provided requirement files:

```bash
# DreamZero
conda create -n firm-dreamzero python=3.11 -y
conda activate firm-dreamzero
pip install -r requirements/dreamzero.txt

# LingBot-VA
conda create -n firm-lingbot python=3.10 -y
conda activate firm-lingbot
pip install -r requirements/lingbot-va.txt

# ACT / Pi0.5
conda create -n firm-act python=3.10 -y
conda activate firm-act
pip install -r requirements/act-pi05.txt
```

The top-level `requirements.txt` points to the ACT/Pi0.5 environment as the default lightweight install path:

```bash
pip install -r requirements.txt
```

For ACT/Pi0.5 scripts, create the activation helper expected by `dataset_common.sh`:

```bash
cd projects/act-grpo-datatest
mkdir -p env
cat > env/activate_act_grpo.sh <<'EOF'
source /path/to/miniconda3/bin/activate firm-act
EOF
```

## Model Preparation

Training scripts read model paths from command-line arguments, shell variables, or script-level defaults. A typical server layout is:

```text
/models/
  dreamzero/
  lingbot-va-base/
  act/
  pi05/
```

### DreamZero Models

DreamZero requires the Wan2.2/Wan2.1 components and tokenizer:

```text
Wan2.2-TI2V-5B
Wan2.1-I2V-14B-480P text encoder
Wan2.1-I2V-14B-480P image encoder
Wan2.2 VAE
umt5-xxl tokenizer
DreamZero or FIRM SFT checkpoint
```

Example download commands:

```bash
mkdir -p /models/dreamzero
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /models/dreamzero/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir /models/dreamzero/Wan2.1-I2V-14B-480P
huggingface-cli download google/umt5-xxl \
  --local-dir /models/dreamzero/umt5-xxl
```

Before running DreamZero scripts, check these path arguments:

```text
dit_version
text_encoder_pretrained_path
image_encoder_pretrained_path
vae_pretrained_path
tokenizer_path
pretrained_model_path
```

### LingBot-VA Models

Expected LingBot-VA base model layout:

```text
/models/lingbot-va-base/
  transformer/
  vae/
  text_encoder/
  tokenizer/
```

Latent extraction uses `MODEL_PATH`. GRPO resume uses `LINGBOT_GRPO_RESUME_FROM`:

```bash
export MODEL_PATH=/models/lingbot-va-base
export LINGBOT_GRPO_RESUME_FROM=/outputs/lingbot_sft/checkpoints/checkpoint_step_3500
```

### ACT and Pi0.5 Models

ACT GRPO resumes from an SFT `pretrained_model` directory:

```text
/outputs/act_sft/checkpoints/003500/pretrained_model/
  config.json
  model.safetensors
  policy_preprocessor...
  policy_postprocessor...
```

Pi0.5 GRPO uses the same `pretrained_model` layout:

```bash
export ACT_GRPO_PRETRAINED_PATH=/outputs/act_sft/checkpoints/003500/pretrained_model
export PI05_GRPO_PRETRAINED_PATH=/outputs/pi05_sft/checkpoints/003500/pretrained_model
```

## Dataset Format

### LeRobot v3

ACT and Pi0.5 use LeRobot v3 datasets. Minimal layout:

```text
dataset_root/
  meta/
    info.json
    stats.json
    tasks.parquet
  data/
    chunk-000/
      episode_000000.parquet
      ...
  videos/
    observation.images.<camera_name>/
      chunk-000/
        file-000.mp4
        ...
```

Dataset entries are configured in `projects/act-grpo-datatest/configs/datasets/*.env`:

```bash
DATASET_KEY=tianqing_mixed
DATASET_REPO_ID=tianqing/tianqing_mixed
DATASET_ROOT=/data/tianqing_mixed
OUTPUT_GROUP=tianqing_mixed
```

To add a new dataset, copy an existing `.env` file and update `DATASET_KEY`, `DATASET_REPO_ID`, `DATASET_ROOT`, and `OUTPUT_GROUP`.

### DreamZero Dataset

DreamZero uses LeRobot data plus GEAR/DreamZero metadata:

```text
dataset_root/
  data/
    chunk-000/
      episode_000000.parquet
  videos/
    chunk-000/
      observation.images.<camera_name>/
        episode_000000.mp4
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json
    embodiment.json
    stats.json
    relative_stats_dreamzero.json
```

Metadata conversion example:

```bash
cd projects/dreamzero
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /data/haimiandian_50 \
  --embodiment-tag haimiandian \
  --state-keys '{"left_arm":[0,7],"right_arm":[7,14],"gripper":[14,16]}' \
  --action-keys '{"left_arm":[0,7],"right_arm":[7,14],"gripper":[14,16]}' \
  --task-key annotation.task \
  --force
```

The Haimiandian data config is:

```text
projects/dreamzero/groot/vla/configs/data/dreamzero/haimiandian.yaml
```

It uses `video.head`, `state.left_arm`, `state.right_arm`, `state.gripper`, and the matching action keys.

### LingBot-VA Latent Dataset

LingBot-VA reads LeRobot data with precomputed video latents:

```text
haimiandian_50/
  meta/
    info.json
    episodes.jsonl
  data/
    chunk-000/
      episode_000000.parquet
  latents/
    chunk-000/
      <camera_key>/
        episode_000000_<start>_<end>.pth
  empty_emb.pt
```

`episodes.jsonl` may contain an `action_config` field with `start_frame`, `end_frame`, and `action_text`. If it is absent, the dataset loader falls back to episode-level task text.

## Training Commands

### DreamZero SFT

```bash
cd projects/dreamzero
conda activate firm-dreamzero
bash train_wam_sft.sh
```

Key settings:

```text
training_args.max_steps=5000
training_args.per_device_train_batch_size=1
training_args.gradient_accumulation_steps=8
training_args.deepspeed=groot/vla/configs/deepspeed/zero3.json
use_grpo=false
```

### DreamZero GRPO

```bash
cd projects/dreamzero
conda activate firm-dreamzero
RUN_ID=paper_grpo EVAL_SAMPLES=100 bash run_grpo3500_compare.sh
```

Set `SFT_CKPT` in the script or shell so it points to a readable SFT checkpoint. Default GRPO settings:

```text
use_grpo=true
grpo_weight=0.1
grpo_reward_scale=1.0
grpo_buffer_size=32
grpo_buffer_warmup_steps=4
grpo_advantage_alpha=1.0
grpo_weight_clamp_min=0.5
grpo_weight_clamp_max=2.0
```

### LingBot-VA Latent Extraction

```bash
cd projects/lingbot-va
conda activate firm-lingbot
MODEL_PATH=/models/lingbot-va-base \
DATASET_PATH=/data/haimiandian_50 \
DEVICE=cuda:0 \
bash script/run_extract_haimiandian_latents.sh --force
```

### LingBot-VA SFT

```bash
cd projects/lingbot-va
conda activate firm-lingbot
NGPU=4 \
MASTER_PORT=29521 \
CONFIG_NAME=haimiandian_train \
SAVE_ROOT=/outputs/lingbot_haimiandian_sft \
bash script/run_haimiandian_posttrain.sh
```

Main training config:

```text
wan_va/configs/va_haimiandian_train_cfg.py
```

Important defaults: `learning_rate=1e-5`, `batch_size=1`, `gradient_accumulation_steps=8`, `num_steps=5000`, and `save_interval=500`.

### LingBot-VA GRPO

```bash
cd projects/lingbot-va
conda activate firm-lingbot
export LINGBOT_GRPO_RESUME_FROM=/outputs/lingbot_haimiandian_sft/checkpoints/checkpoint_step_3500
bash script/run_lingbot_grpo_ablation.sh true 1.0 1.0 32 3000 grpo3000_final
```

Arguments:

```text
use_grpo reward_scale advantage_alpha buffer_size num_steps output_name
```

### ACT SFT

```bash
cd projects/act-grpo-datatest
conda activate firm-act
bash scripts/train_act_bc_dataset.sh tianqing_mixed --dry-run
bash scripts/train_act_bc_dataset.sh tianqing_mixed
```

Common overrides:

```bash
export CUDA_VISIBLE_DEVICES=0
export ACT_STEPS=3500
export ACT_BATCH_SIZE=16
export ACT_NUM_WORKERS=8
export ACT_LR=1e-5
```

### ACT GRPO

```bash
cd projects/act-grpo-datatest
conda activate firm-act
export ACT_GRPO_PRETRAINED_PATH=/outputs/act_sft/checkpoints/003500/pretrained_model
bash scripts/train_act_grpo_dataset.sh tianqing_mixed --dry-run
bash scripts/train_act_grpo_dataset.sh tianqing_mixed
```

### Pi0.5 GRPO

```bash
cd projects/act-grpo-datatest
conda activate firm-act
export PI05_GRPO_PRETRAINED_PATH=/outputs/pi05_sft/checkpoints/003500/pretrained_model
export CUDA_VISIBLE_DEVICES=0
bash scripts/train_pi05_grpo_dataset.sh tianqing_mixed --dry-run
bash scripts/train_pi05_grpo_dataset.sh tianqing_mixed
```

## Tests

DreamZero:

```bash
cd projects/dreamzero
python test_grpo.py
python test_dream_grpo_lingbot_parity.py
```

LingBot-VA:

```bash
cd projects/lingbot-va
python -m pytest wan_va/tests/test_grpo.py
```

ACT/Pi0.5:

```bash
cd ../..
python -m pytest tests/utils/test_act_grpo.py tests/policies/test_act_grpo_loss.py
```

Repository integrity check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_release.ps1
```
