# FIRM SFT/GRPO

[English](README.md) | [中文](README_CN.md)

FIRM SFT/GRPO 提供 DreamZero、LingBot-VA、ACT 与 Pi0.5 四条机器人策略训练线中的监督微调（SFT）与离线 GRPO 风格优化代码。项目面向具身操作任务中的视觉-动作建模，覆盖 LeRobot 数据读取、视频 latent 准备、SFT 训练、GRPO 奖励加权以及离线评估等关键流程。

项目将不同上游训练框架保留在各自子目录中，同时在顶层提供统一的环境配置、模型准备、数据格式与启动命令说明。

当前副本已集成到主 FIRM 仓库的 `grpo_sft/`。原始归档中的重复 `projects/act-grpo-datatest/resources/lerobot` 快照已移除，请改用仓库根目录的 `src/lerobot/` 包。

## 代码结构

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

`projects/dreamzero`、`projects/lingbot-va` 与 `projects/act-grpo-datatest` 尽量保留原始项目布局。`projects/pi05` 存放 Pi0.5 专用配置和脚本副本。ACT/Pi0.5 训练应使用主 FIRM 仓库根目录的 `src/lerobot/` 包，而不是嵌套的 LeRobot 副本。

## 功能模块

### DreamZero

DreamZero 位于 `projects/dreamzero`，包含基于 Wan2.2-TI2V-5B 的 WAM/FIRM 训练路径。

主要入口：

- SFT：`train_wam_sft.sh`
- GRPO：`run_grpo3500_compare.sh`

SFT 阶段在 FIRM/Haimiandian 数据上执行行为克隆式监督微调。当前脚本采用 LoRA 训练、DeepSpeed ZeRO-3、四卡分布式执行，并默认训练 5000 step。GRPO 阶段从 SFT checkpoint 继续优化，默认额外训练 3500 step，并调用 `SFT_test/evaluate_zero_sft_grpo.py` 对 zero-shot、SFT 与 GRPO 策略进行对比评估。

DreamZero GRPO 的核心实现位于：

```text
projects/dreamzero/groot/vla/grpo_simple.py
projects/dreamzero/groot/vla/experiment/base.py
```

当前实现将 DreamZero 的 GRPO 奖励机制与 LingBot-VA 的 action-only intrinsic reward 对齐。奖励由一阶动作平滑项、二阶 jerk 项、动作幅值正则项和夹爪一致性项组成。`GRPOWeightBuffer` 维护滑动奖励历史，将当前奖励转化为 advantage，并通过指数映射与上下界裁剪得到训练 loss 权重。

### LingBot-VA

LingBot-VA 位于 `projects/lingbot-va`，包含视频-动作 latent 提取、SFT/post-training、GRPO 消融训练与评估工具。

主要文件：

```text
script/run_extract_haimiandian_latents.sh
script/run_haimiandian_posttrain.sh
script/run_lingbot_grpo_ablation.sh
wan_va/train.py
wan_va/grpo.py
wan_va/dataset/lerobot_latent_dataset.py
```

LingBot-VA GRPO 不执行在线 rollout，而是在训练 batch 内根据动作序列计算 intrinsic reward，并对原始训练 loss 进行加权。奖励计算与滑动权重缓存位于 `wan_va/grpo.py`，训练循环中的调用逻辑位于 `wan_va/train.py`。

### ACT

ACT 位于 `projects/act-grpo-datatest`。该模块在 LeRobot 训练入口上扩展了 ACT SFT 与 ACT GRPO 脚本：

```text
scripts/train_act_bc_dataset.sh
scripts/train_act_grpo_dataset.sh
../src/lerobot/utils/act_grpo.py
../src/lerobot/configs/train.py
../src/lerobot/scripts/lerobot_train.py
```

ACT-GRPO 实现为 offline batch intrinsic reward-weighted BC，不依赖在线环境交互。训练时对每个 batch 结合 BC 质量、动作平滑性、加速度和夹爪一致性构造 intrinsic reward，再通过 batch 内 z-score advantage 生成样本权重。默认奖励权重为：

```text
bc_reward_weight=0.45
smooth_reward_weight=0.25
accel_reward_weight=0.20
gripper_reward_weight=0.10
```

### Pi0.5

Pi0.5 复用 ACT/LeRobot 的 `--use_grpo` 训练路径。

核心代码：

```text
../src/lerobot/policies/pi05/
projects/act-grpo-datatest/scripts/train_pi05_grpo_dataset.sh
```

`projects/pi05` 保存 Pi0.5 配置、基础策略配置文件和脚本副本，便于单独查阅。实际训练建议使用 `projects/act-grpo-datatest`，以保持数据集配置、共享脚本和 LeRobot 源码的一致性。

## 环境配置

完整硬件与软件需求见 [ENVIRONMENT_REQUIREMENTS.txt](ENVIRONMENT_REQUIREMENTS.txt)。

三条主要训练栈依赖的 PyTorch 与 Transformers 版本不兼容，建议分别创建 Python 环境：

```text
DreamZero:   Python 3.11, torch==2.8.0, transformers==4.51.3
LingBot-VA:  Python 3.10, torch==2.9.0, transformers>=4.55.4
ACT/Pi0.5:   Python 3.10, torch>=2.2.1,<2.8.0, local LeRobot 0.4.3 tree
```

在 `firm` 根目录使用以下 requirements 文件安装：

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

顶层 `requirements.txt` 默认指向 ACT/Pi0.5 环境，可用于轻量默认安装：

```bash
pip install -r requirements.txt
```

ACT/Pi0.5 训练脚本会读取 `dataset_common.sh` 期望的环境激活脚本，可按实际 conda 路径创建：

```bash
cd projects/act-grpo-datatest
mkdir -p env
cat > env/activate_act_grpo.sh <<'EOF'
source /path/to/miniconda3/bin/activate firm-act
EOF
```

## 模型准备

训练脚本通过命令行参数、环境变量或脚本默认变量读取模型路径。建议服务器侧采用如下统一目录：

```text
/models/
  dreamzero/
  lingbot-va-base/
  act/
  pi05/
```

### DreamZero 模型

DreamZero 需要 Wan2.2/Wan2.1 相关组件和 tokenizer：

```text
Wan2.2-TI2V-5B
Wan2.1-I2V-14B-480P text encoder
Wan2.1-I2V-14B-480P image encoder
Wan2.2 VAE
umt5-xxl tokenizer
DreamZero or FIRM SFT checkpoint
```

示例下载命令：

```bash
mkdir -p /models/dreamzero
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /models/dreamzero/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir /models/dreamzero/Wan2.1-I2V-14B-480P
huggingface-cli download google/umt5-xxl \
  --local-dir /models/dreamzero/umt5-xxl
```

运行 DreamZero 脚本前，需要确认以下路径参数：

```text
dit_version
text_encoder_pretrained_path
image_encoder_pretrained_path
vae_pretrained_path
tokenizer_path
pretrained_model_path
```

### LingBot-VA 模型

LingBot-VA 基础模型推荐目录结构：

```text
/models/lingbot-va-base/
  transformer/
  vae/
  text_encoder/
  tokenizer/
```

latent 提取使用 `MODEL_PATH`，GRPO 从 SFT checkpoint 恢复时使用 `LINGBOT_GRPO_RESUME_FROM`：

```bash
export MODEL_PATH=/models/lingbot-va-base
export LINGBOT_GRPO_RESUME_FROM=/outputs/lingbot_sft/checkpoints/checkpoint_step_3500
```

### ACT 与 Pi0.5 模型

ACT GRPO 从 SFT 后的 `pretrained_model` 目录恢复：

```text
/outputs/act_sft/checkpoints/003500/pretrained_model/
  config.json
  model.safetensors
  policy_preprocessor...
  policy_postprocessor...
```

Pi0.5 GRPO 使用相同的 `pretrained_model` 结构：

```bash
export ACT_GRPO_PRETRAINED_PATH=/outputs/act_sft/checkpoints/003500/pretrained_model
export PI05_GRPO_PRETRAINED_PATH=/outputs/pi05_sft/checkpoints/003500/pretrained_model
```

## 数据集格式

### LeRobot v3

ACT 与 Pi0.5 使用 LeRobot v3 数据集。最小目录结构如下：

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

数据集条目配置在 `projects/act-grpo-datatest/configs/datasets/*.env` 中：

```bash
DATASET_KEY=tianqing_mixed
DATASET_REPO_ID=tianqing/tianqing_mixed
DATASET_ROOT=/data/tianqing_mixed
OUTPUT_GROUP=tianqing_mixed
```

新增数据集时，可复制已有 `.env` 文件，并更新 `DATASET_KEY`、`DATASET_REPO_ID`、`DATASET_ROOT` 与 `OUTPUT_GROUP`。

### DreamZero 数据集

DreamZero 使用 LeRobot 数据，并额外依赖 GEAR/DreamZero 元数据：

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

元数据转换示例：

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

Haimiandian 数据配置为：

```text
projects/dreamzero/groot/vla/configs/data/dreamzero/haimiandian.yaml
```

该配置使用 `video.head`、`state.left_arm`、`state.right_arm`、`state.gripper` 以及对应 action keys。

### LingBot-VA latent 数据集

LingBot-VA 读取带有预计算视频 latents 的 LeRobot 数据：

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

`episodes.jsonl` 可包含 `action_config` 字段，其中定义 `start_frame`、`end_frame` 与 `action_text`。若该字段不存在，数据加载器会回退使用 episode 级任务文本。

## 训练命令

### DreamZero SFT

```bash
cd projects/dreamzero
conda activate firm-dreamzero
bash train_wam_sft.sh
```

关键设置：

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

需要在脚本或 shell 中设置 `SFT_CKPT`，使其指向可读取的 SFT checkpoint。默认 GRPO 设置为：

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

### LingBot-VA latent 提取

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

主训练配置：

```text
wan_va/configs/va_haimiandian_train_cfg.py
```

重要默认值包括 `learning_rate=1e-5`、`batch_size=1`、`gradient_accumulation_steps=8`、`num_steps=5000` 与 `save_interval=500`。

### LingBot-VA GRPO

```bash
cd projects/lingbot-va
conda activate firm-lingbot
export LINGBOT_GRPO_RESUME_FROM=/outputs/lingbot_haimiandian_sft/checkpoints/checkpoint_step_3500
bash script/run_lingbot_grpo_ablation.sh true 1.0 1.0 32 3000 grpo3000_final
```

参数含义：

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

常用覆盖参数：

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

## 测试

DreamZero：

```bash
cd projects/dreamzero
python test_grpo.py
python test_dream_grpo_lingbot_parity.py
```

LingBot-VA：

```bash
cd projects/lingbot-va
python -m pytest wan_va/tests/test_grpo.py
```

ACT/Pi0.5：

```bash
cd ../..
python -m pytest tests/utils/test_act_grpo.py tests/policies/test_act_grpo_loss.py
```

项目完整性检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_release.ps1
```
