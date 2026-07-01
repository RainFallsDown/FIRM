#!/bin/bash

# 激活 conda 环境
source ${CONDA_ROOT:-/path/to/miniconda3}/etc/profile.d/conda.sh
conda activate sam3d-objects

# 设置环境变量
export PYTHONPATH="${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero:$PYTHONPATH"
export HYDRA_FULL_ERROR=1

# 配置路径
TIANQING_DATA_ROOT="${TIANQING_DATA_ROOT:-/path/to/tianqing}/tianqing_data/data_valid/A2p_dataset_0302_330"
OUTPUT_DIR="${DREAMZERO_ROOT:-/path/to/dreamzero}/checkpoints/dreamzero_tianqing_lora_5k"
TOKENIZER_DIR="/share/project/zjk/dreamzero/hf_cache/hub/models--google--umt5-xxl/snapshots/66cb9e7e85526fe440a945569e42c72fb6cbc0ad"

# Wan 模型路径
WAN_CKPT_DIR="/share/project/zjk/dreamzero/hf_cache/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279"

# 预训练模型路径 (修正: 直接指向 pre_models_new 目录)
PRETRAINED_MODEL="/share/project/zjk/dreamzero/pre_models_new"

# 启动训练
torchrun --nproc_per_node=4 --standalone groot/vla/experiment/experiment.py \
  report_to=none \
  data=dreamzero/tianqing_relative \
  wandb_project=dreamzero \
  train_architecture=lora \
  num_frames=33 \
  action_horizon=24 \
  num_views=3 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate=1e-5 \
  training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
  save_steps=2500 \
  training_args.warmup_ratio=0.05 \
  output_dir=$OUTPUT_DIR \
  per_device_train_batch_size=1 \
  max_steps=5000 \
  weight_decay=1e-5 \
  save_total_limit=10 \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=false \
  dataloader_num_workers=1 \
  image_resolution_width=320 \
  image_resolution_height=176 \
  save_lora_only=true \
  max_chunk_size=4 \
  frame_seqlen=880 \
  save_strategy=steps \
  tianqing_data_root=$TIANQING_DATA_ROOT \
  dit_version=$WAN_CKPT_DIR \
  text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
  image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
  tokenizer_path=$TOKENIZER_DIR \
  pretrained_model_path=$PRETRAINED_MODEL \
  ++action_head_cfg.config.skip_component_loading=true \
  ++action_head_cfg.config.defer_lora_injection=true
