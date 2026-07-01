#!/bin/bash
# DreamZero Tianqing Training Script

# Activate conda environment
source ${CONDA_ROOT:-/path/to/miniconda3}/etc/profile.d/conda.sh
conda activate sam3d-objects

export HYDRA_FULL_ERROR=1
export PYTHONPATH=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero:$PYTHONPATH

# ============ CONFIGURATION ============
TIANQING_DATA_ROOT=${TIANQING_DATA_ROOT:-"${TIANQING_DATA_ROOT:-/path/to/tianqing}/tianqing_data/data_valid/A2p_dataset_0302_330"}
OUTPUT_DIR=${OUTPUT_DIR:-"${DREAMZERO_ROOT:-/path/to/dreamzero}/checkpoints/dreamzero_tianqing_lora_5k"}
NUM_GPUS=4
# Use HuggingFace cache directory
HF_CACHE_DIR="/share/project/zjk/dreamzero/hf_cache/hub/models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/6b73f84e66371cdfe870c72acd6826e1d61cf279"
TOKENIZER_DIR="/share/project/zjk/umt5-xxl"
PRETRAINED_MODEL="/share/project/zjk/dreamzero/pre_models_new"
# =======================================

if [ ! -d "$TIANQING_DATA_ROOT" ]; then
    echo "ERROR: Tianqing dataset not found at $TIANQING_DATA_ROOT"
    exit 1
fi

echo "=== Starting Tianqing Training ==="
echo "Dataset: $TIANQING_DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Model Cache: $HF_CACHE_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "=================================="

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/tianqing_relative \
    wandb_project=dreamzero_tianqing \
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
    dit_version=$HF_CACHE_DIR \
    text_encoder_pretrained_path=$HF_CACHE_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$HF_CACHE_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$HF_CACHE_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
