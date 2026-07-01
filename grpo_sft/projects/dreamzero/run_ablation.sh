#!/bin/bash
# GRPO Ablation Study
# 5 组实验，每组 500 步，约 30 分钟/组

BASE_DIR=${DREAMZERO_ROOT:-/path/to/dreamzero}
CODE_DIR=/code/bc-grpo
OUTPUT_BASE=/outputs/ablation

mkdir -p 

run_exp() {
    local EXP_NAME=
    local USE_GRPO=
    local GRPO_WEIGHT=
    local EXTRA_ARGS=

    echo "===== 开始实验:  ====="
    local OUT_DIR=/
    rm -rf  && mkdir -p 

    cd 
    torchrun --nproc_per_node=4 --master_port=29510         groot/vla/experiment/experiment.py         data=dreamzero/haimiandian         train_architecture=lora         model/dreamzero/action_head=wan_flow_matching_action_tf_wan22         wandb_project=haimiandian_ablation         tokenizer_path=./checkpoints/umt5-xxl         dit_version=/models/Wan2.2-TI2V-5B         text_encoder_pretrained_path=/models/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth         image_encoder_pretrained_path=/models/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth         vae_pretrained_path=/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth         training_args.output_dir=         training_args.run_name=         training_args.report_to=none         training_args.per_device_train_batch_size=1         training_args.gradient_accumulation_steps=2         training_args.max_steps=500         training_args.save_steps=500         training_args.logging_steps=10         training_args.bf16=true         training_args.dataloader_num_workers=1         training_args.deepspeed=groot/vla/configs/deepspeed/zero3.json         use_grpo=         grpo_weight=         grpo_reward_scale=1.0         grpo_reward_shaper_scale=0.01                  > /training.log 2>&1

    echo "===== 完成:  ====="
    grep -E 'loss.*grad_norm' /training.log | tail -3
    grep -E 'grpo_reward|grpo_advantage|grpo_weight[^_]' /training.log | tail -3
    echo ""
}

export CUDA_VISIBLE_DEVICES=0,1,2,3

# A: 纯 BC baseline
run_exp A_pure_bc false 0.0

# B: 当前配置（grpo_weight=0.1, alpha=1.0, buf=32）
run_exp B_grpo_w01 true 0.1

# C: 更强 GRPO（grpo_weight=0.3）
run_exp C_grpo_w03 true 0.3

# D: 更激进 advantage（alpha=2.0，需要改代码）
# 暂时跳过，需要修改 base.py 中的 alpha 参数

# E: 更短历史窗口（buf=8，需要改代码）
# 暂时跳过，需要修改 base.py 中的 buffer_size

echo "===== Ablation 完成 ====="
echo "结果目录: "
ls /
