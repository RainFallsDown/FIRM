#!/bin/bash
export PATH=/root/.local/bin:$PATH
export AGIBOT_DATA_ROOT=/share/project/cyfu/agibot_train_data
export PYTHONPATH=${DREAMZERO_ROOT:-/path/to/dreamzero}/code/dreamzero:$PYTHONPATH

torchrun --nproc_per_node=4 --master_port=29500   groot/vla/train_vla.py   report_to=wandb   data=dreamzero/agibot_relative   wandb_project=dreamzero   train_architecture=lora   num_frames=33   action_horizon=24   num_views=3   model=dreamzero/vla   model/dreamzero/action_head=wan_flow_matching_action_tf   model/dreamzero/transform=dreamzero_cotrain   num_frame_per_block=2   num_action_per_block=24   num_state_per_block=1   seed=42   training_args.learning_rate=1e-5   training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json   training_args.per_device_train_batch_size=1   training_args.gradient_accumulation_steps=1   training_args.num_train_epochs=1   training_args.max_steps=5000   training_args.save_steps=500   training_args.logging_steps=10   training_args.output_dir=outputs/agibot_lora   training_args.run_name=agibot_lora_training   agibot_data_root=$AGIBOT_DATA_ROOT
