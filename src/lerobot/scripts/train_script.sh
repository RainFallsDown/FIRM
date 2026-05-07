dataset=pc_shuomingshu_task_7_485_1773627018
project_name=${dataset}_diffusion_default_steps_150000_mini_batch_16_img_transform_no_crop_resize_224_resnet_pretrain 
data_root=/DATA/disk0/home/yangwu/data

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --gpu_ids 4,5,6,7 \
  --num_machines 1 \
  $(which lerobot-train) \
  --dataset.repo_id=armpi_data_lerobot \
  --dataset.image_transforms.enable=true \
  --dataset.root=${data_root}/${dataset} \
  --wandb.enable=true \
  --wandb.project=${dataset} \
  --num_workers=12 \
  --output_dir=outputs/train/${dataset}/${project_name} \
  --job_name=${project_name} \
  --policy.type=diffusion \
  --policy.crop_shape null \
  --policy.pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1 \
  --policy.use_group_norm false \
  --steps 150000 \
  --batch_size 24
