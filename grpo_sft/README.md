# GRPO and SFT

This folder contains the GRPO/SFT experiment code extracted from the original archive.

## Layout

```text
projects/act-grpo-datatest/     ACT/Pi0.5 BC, SFT, and GRPO scripts
projects/dreamzero/             DreamZero / VLA / WAM SFT and GRPO code
projects/lingbot-va/            LingBot-VA SFT, GRPO, and evaluation utilities
projects/pi05/                  Pi0.5 configs and launcher wrappers
requirements/                   Dependency lists by project
scripts/                        Release/check helper scripts
```

The duplicated LeRobot source snapshot from the original archive was removed. ACT/Pi0.5 GRPO uses the root package under `../src/lerobot/`.

## ACT-GRPO Integration

The main package now includes the minimal ACT-GRPO hooks:

```text
../src/lerobot/utils/act_grpo.py
../src/lerobot/configs/train.py
../src/lerobot/policies/act/modeling_act.py
../src/lerobot/scripts/lerobot_train.py
```

Example:

```bash
lerobot-train \
  --dataset.repo_id=<dataset_repo_id> \
  --dataset.root=/path/to/lerobot_dataset \
  --policy.type=act \
  --use_grpo=true \
  --grpo_beta=1.0 \
  --output_dir=outputs/train/<run_name>
```

See `README_GRPO_SOURCE.md` for the original extracted project guide.
