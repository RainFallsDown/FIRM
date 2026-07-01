# Repository Layout

This document maps the main functional areas in the FIRM repository. The repository is intentionally kept as a runnable LeRobot fork, so core package code remains under `src/lerobot/`.

## Top-Level Structure

```text
.
├── real_robot/                          # Real-robot training and Tianqing A2 deployment guide
├── simulation/                          # Benchmark, Genesis FIRM-Sim, and video benchmark tools
├── grpo_sft/                            # Extracted GRPO/SFT experiment code
├── archive/                             # Original archives and historical artifacts
├── src/lerobot/                         # Main installable Python package
├── examples/                            # Runnable examples inherited from / added to LeRobot
├── tests/                               # Unit and integration tests
├── docker/                              # Container build files
├── media/                               # README images and media assets
├── pyproject.toml                       # Package metadata, dependencies, CLI entry points
└── requirements-*.txt                   # Platform-specific dependency snapshots
```

## Real-Robot Training and Deployment

```text
src/lerobot/scripts/lerobot_train.py             # Offline training CLI implementation
src/lerobot/scripts/lerobot_record.py            # Data recording CLI
src/lerobot/scripts/lerobot_replay.py            # Dataset replay CLI
src/lerobot/scripts/lerobot_eval.py              # Policy evaluation CLI
src/lerobot/async_inference/                     # Policy server / robot client infrastructure
examples/rtc/eval_with_real_robot.py             # RTC real-robot deployment example
examples/rtc/eval_dataset.py                     # Offline RTC-style dataset evaluation
```

## Robot and Hardware Interfaces

```text
src/lerobot/robots/tianqing_a2/                  # tianqing_a2 robot interface and ROS2/ZMQ bridge
src/lerobot/robots/                              # Other robot adapters inherited from LeRobot
src/lerobot/cameras/                             # OpenCV, RealSense, ZMQ, and other camera interfaces
src/lerobot/motors/                              # Motor bus implementations
src/lerobot/teleoperators/                       # Teleoperation devices and adapters
```

The most relevant FIRM real-robot bridge files are:

```text
src/lerobot/robots/tianqing_a2/config_tianqing_a2.py
src/lerobot/robots/tianqing_a2/tianqing_a2.py
src/lerobot/robots/tianqing_a2/tianqing_a2_ros2.py
src/lerobot/robots/tianqing_a2/tianqing_a2_server.py
```

## Policies

```text
src/lerobot/policies/act/        # ACT
src/lerobot/policies/pi0/        # Pi0
src/lerobot/policies/pi05/       # Pi0.5
src/lerobot/policies/rtc/        # Real-Time Chunking components
src/lerobot/policies/diffusion/  # Diffusion policy
src/lerobot/policies/smolvla/    # SmolVLA
src/lerobot/policies/groot/      # GR00T-style policy support
src/lerobot/policies/wall_x/     # Wall-X
src/lerobot/policies/xvla/       # XVLA
```

## FIRM Benchmark

```text
simulation/benchmark/
├── README.md
├── EVALUATION_GUIDE.md
├── BENCHMARK_SCORING_EXPLAINED.md
├── configs/
│   ├── metric_extraction_config.json
│   └── scoring_config.json
└── firm_benchmark/
    ├── generate_sam3_masks_real.py
    ├── mask_metric_extractor.py
    ├── episode_scorer.py
    └── dap_eval.py
```

This module is independent from real-robot deployment code. It consumes recorded episodes and annotations, then produces metrics and reports.

## Simulation

```text
simulation/genesis_firm_sim/
├── firm_sim/
│   ├── runtime.py
│   ├── scenes/layer1_workspace.py
│   └── tasks/registry.py
├── scripts/
│   ├── list_task_scenes.py
│   ├── launch_interactive_scene.py
│   └── render_scene_snapshot.py
├── docs/scene_parameters.md
└── tests/test_task_registry.py
```

The Genesis scene package is designed for visual inspection and scene construction. Policy integration and benchmark scoring are separate layers.

## GRPO/SFT

`grpo_sft/` is the extracted and cleaned GRPO/SFT source tree. Notable contents:

```text
grpo_sft/projects/act-grpo-datatest/     # ACT/Pi0.5 BC, SFT, GRPO scripts and configs
grpo_sft/projects/dreamzero/             # DreamZero / VLA / SFT / GRPO code
grpo_sft/projects/lingbot-va/            # LingBot-VA resources
grpo_sft/projects/pi05/                  # Pi0.5 resources
grpo_sft/requirements/                   # Requirement files by project
grpo_sft/scripts/check_release.ps1       # Release check helper
```

The original archive included `firm/projects/act-grpo-datatest/resources/lerobot/`, which duplicated the main LeRobot source. That duplicate source tree has been excluded from `grpo_sft/`; GRPO scripts should depend on the root `src/lerobot/` package. `archive/firm-sft-grpo.rar` is kept as a source backup.

## Git Hygiene

Keep generated or local-only files out of Git:

```text
data/
outputs/
logs/
wandb/
checkpoints/
*.pt
*.pth
*.safetensors
assets/tianqing_urdf/
firm/
```

If a large binary is required for reproducibility, prefer Git LFS or an external release artifact rather than committing it directly.
