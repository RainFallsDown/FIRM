# FIRM

FIRM is a robotics repository organized around industrial flexible-object manipulation. It has three main tracks: real-robot training/deployment, simulation and benchmark evaluation, and GRPO/SFT experiments.

## Entry Points

```text
real_robot/       Real-robot training, Tianqing A2 deployment, RTC inference
simulation/       Genesis FIRM-Sim scenes, FIRM benchmark, video benchmark
grpo_sft/         ACT/Pi0.5, DreamZero, and LingBot-VA SFT/GRPO experiments
archive/          Original archives and historical release artifacts
src/lerobot/      Installable Python package and LeRobot fork source
examples/         LeRobot examples and RTC examples
tests/            Unit and integration tests
docs/             Additional repository layout notes
```

## Quick Start

Python 3.10 is recommended.

```bash
pip install -e .
pip install -e ".[async]"
pip install -e ".[tianqing_a2]"
```

For real-robot deployment, start with [real_robot/README.md](real_robot/README.md).

For simulation and evaluation, start with [simulation/README.md](simulation/README.md).

For GRPO/SFT experiments, start with [grpo_sft/README.md](grpo_sft/README.md).

## Design Note

`src/lerobot/` remains the Python package source tree. Robot implementations such as `tianqing_a2` are not moved out of `src/`, because CLI commands, config registration, tests, and external imports depend on that package path. The top-level functional folders provide clean entry points and documentation without breaking installability.

## Git Hygiene

Do not commit local datasets, outputs, model weights, logs, caches, temporary extraction directories, or machine-private configuration. The original GRPO/SFT archive is stored under `archive/`; day-to-day work should use the organized source under `grpo_sft/`.
