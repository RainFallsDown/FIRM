# Simulation and Benchmark

This folder groups the simulation scenes and benchmark tooling used by FIRM.

## Layout

```text
benchmark/              FIRM benchmark evaluation utilities
genesis_firm_sim/       Genesis World FIRM-Sim scenes
video_benchmark.py      Video encoding benchmark utility inherited from LeRobot
VIDEO_BENCHMARK.md      Video benchmark notes
```

## Genesis FIRM-Sim

```bash
cd simulation/genesis_firm_sim
python scripts/list_task_scenes.py
python scripts/launch_interactive_scene.py --scene instruction_manual
python scripts/render_scene_snapshot.py --scene box_folding --output outputs/box_folding.png
python -m unittest tests/test_task_registry.py
```

## FIRM Benchmark

```bash
cd simulation/benchmark
pip install -r requirements.txt
python firm_benchmark/dap_eval.py --help
```

The benchmark pipeline covers object mask generation, mask-based metric extraction, episode scoring, and report generation. See `benchmark/README.md` for details.
