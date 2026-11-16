# FIRM-Sim on Genesis

Genesis environments and perturbation robustness evaluation for FIRM: A Benchmark for Industrial Flexible-Object Robot Manipulation.

The package provides five manipulation scenes, reproducible perturbation schedules, a policy rollout interface, and PR-AUC aggregation. Evaluation uses Genesis World 1.2.3. Task objects use geometric, articulated, or PBD proxies; results describe robustness within these models and the configured perturbation range.

## Installation

```bash
git clone https://github.com/RainFallsDown/FIRM.git
cd FIRM/simulation/genesis_firm_sim
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

The robot asset is stored in `tianqing_urdf.zip` and extracted into `assets/tianqing_urdf/` on first use. Evaluation requires a Genesis-supported rendering environment for RGB-D observations, including headless runs.

## Environments

| Scene | Object model |
|---|---|
| `instruction_manual` | Hinged booklet |
| `sponge_pad` | PBD cloth-like sheet |
| `tape_manipulation` | Rigid cylinder |
| `cable_manipulation` | Bundled cable and rigid mouse mesh |
| `box_folding` | Articulated box with a hinged lid |

```bash
python scripts/list_task_scenes.py
python scripts/launch_interactive_scene.py --scene instruction_manual
python scripts/render_scene_snapshot.py --scene box_folding --output outputs/box.png
```

Dimensions and asset conventions are documented in [scene_parameters.md](docs/scene_parameters.md).

## Perturbation evaluation

Generate one schedule and reuse it for every method:

```bash
python -m firm_sim.evaluation.cli schedule \
  --tasks tape_manipulation --seed 42 --output outputs/schedule.json
python -m firm_sim.evaluation.cli run \
  --schedule outputs/schedule.json --config configs/hold_position.json \
  --output outputs/hold_position
python -m firm_sim.evaluation.cli summarize --output outputs/hold_position
```

Inspect a saved physical perturbation in the scene viewer:

```bash
python scripts/launch_interactive_scene.py --schedule outputs/schedule.json --condition-index 80
```

The viewer shows physical scene changes; RGB-D noise is applied to policy observations.

Each task has three independently evaluated perturbation families, five strengths (`0, 0.25, 0.5, 0.75, 1`), and 20 rollouts per strength: 300 rollouts per method and task. `hold_position.json` runs a stationary robot baseline with an explicit object-center placement criterion. Configure a task's success criterion and policy before evaluating a manipulation method.

Each rollout starts in a fresh process with a seeded scene and robot initialization. The schedule stores resolved object, fixture, camera, and distractor perturbations. RGB-D noise is deterministic by sample and simulation step. Resume uses completed records only when the schedule, configuration, package source, robot archive, and runtime versions match. A failed or timed-out rollout stops execution and is retried on resume; incomplete runs do not produce PR-AUC.

`summary.json` contains success rates and each family's AUC; `summary.csv` contains method–task PR-AUC. `success_rates.csv` and task-specific PNG figures provide the curves. Individual records include initial and final rigid, joint, particle, and camera state. Logs are retained per condition.

See [evaluation.md](docs/evaluation.md) for distributions, metric definitions, policy integration, and success criteria.

## Package layout

```text
firm_sim/
  runtime.py           Genesis initialization
  scenes/              Workspace and object construction
  tasks/               Scene registry
  perturbations.py     Protocol and resolved sample generation
  evaluation/
    environment.py     Genesis actions, observations, and state
    adapters.py        Stationary baseline and geometric judge
    worker.py          Isolated rollout execution
    metrics.py         Success-rate and PR-AUC aggregation
    report.py          CSV, JSON, and curve figures
    cli.py             Schedule, run, and summarize commands
configs/               Evaluation configurations
scripts/               Interactive scene and snapshot tools
tests/                 Registry and evaluation checks
```

```bash
python -m unittest discover -s tests -v
```
