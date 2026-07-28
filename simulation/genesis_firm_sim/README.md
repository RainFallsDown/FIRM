# Genesis FIRM-Sim

[中文说明](README_zh-CN.md)

Genesis World implementation of the simulation scenes used by **FIRM: A Benchmark for Industrial Flexible-Object Robot Manipulation**.

This package focuses on reproducible scene construction and physical interaction. It does not include policy training, VLA integration, or the FIRM benchmark evaluation pipeline.

## Implemented Scenes

| Scene | Object model | Initial box state |
| --- | --- | --- |
| `instruction_manual` | five full-size paper layers in one bound booklet | fixed open outward |
| `sponge_pad` | thin PBD deformable pad | fixed open outward |
| `tape_manipulation` | mass-matched hollow rigid annulus | fixed open outward |
| `cable_manipulation` | flexible bundled PBD cable connected to a rigid mouse | fixed open outward |
| `box_folding` | articulated cardboard box with a hinged graspable lid | closed |

All scenes use the same table, box geometry, target region, and Tianqing robot placement. Task objects start on the tabletop in front of the box. The released configuration contains the current geometry, mass, friction, compliance, damping, collision, and solver settings.

## Quickstart

```bash
git clone https://github.com/RainFallsDown/FIRM.git
cd FIRM/simulation/genesis_firm_sim
git lfs pull --include="simulation/genesis_firm_sim/tianqing_urdf.zip"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the installation and list the scenes:

```bash
python -c "import genesis as gs; print(gs.__version__)"
python scripts/list_task_scenes.py
```

Launch an interactive scene:

```bash
python scripts/launch_interactive_scene.py --scene sponge_pad
```

Replace `sponge_pad` with `instruction_manual`, `tape_manipulation`, `cable_manipulation`, or `box_folding`.

## Reproducible Perturbations

The launcher supports deterministic perturbation levels and isolated axes:

```bash
python scripts/launch_interactive_scene.py \
  --scene sponge_pad \
  --perturbation-level high \
  --perturbation-axis object_translation \
  --seed 22
```

Available levels are `nominal`, `low`, `medium`, `medium_high`, and `high`, corresponding to normalized strengths `0`, `0.25`, `0.50`, `0.75`, and `1.00`. Available axes are `none`, `object_translation`, `fixture_translation`, `object_yaw`, `pose_noise`, `rgb_noise`, `depth_noise`, and `combined`.

## Offscreen Rendering

```bash
python scripts/render_scene_snapshot.py \
  --scene instruction_manual \
  --camera-preset overhead \
  --output outputs/instruction_manual.png
```

The renderer writes a PNG and a JSON sidecar containing the scene, physical configuration, seed, and perturbation values.

## Validation

Run the lightweight configuration and geometry tests:

```bash
python -m pytest -q
```

The tests check task registration, perturbation determinism, solver configuration, physical mass targets, hollow tape geometry, five-layer manual construction, box-lid initial states, tabletop contact, and object placement.

## Repository Layout

```text
firm_sim/
  perturbations.py
  physical_parameters.py
  runtime.py
  scenes/
    workspace.py
  tasks/
scripts/
  launch_interactive_scene.py
  list_task_scenes.py
  render_scene_snapshot.py
tests/
docs/
  physics_parameters.md
  scene_parameters.md
tianqing_urdf.zip
```

The robot archive is managed with Git LFS and extracted to `assets/tianqing_urdf/` on first launch. MJCF and OBJ collision proxies are generated from the versioned scene parameters at runtime.

## Model Scope

The current cable and sponge use PBD proxies. The cable preserves a physical connection to the rigid mouse, while the sponge reproduces thin-sheet bending and tabletop contact. See [`docs/physics_parameters.md`](docs/physics_parameters.md) for the exact solver settings, parameter mappings, perturbation ranges, and current model limitations.
