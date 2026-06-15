# FIRM-Sim Genesis

Genesis-first reconstruction of the **FIRM-Sim** portion of the paper **FIRM: A Benchmark for Industrial Flexible-Object Robot Manipulation**.

This repository focuses on a practical goal: make the core FIRM task scenes runnable, inspectable, and easy to extend in **Genesis World**, before plugging in policies or VLA systems.

## What This Repo Covers

From the paper, FIRM targets industrial mixed-stiffness manipulation under fixtures, containers, and tight spatial constraints. In this repo, we currently focus on the **simulation scene layer** of that benchmark:

- shared workcell with table, box, target region, and robot
- Genesis-only task scene construction
- interactive scene launching for rapid visual inspection
- proxy object setup for flexible and semi-rigid manipulation tasks

This repo currently does **not** include:

- policy training
- VLA integration
- DAP evaluation implementation
- full FIRM-Real / FIRM-Web benchmark release

## Implemented Scenes

All scenes share the same workcell and robot placement, while keeping task objects independent:

| Scene | Paper object type | Current Genesis proxy |
| --- | --- | --- |
| `instruction_manual` | flexible / semi-rigid manual | bi-fold hinged sheet proxy |
| `sponge_pad` | thin deformable pad | cloth-like deformable sheet |
| `tape_manipulation` | tape roll | rigid cylindrical proxy |
| `cable_manipulation` | bundled cable + rigid end object | bundled cable mesh + rigid mouse proxy |
| `box_folding` | cardboard box component | articulated box with hinged lid |

## Repository Layout

```text
firm_sim/
  runtime.py                # Genesis init helper
  scenes/
    layer1_workspace.py     # shared workcell + all scene builders
  tasks/
    registry.py             # task registry and scene metadata
scripts/
  list_task_scenes.py       # list available scenes
  launch_layer1_interactive.py
  render_scene_snapshot.py
docs/
  scene_parameters.md
  genesis_advanced_ik_notes.md
  genesis_firm_sim_plan.md
PYBULLET/                   # archived legacy baseline, not the active path
```

## Quickstart

### 1. Environment

Install Genesis World and make sure this succeeds:

```bash
python -c "import genesis; print('Genesis OK')"
```

If you want to version the robot asset in GitHub, install Git LFS:

```bash
git lfs install
```

### 2. Clone

```bash
git clone https://github.com/RainFallsDown/FIRM.git
cd FIRM
git lfs pull
```

### 3. Launch a Scene

List all implemented scenes:

```bash
python scripts/list_task_scenes.py
```

Start an interactive viewer:

```bash
python scripts/launch_layer1_interactive.py --scene instruction_manual
python scripts/launch_layer1_interactive.py --scene sponge_pad
python scripts/launch_layer1_interactive.py --scene tape_manipulation
python scripts/launch_layer1_interactive.py --scene cable_manipulation
python scripts/launch_layer1_interactive.py --scene box_folding
```

Render an offscreen snapshot:

```bash
python scripts/render_scene_snapshot.py --scene box_folding --output outputs/box_folding.png
```

Run the lightweight registry test:

```bash
python -m unittest tests/test_task_registry.py
```

## Robot Asset

The shared workcell currently uses the local robot asset:

- `tianqing_urdf.zip`

At runtime, the code automatically extracts it into:

- `assets/tianqing_urdf/`

This extracted folder is intentionally ignored in Git because it is runtime-generated.

## Notes on Paper Assets

The local paper copies in this workspace are for internal reference while building scenes.

- Do **not** publicly upload confidential or reviewer-copy manuscript PDFs.
- In particular, derived files such as translated or dual-language PDFs are ignored by default.

## Active Conventions

- **Genesis World** is the only active simulation path.
- `PYBULLET/` is retained only as an archived legacy baseline.
- New scenes, assets, and runtime scripts should stay on the Genesis path.

## Uploading to GitHub

This workspace is prepared to be uploaded into:

- upstream repo: `https://github.com/RainFallsDown/FIRM`
- contributor account used for push/PR: `ChinChilla-HTL`

See [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md) for the exact direct-push and fork-plus-PR flows.
