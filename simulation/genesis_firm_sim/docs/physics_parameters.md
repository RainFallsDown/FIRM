# FIRM-Sim Physical Parameters

This file documents the single physical configuration shipped with FIRM-Sim. It contains the current object dimensions, masses, friction coefficients, deformable compliance, damping, collision, and solver settings. Earlier prototype parameters are not included in the release package.

## Launch

```bash
python scripts/launch_interactive_scene.py --scene cable_manipulation
```

Replace `cable_manipulation` with `instruction_manual`, `sponge_pad`, `tape_manipulation`, or `box_folding` to inspect another task.

## Applied Changes

| Object | Released configuration |
|---|---:|
| Sponge PBD areal density | 0.046 kg/m², approximately 1.30 g total |
| Tape geometry | 95/76.2 mm rigid annulus, approximately 32 g total |
| Box equivalent density | 61 kg/m³ |
| Manual geometry | 200 × 150 × 0.5 mm, five explicit 0.1 mm paper layers |
| Manual equivalent density | 800 kg/m³, approximately 12 g total |
| Manual binding | five full-size sheets in one rigid booklet body, without a central hinge |
| Box footprint | 245 × 180 mm |
| Box-lid hinge | 0.20 N·m/rad stiffness + damping |
| Table/object friction | 0.3–0.8 by contact class |
| Cable | PBD flexible line physically attached to the rigid mouse body |

The first four scenes use a fixed box lid that remains open outward at `2.80 rad`. Only the dedicated box-folding scene has a dynamic hinge and starts closed at `0 rad`. The cable, tape, sponge, and manual nominal spawn region is centered at `(0.55, 0.18)` m, directly in front of the box on its centerline.

The manual is `200 × 150 mm`. The box interior is approximately `237 × 172 mm` after subtracting the 4 mm walls, leaving 18.5 mm and 11 mm clearance per side. The five full-size paper layers form one rigid booklet proxy; page bending and inter-page sliding are not yet modeled.

The sponge is `196 × 144 mm` and the box footprint is `245 × 180 mm`, so the nominal sponge is exactly 80% of the box in both planar dimensions. A `high` object-translation perturbation moves the sponge by 40 mm, which is intentionally larger than the nominal 18–24.5 mm per-side clearance and can therefore place part of the pad outside the box. Use `nominal + none` when inspecting geometry rather than robustness.

## Solver and Collision Configuration

| Scene type | timestep | substeps | solver |
|---|---:|---:|---|
| Rigid-only | 0.010 s | 2 | Genesis rigid Newton, 100 iterations |
| PBD deformable | 0.004 s | 10 | PBD stretch 8 iterations, bending 3 iterations |

Rigid collision is enabled, rigid self-collision is disabled, and the PBD particle size is 5 mm. These settings are stored in `firm_sim/physical_parameters.py` rather than being hidden in a scene builder.

## Perturbation Schedule

| Level | Object translation | Fixture translation | Object yaw | Pose noise | RGB noise | Depth noise |
|---|---:|---:|---:|---:|---:|---:|
| `nominal` | 0 mm | 0 mm | 0° | 0 mm / 0° | 0 | 0 mm |
| `low` | 10 mm | 5 mm | 5° | 2.5 mm / 1.25° | 2.5/255 | 1.25 mm |
| `medium` | 20 mm | 10 mm | 10° | 5 mm / 2.5° | 5/255 | 2.5 mm |
| `medium_high` | 30 mm | 15 mm | 15° | 7.5 mm / 3.75° | 7.5/255 | 3.75 mm |
| `high` | 40 mm | 20 mm | 20° | 10 mm / 5° | 10/255 | 5 mm |

Each evaluation condition records `level`, `axis`, and `seed`. Translation uses the exact level magnitude with a seeded planar direction; yaw uses a seeded sign. RGB and depth values are maximum absolute uniform-noise amplitudes. PR-AUC should be reported separately for each axis. The `combined` axis is provided only for stress tests.

## Important Limitation

Genesis PBD cloth exposes solver compliance rather than a direct rod `EI`. The configuration records the physical target `EI=0.0012 N·m²`, line density `0.037 kg/m`, and bundled rest curvature `16.7 m⁻¹`, but the current cable remains a PBD-strip approximation. It should not yet be described as a fully identified Cosserat-rod model.

The current sponge is likewise a two-dimensional PBD sheet proxy. This reproduces planar bending and table contact, but it does not model three-dimensional compression and elastic recovery. A volumetric MPM/FEM model would be required for force-level compression studies.

The tape annulus uses an effective density of 1266 kg/m³ to match a 32 g total mass and annular inertia. The profile records a rolling-resistance target of `Crr=0.020`, but Genesis 1.2.3 does not expose a direct rolling-resistance parameter for this rigid proxy. Likewise, pose-noise amplitudes are recorded for a future observation interface; without a policy/VLA observation consumer, they do not alter scene geometry.

The confirmed object dimensions are recorded in `scene_parameters.md`; the executable values are stored in `firm_sim/physical_parameters.py` and `firm_sim/perturbations.py`.
