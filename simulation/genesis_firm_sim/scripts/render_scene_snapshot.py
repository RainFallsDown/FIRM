#!/usr/bin/env python3
"""Render an offscreen snapshot for a Genesis FIRM scene."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.perturbations import apply_depth_noise, apply_rgb_noise
from firm_sim.tasks import (
    build_task_scene,
    perturbation_axes,
    perturbation_levels,
    sample_perturbation,
    task_names,
)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--scene", choices=task_names(), default="instruction_manual")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depth-output")
    parser.add_argument("--metadata-output")
    parser.add_argument(
        "--camera-preset",
        choices=("perspective", "overhead"),
        default="perspective",
    )
    parser.add_argument("--perturbation-level", choices=perturbation_levels(), default="nominal")
    parser.add_argument("--perturbation-axis", choices=perturbation_axes(), default="none")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    perturbation = sample_perturbation(
        level=args.perturbation_level,
        axis=args.perturbation_axis,
        seed=args.seed,
    )

    if args.camera_preset == "overhead":
        camera_specs = {
            "res": (1280, 960),
            "pos": (0.55, -0.02, 1.25),
            "lookat": (0.55, 0.04, 0.71),
            "fov": 45,
            "GUI": False,
        }
    else:
        camera_specs = {
            "res": (1280, 960),
            "pos": (2.4, -1.7, 1.8),
            "lookat": (0.55, 0.02, 0.67),
            "fov": 36,
            "GUI": False,
        }

    scene, entities = build_task_scene(
        args.scene,
        show_viewer=False,
        camera_specs=camera_specs,
        perturbation=perturbation,
    )

    for _ in range(args.steps):
        scene.step()

    camera = entities["snapshot_camera"]
    rendered = camera.render(rgb=True, depth=args.depth_output is not None)
    if isinstance(rendered, tuple):
        rgb = rendered[0]
        depth = rendered[1] if args.depth_output is not None else None
    else:
        rgb = rendered
        depth = None
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = apply_rgb_noise(image, perturbation.rgb_noise, seed=args.seed + 10_001)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)
    if args.depth_output is not None:
        if depth is None:
            raise RuntimeError("Genesis did not return a depth image")
        depth_image = apply_depth_noise(
            np.asarray(depth),
            perturbation.depth_noise_m,
            seed=args.seed + 20_001,
        )
        depth_output = Path(args.depth_output)
        depth_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_output, depth_image)

    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(
            {
                "scene": args.scene,
                "physics_configuration": "released_parameters",
                "camera_preset": args.camera_preset,
                "steps": args.steps,
                "perturbation": perturbation.as_dict(),
                "rgb_output": str(output),
                "depth_output": args.depth_output,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(metadata_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
