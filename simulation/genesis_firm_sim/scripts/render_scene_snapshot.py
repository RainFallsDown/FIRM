#!/usr/bin/env python3
"""Render an offscreen snapshot for a Genesis FIRM scene."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.scenes import build_common_workspace_scene
from firm_sim.tasks import build_task_scene, task_names


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--scene", choices=("common", *task_names()), default="instruction_manual")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    camera_specs = {
        "res": (1280, 960),
        "pos": (2.4, -1.7, 1.8),
        "lookat": (0.55, 0.02, 0.67),
        "fov": 36,
        "GUI": False,
    }

    if args.scene == "common":
        scene, entities = build_common_workspace_scene(show_viewer=False, camera_specs=camera_specs)
    else:
        scene, entities = build_task_scene(args.scene, show_viewer=False, camera_specs=camera_specs)

    for _ in range(args.steps):
        scene.step()

    camera = entities["snapshot_camera"]
    rgb = camera.render()
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
