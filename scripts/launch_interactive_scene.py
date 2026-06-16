#!/usr/bin/env python3
"""Launch a Genesis FIRM scene interactively."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.scenes import (
    build_common_workspace_scene,
    build_instruction_manual_scene,
)
from firm_sim.tasks import build_task_scene, task_names


DROP_OFFSETS = {
    "cable_manipulation": (0.0, 0.0),
    "tape_manipulation": (0.10, 0.05),
}


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--layer", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--scene",
        choices=("common", *task_names()),
        default="instruction_manual",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--drop-height", type=float, default=0.18)
    args = parser.parse_args()

    if args.scene == "common":
        builder = build_common_workspace_scene
    else:
        builder = lambda show_viewer: build_task_scene(args.scene, show_viewer=show_viewer)
    scene, entities = builder(show_viewer=not args.headless)

    offset_x, offset_y = DROP_OFFSETS.get(args.scene, (0.0, 0.0))
    drop_entity_name = {
        "cable_manipulation": "cable_assembly",
        "tape_manipulation": "tape_proxy",
    }.get(args.scene)
    if drop_entity_name is not None and drop_entity_name in entities and args.drop_height > 0.0:
        drop_entity = entities[drop_entity_name]
        drop_entity.set_pos((offset_x, offset_y, args.drop_height), relative=True, zero_velocity=True)

    print(f"Genesis layer-{args.layer} scene '{args.scene}' is ready.")
    print(f"Loaded entities: {', '.join(sorted(entities.keys()))}")
    if args.headless:
        print("Running headless. Press Ctrl-C to stop.")
    else:
        print("Interactive viewer is open. Press Ctrl-C in the terminal to stop it.")

    steps = 0
    try:
        while args.max_steps <= 0 or steps < args.max_steps:
            scene.step()
            steps += 1
            if args.headless:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped interactive viewer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
