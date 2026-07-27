#!/usr/bin/env python3
"""Launch a Genesis FIRM scene interactively."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.tasks import (
    build_task_scene,
    perturbation_axes,
    perturbation_levels,
    sample_perturbation,
    task_names,
)


DROP_OFFSETS = {
    "cable_manipulation": (0.0, 0.0),
    "tape_manipulation": (0.10, 0.05),
}


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene",
        choices=task_names(),
        default="instruction_manual",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--drop-height", type=float, default=0.18)
    parser.add_argument("--perturbation-level", choices=perturbation_levels(), default="nominal")
    parser.add_argument("--perturbation-axis", choices=perturbation_axes(), default="none")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    perturbation = sample_perturbation(
        level=args.perturbation_level,
        axis=args.perturbation_axis,
        seed=args.seed,
    )

    builder = lambda show_viewer: build_task_scene(
        args.scene,
        show_viewer=show_viewer,
        perturbation=perturbation,
    )
    scene, entities = builder(show_viewer=not args.headless)

    offset_x, offset_y = DROP_OFFSETS.get(args.scene, (0.0, 0.0))
    drop_entity_name = {
        "tape_manipulation": "tape_proxy",
    }.get(args.scene)
    if args.scene == "cable_manipulation" and "cable_connected_line_proxy" in entities and args.drop_height > 0.0:
        line = entities["cable_connected_line_proxy"]
        particles = line.get_particles_pos().detach().cpu().numpy()
        particles += np.array([offset_x, offset_y, args.drop_height], dtype=np.float32)
        line.set_particles_pos(particles)
        mouse = entities["cable_connected_mouse"]
        mouse_pos = mouse.get_pos().detach().cpu().numpy()
        mouse_pos += np.array([offset_x, offset_y, args.drop_height], dtype=np.float32)
        mouse.set_pos(
            mouse_pos,
            relative=False,
            zero_velocity=True,
        )
    elif drop_entity_name is not None and drop_entity_name in entities and args.drop_height > 0.0:
        drop_entity = entities[drop_entity_name]
        position = drop_entity.get_pos().detach().cpu().numpy()
        position += np.array([offset_x, offset_y, args.drop_height], dtype=np.float32)
        drop_entity.set_pos(position, relative=False, zero_velocity=True)

    print(
        f"Genesis FIRM scene '{args.scene}' is ready "
        "with the released FIRM physical parameters."
    )
    print(f"Loaded entities: {', '.join(sorted(entities.keys()))}")
    print("Perturbation sample:")
    print(json.dumps(perturbation.as_dict(), indent=2, sort_keys=True))
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
