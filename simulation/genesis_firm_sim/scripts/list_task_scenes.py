#!/usr/bin/env python3
"""List the currently implemented Genesis FIRM task scenes."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.tasks import (
    get_task_spec,
    perturbation_axes,
    perturbation_levels,
    task_names,
)


def main() -> int:
    print("Implemented Genesis task scenes:")
    for name in task_names():
        spec = get_task_spec(name)
        print(f"- {spec.name}: {spec.object_class} | {spec.description}")
    print("Physics configuration: released FIRM physical parameters")
    print(f"Perturbation levels: {', '.join(perturbation_levels())}")
    print(f"Perturbation axes: {', '.join(perturbation_axes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
