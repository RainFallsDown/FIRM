#!/usr/bin/env python3
"""List registered FIRM task families."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.registry import make_task, task_names


def main() -> int:
    print("Registered FIRM task families:")
    for name in task_names():
        task = make_task(name)
        status = "placeholder" if task.placeholder else "implemented"
        print(f"- {task.name}: {status} | {task.physical_challenge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
