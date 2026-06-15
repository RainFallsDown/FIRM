#!/usr/bin/env python3
"""Run a single task family in GUI or headless mode."""

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.env import BenchmarkEnv, BenchmarkEnvConfig
from firm_sim.exceptions import PlaceholderTaskError
from firm_sim.policies import make_policy
from firm_sim.registry import make_task
from firm_sim.rollout import rollout


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", default="no_op")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scene-only", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    args = parser.parse_args()

    env = BenchmarkEnv(BenchmarkEnvConfig(gui=not args.headless))
    task = make_task(args.task)
    policy = make_policy(args.policy)

    try:
        if args.scene_only:
            env.reset(task)
            steps = max(1, int(args.hold_seconds / env.config.time_step))
            for _ in range(steps):
                p.stepSimulation()
                time.sleep(env.config.time_step)
            print(
                f"Displayed task '{task.name}' scene for {args.hold_seconds:.1f} second(s)."
            )
            return 0

        result = rollout(policy, env, task, {"max_steps": task.max_steps})
    except PlaceholderTaskError as exc:
        print(f"Placeholder task: {exc}")
        return 0
    finally:
        env.close()

    print(f"Ran task '{task.name}' with policy '{policy.name}'.")
    print(f"Steps: {result['steps']}, total_reward: {result['total_reward']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
