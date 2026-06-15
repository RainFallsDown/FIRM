#!/usr/bin/env python3
"""Run the shared evaluation scaffold."""

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from firm_sim.env import BenchmarkEnvConfig
from firm_sim.evaluation import Evaluator
from firm_sim.exceptions import PlaceholderTaskError


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", default="no_op")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", default="outputs/eval")
    args = parser.parse_args()

    evaluator = Evaluator(
        output_dir=ROOT / args.output_dir,
        env_config=BenchmarkEnvConfig(gui=not args.headless),
    )

    try:
        result = evaluator.evaluate(
            task_name=args.task,
            policy_name=args.policy,
            episodes=args.episodes,
            seed=args.seed,
        )
    except PlaceholderTaskError as exc:
        print(f"Placeholder task: {exc}")
        print(f"Created evaluation scaffold in: {ROOT / args.output_dir}")
        return 0

    print(
        f"Evaluation finished for task '{result['task_name']}' "
        f"with policy '{result['policy_name']}'."
    )
    print(f"Collected {len(result['results'])} episode result(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
