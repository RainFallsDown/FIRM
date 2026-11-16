"""Execute one rollout in a fresh process and record its resolved initial state."""

import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from firm_sim.perturbations import Condition


def load_factory(name):
    module, attribute = name.split(":", 1)
    return getattr(importlib.import_module(module), attribute)


def execute(condition, config):
    if version("genesis-world") != "1.2.3":
        raise RuntimeError("Evaluation requires genesis-world==1.2.3")
    from firm_sim.evaluation.environment import GenesisEnvironment

    env = GenesisEnvironment(condition, config["camera"], config.get("backend", "cpu"))
    initial = env.state()
    policy = load_factory(config["policy"])(config.get("policy_options", {}))
    judge = load_factory(config["judge"])(config.get("judge_options", {}))
    policy.reset(env.observation())
    judge.reset(env)
    success = False
    for step in range(config["max_steps"]):
        env.step(policy.act(env.observation()))
        success, terminated = judge.evaluate(env)
        if not isinstance(success, (bool, np.bool_)) or not isinstance(
            terminated, (bool, np.bool_)
        ):
            raise TypeError("Judge must return (success: bool, terminated: bool)")
        if success or terminated:
            break
    return {
        "condition_id": condition.id,
        "status": "completed",
        "success": bool(success),
        "steps": step + 1,
        "initial_state": initial,
        "final_state": env.state(),
        "genesis_version": version("genesis-world"),
    }


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    condition = Condition(**request["condition"])
    try:
        result = execute(condition, request["config"])
    except Exception:
        result = {
            "condition_id": condition.id,
            "status": "error",
            "error": traceback.format_exc(),
        }
    destination = Path(sys.argv[2])
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(destination)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
