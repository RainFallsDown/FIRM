"""Shared evaluator for placeholder and future tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from firm_sim.env import BenchmarkEnv, BenchmarkEnvConfig
from firm_sim.policies import make_policy
from firm_sim.registry import make_task
from firm_sim.rollout import rollout


@dataclass
class Evaluator:
    output_dir: Path
    env_config: BenchmarkEnvConfig = field(default_factory=BenchmarkEnvConfig)

    def evaluate(self, task_name: str, policy_name: str, episodes: int, seed: int) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env = BenchmarkEnv(self.env_config)
        task = make_task(task_name)
        policy = make_policy(policy_name, seed=seed)
        results: List[Dict[str, Any]] = []

        try:
            for episode_idx in range(episodes):
                episode_seed = seed + episode_idx
                policy.reset(episode_seed)
                results.append(
                    rollout(
                        policy=policy,
                        env=env,
                        task=task,
                        config={"max_steps": task.max_steps},
                    )
                )
        finally:
            env.close()

        metrics = self._aggregate_metrics(task, results)
        payload = {
            "task_name": task_name,
            "policy_name": policy_name,
            "episodes": episodes,
            "seed": seed,
            "metrics": metrics,
            "results": results,
        }
        return payload

    def _aggregate_metrics(self, task: Any, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "task_family": task.name,
            "placeholder": task.placeholder,
            "binary_success": None,
            "completion_quality": None,
            "deformation_quality": None,
            "robustness": None,
            "num_results": len(results),
        }
