"""Task-agnostic rollout helpers."""

from __future__ import annotations

from typing import Any, Dict


def rollout(policy: Any, env: Any, task: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    max_steps = int(config.get("max_steps", task.max_steps))
    observation = env.reset(task)
    episode = []
    total_reward = 0.0
    final_info = {"reason": "not_started"}

    for step in range(max_steps):
        action = policy.act(observation, env.info())
        observation, reward, done, info = env.step(action)
        episode.append(
            {
                "step": step,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        total_reward += reward
        final_info = info
        if done:
            break

    return {
        "task_name": task.name,
        "policy_name": policy.name,
        "steps": len(episode),
        "total_reward": total_reward,
        "final_info": final_info,
        "events": episode,
    }
