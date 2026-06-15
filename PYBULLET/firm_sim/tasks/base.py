"""Base task interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from firm_sim.exceptions import PlaceholderTaskError


@dataclass
class Task:
    name: str
    description: str
    physical_challenge: str
    object_class: str
    metric_slots: List[str] = field(
        default_factory=lambda: [
            "binary_success",
            "completion_quality",
            "deformation_quality",
            "robustness",
        ]
    )
    placeholder: bool = True
    max_steps: int = 1

    def reset(self, env: Any) -> None:
        if self.placeholder:
            raise PlaceholderTaskError(
                f"Task '{self.name}' is registered, but its scene is not implemented yet."
            )

    def apply_action(self, env: Any, action: Optional[Dict[str, Any]]) -> None:
        del env, action

    def reward(self, env: Any) -> tuple[float, Dict[str, Any]]:
        del env
        return 0.0, self.metric_state()

    def done(self, env: Any) -> bool:
        del env
        return True

    def post_step(
        self,
        env: Any,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        del env, observation, reward, done, info

    def oracle(self) -> None:
        raise NotImplementedError(f"Task '{self.name}' does not define an oracle yet.")

    def metric_state(self) -> Dict[str, Optional[float]]:
        return {key: None for key in self.metric_slots}
