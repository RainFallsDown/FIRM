"""Base policy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Policy:
    name: str
    seed: int = 0

    def reset(self, seed: int) -> None:
        self.seed = seed

    def act(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        del observation, info
        return {}
