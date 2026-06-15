"""Shared PyBullet environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pybullet as p
import pybullet_data


@dataclass
class BenchmarkEnvConfig:
    gui: bool = False
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    workspace_bounds: Dict[str, list[float]] = field(
        default_factory=lambda: {"x": [0.2, 0.8], "y": [-0.4, 0.4], "z": [0.0, 0.5]}
    )
    solver_iterations: int = 150


class BenchmarkEnv:
    """Minimal benchmark environment with stable lifecycle hooks."""

    def __init__(self, config: Optional[BenchmarkEnvConfig] = None):
        self.config = config or BenchmarkEnvConfig()
        self.client_id: Optional[int] = None
        self.task = None
        self.episode_step = 0
        self.connected = False

    def connect(self) -> int:
        if self.connected and self.client_id is not None:
            return self.client_id

        mode = p.GUI if self.config.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.time_step)
        p.setGravity(*self.config.gravity)
        p.setPhysicsEngineParameter(numSolverIterations=self.config.solver_iterations)
        self.connected = True
        return self.client_id

    def close(self) -> None:
        if self.connected:
            p.disconnect(self.client_id)
            self.connected = False
            self.client_id = None

    def reset(self, task: Any) -> Dict[str, Any]:
        self.connect()
        p.resetSimulation()
        p.setGravity(*self.config.gravity)
        p.setTimeStep(self.config.time_step)
        p.setPhysicsEngineParameter(numSolverIterations=self.config.solver_iterations)
        p.loadURDF("plane.urdf")
        self.task = task
        self.episode_step = 0
        task.reset(self)
        return self.observe()

    def step(self, action: Optional[Dict[str, Any]] = None):
        if self.task is None:
            raise RuntimeError("Environment step() called before reset().")

        self.episode_step += 1
        self.task.apply_action(self, action)
        p.stepSimulation()
        reward, reward_info = self.task.reward(self)
        done = self.task.done(self)
        info = self.info()
        info["task_info"] = reward_info
        info["task_name"] = self.task.name
        observation = self.observe()
        self.task.post_step(self, observation, reward, done, info)
        return observation, reward, done, info

    def observe(self) -> Dict[str, Any]:
        task_name = self.task.name if self.task is not None else None
        return {
            "task": task_name,
            "episode_step": self.episode_step,
            "workspace_bounds": self.config.workspace_bounds,
        }

    def render(self) -> Dict[str, Any]:
        return {"mode": "gui" if self.config.gui else "headless"}

    def info(self) -> Dict[str, Any]:
        return {
            "episode_step": self.episode_step,
            "physics": {
                "time_step": self.config.time_step,
                "gravity": self.config.gravity,
                "solver_iterations": self.config.solver_iterations,
            },
        }
