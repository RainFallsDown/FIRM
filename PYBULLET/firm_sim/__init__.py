"""FIRM PyBullet benchmark scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from firm_sim.tasks.base import Task


def make_task(name: str) -> "Task":
    from firm_sim.registry import make_task as _make_task

    return _make_task(name)


def task_names() -> list[str]:
    from firm_sim.registry import task_names as _task_names

    return _task_names()


__all__ = ["make_task", "task_names"]
