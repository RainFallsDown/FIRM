"""Task registry for FIRM task families."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Type

from firm_sim.tasks.base import Task


TASK_REGISTRY: Dict[str, tuple[str, str]] = {
    "instruction_manual_insertion": (
        "firm_sim.tasks.manual_insertion",
        "InstructionManualInsertionTask",
    ),
    "cable_manipulation": (
        "firm_sim.tasks.cable_manipulation",
        "CableManipulationTask",
    ),
    "box_folding": (
        "firm_sim.tasks.box_folding",
        "BoxFoldingTask",
    ),
    "sponge_pad_placement": (
        "firm_sim.tasks.sponge_pad_placement",
        "SpongePadPlacementTask",
    ),
    "tape_manipulation": (
        "firm_sim.tasks.tape_manipulation",
        "TapeManipulationTask",
    ),
}


def task_names() -> list[str]:
    return sorted(TASK_REGISTRY.keys())


def make_task(name: str) -> Task:
    try:
        module_name, class_name = TASK_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(task_names())
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from exc

    module = import_module(module_name)
    task_cls: Type[Task] = getattr(module, class_name)
    return task_cls()
