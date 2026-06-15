"""Metadata definitions for Genesis FIRM task scenes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSceneSpec:
    """High-level description of one FIRM task scene."""

    name: str
    scene_name: str
    object_class: str
    description: str
    notes: str
