"""Registry for Genesis-based FIRM task scenes."""

from __future__ import annotations

from firm_sim.perturbations import PerturbationSample
from firm_sim import scenes as scene_builders
from firm_sim.tasks.base import TaskSceneSpec


TASK_SCENE_REGISTRY: dict[str, tuple[TaskSceneSpec, str]] = {
    "instruction_manual": (
        TaskSceneSpec(
            name="instruction_manual",
            scene_name="instruction_manual",
            object_class="five-layer bound booklet proxy",
            description="Insert a five-sheet bound instruction manual into the box fixture.",
            notes="Uses five full-size paper layers in one booklet body and a fixed outward-open box lid.",
        ),
        "build_instruction_manual_scene",
    ),
    "sponge_pad": (
        TaskSceneSpec(
            name="sponge_pad",
            scene_name="sponge_pad",
            object_class="thin PBD deformable pad",
            description="Place a sponge pad into the same box fixture.",
            notes="Uses the shared table workspace with the articulated cardboard box base and an outward-open lid.",
        ),
        "build_sponge_pad_scene",
    ),
    "cable_manipulation": (
        TaskSceneSpec(
            name="cable_manipulation",
            scene_name="cable_manipulation",
            object_class="bundled cable + rigid mouse proxy",
            description="Manipulate a rigid mouse body attached to a bundled cable proxy on the shared table workspace.",
            notes="Current Genesis scene uses a continuous bundled cable mesh repositioned to the central spawn zone on the shared articulated-box workspace.",
        ),
        "build_cable_scene",
    ),
    "box_folding": (
        TaskSceneSpec(
            name="box_folding",
            scene_name="box_folding",
            object_class="cardboard box proxy",
            description="Manipulate a cardboard box proxy toward a target folding region on the shared table workspace.",
            notes="Current Genesis scene uses a centered cardboard box proxy with a single articulated lid hinge and an extended graspable lid edge.",
        ),
        "build_box_folding_scene",
    ),
    "tape_manipulation": (
        TaskSceneSpec(
            name="tape_manipulation",
            scene_name="tape_manipulation",
            object_class="rigid annulus proxy",
            description="Manipulate a tape roll on the shared table workspace.",
            notes="First Genesis proxy uses the documented tape diameter and width in the same central spawn zone as the sponge scene.",
        ),
        "build_tape_scene",
    ),
}


def task_names() -> list[str]:
    return sorted(TASK_SCENE_REGISTRY.keys())


def get_task_spec(name: str) -> TaskSceneSpec:
    try:
        return TASK_SCENE_REGISTRY[name][0]
    except KeyError as exc:
        available = ", ".join(task_names())
        raise KeyError(f"Unknown task scene '{name}'. Available task scenes: {available}") from exc


def build_task_scene(
    name: str,
    show_viewer: bool = True,
    camera_specs: dict[str, object] | None = None,
    perturbation: PerturbationSample | None = None,
):
    try:
        _, builder_name = TASK_SCENE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(task_names())
        raise KeyError(f"Unknown task scene '{name}'. Available task scenes: {available}") from exc

    builder = getattr(scene_builders, builder_name)
    return builder(
        show_viewer=show_viewer,
        camera_specs=camera_specs,
        perturbation=perturbation,
    )
