"""Sponge pad placement task placeholder."""

from firm_sim.tasks.base import Task


class SpongePadPlacementTask(Task):
    def __init__(self):
        super().__init__(
            name="sponge_pad_placement",
            description="Place a compressible sponge pad while managing recoil and shape recovery.",
            physical_challenge="Volumetric compression and elastic recoil",
            object_class="3D volumetric deformable",
        )
