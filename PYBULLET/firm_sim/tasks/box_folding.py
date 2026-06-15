"""Box folding task placeholder."""

from firm_sim.tasks.base import Task


class BoxFoldingTask(Task):
    def __init__(self):
        super().__init__(
            name="box_folding",
            description="Fold articulated cardboard components into a target box configuration.",
            physical_challenge="Articulated structure and hinge-like constraints",
            object_class="Articulated semi-rigid structure",
        )
