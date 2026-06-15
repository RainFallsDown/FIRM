"""Tape manipulation task placeholder."""

from firm_sim.tasks.base import Task


class TapeManipulationTask(Task):
    def __init__(self):
        super().__init__(
            name="tape_manipulation",
            description="Manipulate a closed-loop tape object under rolling and slipping constraints.",
            physical_challenge="Closed-loop geometry, rolling, and slipping",
            object_class="1D closed-loop flexible object",
        )
