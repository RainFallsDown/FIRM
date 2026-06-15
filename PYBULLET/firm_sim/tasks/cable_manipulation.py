"""Cable manipulation task placeholder."""

from firm_sim.tasks.base import Task


class CableManipulationTask(Task):
    def __init__(self):
        super().__init__(
            name="cable_manipulation",
            description="Manipulate and place a cable under industrial mixed-stiffness constraints.",
            physical_challenge="Bending stiffness and center-of-mass shift",
            object_class="1D bending object",
        )
