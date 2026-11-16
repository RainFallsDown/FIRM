"""Reference policy and configurable geometric success criteria for proxy scenes."""

import numpy as np
from scipy.spatial.transform import Rotation
from firm_sim.evaluation.environment import array, center


class HoldPosition:
    """Stationary control baseline for checking the evaluation installation."""

    def __init__(self, options):
        pass

    def reset(self, observation):
        self.target = observation["joint_position"].copy()

    def act(self, observation):
        return self.target


class GeometricJudge:
    """Object-center bounds in the target frame, optional hinge bounds, stable hold."""

    def __init__(self, options):
        self.lower = np.asarray(options["lower_m"], dtype=float)
        self.upper = np.asarray(options["upper_m"], dtype=float)
        self.hold_steps = options["hold_steps"]
        self.hinge = options.get("hinge")
        if (
            self.lower.shape != (3,)
            or self.upper.shape != (3,)
            or not np.all(self.lower < self.upper)
        ):
            raise ValueError("Expected strictly ordered finite 3D target bounds")
        if (
            not np.all(np.isfinite([self.lower, self.upper]))
            or type(self.hold_steps) is not int
            or self.hold_steps < 1
        ):
            raise ValueError("Invalid bounds or hold_steps")

    def reset(self, env):
        self.count = 0

    def evaluate(self, env):
        obj = env.entities[env.object_name]
        target = env.entities["target_zone"]
        quat = array(target.get_quat(relative=False)).reshape(4)
        relative = (
            Rotation.from_quat(quat[[1, 2, 3, 0]])
            .inv()
            .apply(center(obj) - array(target.get_pos(relative=False)).reshape(3))
        )
        inside = bool(np.all(relative >= self.lower) and np.all(relative <= self.upper))
        if self.hinge:
            position = array(
                env.entities[self.hinge["entity"]].get_dofs_position()
            ).reshape(-1)
            angle = float(position[self.hinge["dof"]])
            inside = (
                inside and self.hinge["lower_rad"] <= angle <= self.hinge["upper_rad"]
            )
        self.count = self.count + 1 if inside else 0
        return self.count >= self.hold_steps, False
