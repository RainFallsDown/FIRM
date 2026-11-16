"""Genesis scene adapter with state-based judging and noisy policy observations."""

import numpy as np
from scipy.spatial.transform import Rotation
from firm_sim.perturbations import noisy_observation


def array(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def center(entity):
    if hasattr(entity, "get_particles_pos"):
        return array(entity.get_particles_pos()).reshape(-1, 3).mean(axis=0)
    return array(entity.get_pos(relative=False)).reshape(3)


def transform_entity(entity, xy, yaw, pivot):
    rotation = Rotation.from_euler("z", yaw, degrees=True)
    if hasattr(entity, "get_particles_pos"):
        points = array(entity.get_particles_pos()).reshape(-1, 3)
        entity.set_particles_pos(
            pivot + rotation.apply(points - pivot) + np.array([*xy, 0.0])
        )
        entity.set_particles_vel(np.zeros_like(points))
        return
    position = array(entity.get_pos(relative=False)).reshape(3)
    quat = array(entity.get_quat(relative=False)).reshape(4)  # Genesis wxyz
    position = pivot + rotation.apply(position - pivot) + np.array([*xy, 0.0])
    rotated = (rotation * Rotation.from_quat(quat[[1, 2, 3, 0]])).as_quat()
    entity.set_pos(position, zero_velocity=True, relative=False)
    entity.set_quat(rotated[[3, 0, 1, 2]], zero_velocity=True, relative=False)


class GenesisEnvironment:
    def __init__(self, condition, camera, backend="cpu", show_viewer=False):
        import genesis as gs
        from firm_sim.runtime import init_genesis
        from firm_sim.tasks import build_task_scene, get_task_spec

        init_genesis(backend=getattr(gs, backend), seed=condition.seed)
        self.condition = condition
        self.frame = 0
        self.scene, self.entities = build_task_scene(
            condition.task,
            show_viewer=show_viewer,
            camera_specs=camera,
            condition=condition,
        )
        self.robot = self.entities["tianqing_robot"]
        self.camera = self.entities["snapshot_camera"]
        self.object_name = get_task_spec(condition.task).object_entity
        obj = self.entities[self.object_name]
        pivot = array(center(obj)).reshape(3)
        transform_entity(obj, condition.object_xy, condition.object_yaw_deg, pivot)
        target = self.entities["target_zone"]
        pivot = array(target.get_pos(relative=False)).reshape(3)
        fixtures = ["target_zone", "box_fold_support"]
        if condition.task != "box_folding":
            fixtures.append("box_folding_proxy")
        for name in fixtures:
            transform_entity(
                self.entities[name],
                condition.fixture_xy,
                condition.fixture_yaw_deg,
                pivot,
            )
        # Camera-to-world transform; camera pose noise is fixed throughout a rollout.
        matrix = array(self.camera.transform).copy()
        rotation = Rotation.from_rotvec(
            np.deg2rad(condition.camera_rotvec_deg)
        ).as_matrix()
        matrix[:3, :3] = rotation @ matrix[:3, :3]
        matrix[:3, 3] += condition.camera_xyz
        self.camera.set_pose(transform=matrix)

    def observation(self):
        rgb, depth, *_ = self.camera.render(rgb=True, depth=True)
        rgb, depth = noisy_observation(
            array(rgb), array(depth), self.condition, self.frame
        )
        return {
            "rgb": rgb,
            "depth_m": depth,
            "joint_position": array(self.robot.get_dofs_position()).copy(),
        }

    def step(self, action):
        if action is not None:
            action = np.asarray(action, dtype=float)
            if action.shape != (self.robot.n_dofs,) or not np.all(np.isfinite(action)):
                raise ValueError(
                    "Action must contain one finite position target per robot DOF"
                )
            self.robot.control_dofs_position(action)
        self.scene.step()
        self.frame += 1

    def state(self):
        result = {}
        for name, entity in self.entities.items():
            if name == "snapshot_camera":
                continue
            if hasattr(entity, "get_particles_pos"):
                result[name] = {
                    "particles_m": array(entity.get_particles_pos()).tolist(),
                    "velocity": array(entity.get_particles_vel()).tolist(),
                }
            elif hasattr(entity, "get_quat"):
                result[name] = {
                    "position": array(entity.get_pos(relative=False)).tolist(),
                    "quaternion_wxyz": array(entity.get_quat(relative=False)).tolist(),
                }
                if hasattr(entity, "get_dofs_position"):
                    result[name]["dofs"] = array(entity.get_dofs_position()).tolist()
                    result[name]["dof_velocity"] = array(
                        entity.get_dofs_velocity()
                    ).tolist()
        result["camera"] = {"transform": array(self.camera.transform).tolist()}
        return result
