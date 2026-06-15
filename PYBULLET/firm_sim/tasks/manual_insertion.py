"""Instruction manual insertion scene scaffold."""

from __future__ import annotations

from typing import Any, Dict

import pybullet as p

from firm_sim.tasks.base import Task


class InstructionManualInsertionTask(Task):
    def __init__(self):
        super().__init__(
            name="instruction_manual_insertion",
            description="Insert a planar instruction manual into a confined packaging structure.",
            physical_challenge="Planar sheet deformation and confined insertion",
            object_class="2D planar sheet",
            placeholder=False,
            max_steps=120,
        )
        self.asset_ids: Dict[str, int] = {}
        self.target_center = (0.61, -0.03, 0.664)
        self.target_half_extents = (0.06, 0.04, 0.001)

    def reset(self, env: Any) -> None:
        self.asset_ids = {}
        self._reset_camera()
        self.asset_ids["table"] = p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], useFixedBase=True)
        self._create_rigid_box_receptacle()
        self.asset_ids["target_marker"] = self._create_target_marker()
        self.asset_ids["manual"] = self._create_manual_proxy()

    def reward(self, env: Any) -> tuple[float, Dict[str, Any]]:
        del env
        return 0.0, {
            "binary_success": None,
            "completion_quality": None,
            "deformation_quality": None,
            "robustness": None,
            "scene_status": "manual_insertion_setup_only",
        }

    def done(self, env: Any) -> bool:
        return env.episode_step >= self.max_steps

    def _reset_camera(self) -> None:
        try:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=45.0,
                cameraPitch=-32.0,
                cameraTargetPosition=[0.55, -0.05, 0.62],
            )
        except p.error:
            pass

    def _create_target_marker(self) -> int:
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.target_half_extents
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self.target_half_extents,
            rgbaColor=[0.2, 0.8, 0.25, 0.45],
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_center,
        )

    def _create_rigid_box_receptacle(self) -> None:
        center_x, center_y, center_z = 0.61, -0.03, 0.655
        inner_length = 0.18
        inner_width = 0.12
        wall_thickness = 0.01
        wall_height = 0.075
        floor_thickness = 0.008
        rgba = [0.78, 0.62, 0.45, 1.0]

        parts = {
            "receptacle_floor": (
                [inner_length / 2.0, inner_width / 2.0, floor_thickness / 2.0],
                [center_x, center_y, center_z + floor_thickness / 2.0],
            ),
            "receptacle_wall_left": (
                [wall_thickness / 2.0, inner_width / 2.0 + wall_thickness, wall_height / 2.0],
                [center_x - inner_length / 2.0 - wall_thickness / 2.0, center_y, center_z + wall_height / 2.0],
            ),
            "receptacle_wall_right": (
                [wall_thickness / 2.0, inner_width / 2.0 + wall_thickness, wall_height / 2.0],
                [center_x + inner_length / 2.0 + wall_thickness / 2.0, center_y, center_z + wall_height / 2.0],
            ),
            "receptacle_wall_back": (
                [inner_length / 2.0 + wall_thickness, wall_thickness / 2.0, wall_height / 2.0],
                [center_x, center_y - inner_width / 2.0 - wall_thickness / 2.0, center_z + wall_height / 2.0],
            ),
            "receptacle_wall_front": (
                [inner_length / 2.0 + wall_thickness, wall_thickness / 2.0, wall_height / 2.0],
                [center_x, center_y + inner_width / 2.0 + wall_thickness / 2.0, center_z + wall_height / 2.0],
            ),
        }

        for name, (half_extents, position) in parts.items():
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
            )
            self.asset_ids[name] = body_id

    def _create_manual_proxy(self) -> int:
        manual_half_extents = [0.09, 0.06, 0.004]
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=manual_half_extents)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=manual_half_extents,
            rgbaColor=[0.95, 0.95, 0.92, 1.0],
        )
        start_position = [0.34, 0.16, 0.645]
        start_orientation = p.getQuaternionFromEuler((0.0, 0.0, -0.4))
        body_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_position,
            baseOrientation=start_orientation,
        )
        p.changeDynamics(body_id, -1, lateralFriction=1.0, rollingFriction=0.001)
        return body_id
