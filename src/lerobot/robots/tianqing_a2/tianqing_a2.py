#!/usr/bin/env python

# Copyright 2025 Project contributors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
from functools import cached_property
from typing import Any

import zmq

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_tianqing_a2 import TianqingA2Config

logger = logging.getLogger(__name__)


class TianqingA2(Robot):
    """
    Tianqing A2 dual-arm robot implementation for the LeRobot framework.

    Communicates with the robot controller over ZMQ PUB/SUB sockets using JSON messages.
    - State subscription: robot publishes joint positions on ``state_port``; we subscribe.
    - Action publishing: we publish target joint positions on ``action_port``; robot subscribes.

    Expected state JSON (published by robot)::

        {
            "joint51": 0.0, ..., "joint67": 0.0,
            "left_joint1": 0.0, "right_joint1": 0.0
        }

    Expected action JSON (consumed by robot)::

        {
            "joint51": 0.0, ..., "joint67": 0.0,
            "left_joint1": 0.0, "right_joint1": 0.0
        }
    """

    config_class = TianqingA2Config
    name = "tianqing_a2"

    def __init__(self, config: TianqingA2Config):
        super().__init__(config)
        self.config: TianqingA2Config = config
        self.cameras = make_cameras_from_configs(config.cameras)

        self._zmq_context: zmq.Context | None = None
        # SUB socket: receives robot state published by the robot controller
        self._state_sub: zmq.Socket | None = None
        # PUB socket: sends action commands to the robot controller
        self._action_pub: zmq.Socket | None = None

    # ------------------------------------------------------------------
    # Feature descriptors (independent of connection state)
    # ------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {name: float for name in self.config.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._zmq_context is not None and not self._zmq_context.closed

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open ZMQ sockets and connect to the robot controller."""
        self._zmq_context = zmq.Context()

        # Subscribe to robot state stream
        self._state_sub = self._zmq_context.socket(zmq.SUB)
        # CONFLATE: only keep the latest message (no stale state accumulation)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)
        self._state_sub.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self._state_sub.connect(f"tcp://{self.config.robot_ip}:{self.config.state_port}")
        self._state_sub.setsockopt_string(zmq.SUBSCRIBE, self.config.state_topic)

        # Publish action commands to robot
        self._action_pub = self._zmq_context.socket(zmq.PUB)
        self._action_pub.connect(f"tcp://{self.config.robot_ip}:{self.config.action_port}")

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected (state=tcp://{self.config.robot_ip}:{self.config.state_port}, "
                    f"action=tcp://{self.config.robot_ip}:{self.config.action_port}).")

    # ------------------------------------------------------------------
    # Calibration (not applicable for a network robot)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """No calibration required for network-connected robot."""
        logger.info(f"{self}: calibration not required, skipping.")

    def configure(self) -> None:
        """No additional configuration required after connection."""

    # ------------------------------------------------------------------
    # Observation & action
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Receive the latest robot state via ZMQ and return as RobotObservation.

        The SUB socket uses CONFLATE so recv() always returns the freshest message.

        Raises:
            TimeoutError: if no state message arrives within ``recv_timeout_ms``.
        """
        start = time.perf_counter()
        obs_dict: dict[str, Any] = {}

        try:
            raw = self._state_sub.recv_string()
        except zmq.Again:
            raise TimeoutError(
                f"{self}: timed out waiting for state from "
                f"tcp://{self.config.robot_ip}:{self.config.state_port} "
                f"(timeout={self.config.recv_timeout_ms}ms)"
            )

        # Strip the topic prefix and parse JSON payload
        _, _, payload = raw.partition(" ")
        state: dict[str, float] = json.loads(payload)

        for name in self.config.motor_names:
            obs_dict[name] = float(state.get(name, 0.0))
            
        # print (f"Received state: {obs_dict}")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture camera images
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Publish a joint position command to the robot controller via ZMQ.

        If ``max_relative_target`` is set, the command is clamped relative to the
        current observed position for safety (requires an extra state read).

        Args:
            action: Target joint positions keyed by motor name.

        Returns:
            The action actually sent (potentially clamped).
        """
        goal_pos: dict[str, float] = {
            name: float(action[name]) for name in self.config.motor_names if name in action
        }

        if self.config.max_relative_target is not None:
            obs = self.get_observation()
            present_pos = {name: obs[name] for name in goal_pos}
            goal_present_pos = {name: (goal_pos[name], present_pos[name]) for name in goal_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        payload = json.dumps(goal_pos)
        self._action_pub.send_string(f"{self.config.action_topic} {payload}")

        logger.debug(f"{self} sent action: {goal_pos}")
        return goal_pos

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    @check_if_not_connected
    def disconnect(self) -> None:
        """Close ZMQ sockets and disconnect cameras."""
        if self._state_sub is not None:
            self._state_sub.close()
            self._state_sub = None

        if self._action_pub is not None:
            self._action_pub.close()
            self._action_pub = None

        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")