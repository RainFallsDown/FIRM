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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tianqing_a2")
@dataclass
class TianqingA2Config(RobotConfig):
    # --------------------
    # Network config (ZMQ)
    # --------------------
    # IP of the robot controller (or bridge server)
    robot_ip: str = "127.0.0.1"
    # Port where the robot publishes state (we subscribe)
    state_port: int = 5555
    # Port where the robot subscribes for actions (we publish)
    action_port: int = 6666
    # ZMQ topic name for robot state messages
    state_topic: str = "tianqing_state"
    # ZMQ topic name for robot action messages
    action_topic: str = "tianqing_action"
    # Timeout in ms when waiting for a state message
    recv_timeout_ms: int = 500

    # --------------------
    # Robot joint config
    # --------------------
    # Safety clamp: max allowed change in joint position per action step
    max_relative_target: float | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Motor names in order (left arm: joint_51~57, right arm: joint_61~67, grippers)
    motor_names: list[str] = field(default_factory=lambda: [
        "joint51",
        "joint52",
        "joint53",
        "joint54",
        "joint55",
        "joint56",
        "joint57",
        "joint61",
        "joint62",
        "joint63",
        "joint64",
        "joint65",
        "joint66",
        "joint67",
        "left_joint1",
        "right_joint1",
    ])
