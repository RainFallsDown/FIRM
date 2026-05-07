#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tianqing A2 ROS2 <-> ZMQ bridge for LeRobot 0.5.0.

This replaces the mock tianqing_a2_server.py with a real robot bridge:
- ROS2 joint/hand state -> ZMQ state stream for LeRobot client
- ZMQ action commands from LeRobot -> ROS2 arm/hand commands
- 14-dof arm command quintic upsampling (30Hz action -> 100Hz publish)
- Optional debug publishers for raw model output

Assumes the same robot-side custom messages used in the previous client:
  bot_custom_interface.msg.AGIRosJointState
  bot_custom_interface.msg.AGIRosJointStateV2
  bot_custom_interface.msg.AGIRosJointCommand
  bot_custom_interface.msg.AGIRosJointCommandV2
"""

import argparse
import collections
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from bot_custom_interface.msg import (
    AGIRosJointCommand,
    AGIRosJointCommandV2,
    AGIRosJointState,
    AGIRosJointStateV2,
)


# ---------------------------------------------------------------------
# ZMQ-facing joint names (must match TianqingA2Config / TianqingA2)
# ---------------------------------------------------------------------
ZMQ_MOTOR_NAMES = [
    "joint51", "joint52", "joint53", "joint54", "joint55", "joint56", "joint57",
    "joint61", "joint62", "joint63", "joint64", "joint65", "joint66", "joint67",
    "left_joint1", "right_joint1",
]

ARM_ZMQ_NAMES = ZMQ_MOTOR_NAMES[:14]
HAND_ZMQ_NAMES = ZMQ_MOTOR_NAMES[14:]

# ---------------------------------------------------------------------
# ROS-side arm command names and state indices from your old client
# ---------------------------------------------------------------------
ROS_ARM_JOINT_NAMES = [
    "joint51", "joint52", "joint53", "joint54", "joint55", "joint56", "joint57",
    "joint61", "joint62", "joint63", "joint64", "joint65", "joint66", "joint67",
]

# From old O2_Joint enum in lerobot_zmq_client.py
ROS_ARM_STATE_INDICES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

STATE_PORT = 5555
ACTION_PORT = 6666
STATE_TOPIC = "tianqing_state"
ACTION_TOPIC = "tianqing_action"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tianqing_a2_server")


class Kalman1D:
    def __init__(self, process_var: float = 5e-4, measurement_var: float = 1e-2) -> None:
        self.Q = process_var
        self.R = measurement_var
        self.x = 0.0
        self.P = 1.0

    def update(self, z: float) -> float:
        self.P += self.Q
        k = self.P / (self.P + self.R)
        self.x += k * (z - self.x)
        self.P *= 1.0 - k
        return self.x


class SafeDeque:
    def __init__(self, maxlen: int = 2) -> None:
        self.q = collections.deque(maxlen=maxlen)
        self.ts = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, item: np.ndarray, timestamp: float) -> None:
        with self.lock:
            self.q.append(np.asarray(item, dtype=np.float64).copy())
            self.ts.append(float(timestamp))

    def get(self):
        with self.lock:
            if len(self.q) < 2:
                return None
            return self.q[0].copy(), self.q[1].copy(), self.ts[0], self.ts[1]

    def clear(self) -> None:
        with self.lock:
            self.q.clear()
            self.ts.clear()

    def __len__(self) -> int:
        with self.lock:
            return len(self.q)


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.data

    def set(self, data):
        with self.lock:
            self.data = data


class TianqingA2Bridge(Node):
    def __init__(self, args):
        super().__init__("tianqing_a2_ros2_zmq_bridge")
        self.args = args

        # ---------------- ROS2 state cache ----------------
        self.low_state_buf = DataBuffer()      # AGIRosJointState
        self.hand_state_buf = DataBuffer()     # AGIRosJointStateV2
        self.latest_state_stamp_s = 0.0

        # Current robot state in model / ZMQ order
        self.current_arm_q = np.zeros(14, dtype=np.float64)
        self.current_hand_norm = np.zeros(2, dtype=np.float32)   # [0,1], same semantics as old client

        # Latest target action from LeRobot in ZMQ order
        self.latest_model_action = np.zeros(16, dtype=np.float32)
        self.latest_arm_target = np.zeros(14, dtype=np.float64)
        self.latest_hand_target = np.zeros(2, dtype=np.float32)
        self.latest_action_stamp_s = 0.0
        self._action_lock = threading.Lock()
        self._action_stream_active = False
        self._rosbag_lock = threading.Lock()
        self._rosbag_process: subprocess.Popen | None = None

        # Upsample state
        self.upsample_buffer = SafeDeque(maxlen=2)
        self.arm_velocity_limit = float(args.arm_velocity_limit)
        self.arm_delta_clip = float(args.arm_delta_clip)
        self.use_kalman = bool(args.use_kalman)
        self.kalman_filters_arm = [Kalman1D() for _ in range(14)]
        self.debug_joint = bool(args.debug_joint)

        # Hand hysteresis / debounce (same idea as old client)
        self.left_hand_cmd_open = True
        self.right_hand_cmd_open = True
        self.left_hand_cmd_count = 0
        self.right_hand_cmd_count = 0

        # Stats
        self._last_state_warn_t = 0.0
        self._running = True

        # ---------------- QoS ----------------
        qos_sensor = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_sensor.reliability = ReliabilityPolicy.RELIABLE
        qos_sensor.durability = DurabilityPolicy.VOLATILE

        qos_hand = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_hand.reliability = ReliabilityPolicy.RELIABLE
        qos_hand.durability = DurabilityPolicy.VOLATILE

        # ---------------- Subscribers ----------------
        self.sub_joint_state = self.create_subscription(
            AGIRosJointState,
            args.ros_joint_state_topic,
            self.joint_state_callback,
            qos_sensor,
        )
        self.sub_hand_state = self.create_subscription(
            AGIRosJointStateV2,
            args.ros_hand_state_topic,
            self.hand_state_callback,
            qos_hand,
        )

        # ---------------- Publishers ----------------
        self.pub_joint_command = self.create_publisher(
            AGIRosJointCommand,
            args.ros_joint_cmd_topic,
            qos_sensor,
        )
        self.pub_hand_command = self.create_publisher(
            AGIRosJointCommandV2,
            args.ros_hand_cmd_topic,
            10,
        )

        self.pub_joint_command_model = self.create_publisher(
            AGIRosJointCommand,
            args.ros_joint_cmd_model_topic,
            qos_sensor,
        )
        self.pub_model_action = self.create_publisher(
            Float32MultiArray,
            args.ros_model_action_topic,
            qos_sensor,
        )

        # ---------------- Timers ----------------
        self.state_timer = self.create_timer(1.0 / args.state_pub_hz, self.publish_state_to_zmq)
        self.upsample_timer = self.create_timer(1.0 / args.command_pub_hz, self.upsample_callback)

        # ---------------- ZMQ ----------------
        self.zmq_ctx = zmq.Context.instance()

        self.state_pub = self.zmq_ctx.socket(zmq.PUB)
        self.state_pub.setsockopt(zmq.SNDHWM, 1)
        self.state_pub.bind(f"tcp://*:{args.state_port}")

        self.action_sub = self.zmq_ctx.socket(zmq.SUB)
        self.action_sub.setsockopt_string(zmq.SUBSCRIBE, args.action_topic)
        self.action_sub.setsockopt(zmq.RCVHWM, 1)
        self.action_sub.setsockopt(zmq.CONFLATE, 1)
        self.action_sub.bind(f"tcp://*:{args.action_port}")

        self.poller = zmq.Poller()
        self.poller.register(self.action_sub, zmq.POLLIN)

        self._action_thread = threading.Thread(target=self.zmq_action_loop, daemon=True)
        self._action_thread.start()

        logger.info("Tianqing A2 ROS2<->ZMQ bridge started")
        logger.info(f"  state_pub  -> tcp://*:{args.state_port} topic={args.state_topic}")
        logger.info(f"  action_sub <- tcp://*:{args.action_port} topic={args.action_topic}")
        logger.info(f"  ros joint state topic: {args.ros_joint_state_topic}")
        logger.info(f"  ros hand  state topic: {args.ros_hand_state_topic}")
        logger.info(f"  ros joint cmd  topic: {args.ros_joint_cmd_topic}")
        logger.info(f"  ros hand  cmd  topic: {args.ros_hand_cmd_topic}")
        if self.debug_joint:
            logger.warning("  [DEBUG_JOINT] debug_joint=True: agiros_joint_commands will NOT be published (dry-run)")

    # -----------------------------------------------------------------
    # ROS2 state callbacks
    # -----------------------------------------------------------------
    def joint_state_callback(self, msg: AGIRosJointState) -> None:
        if msg is None or len(msg.positions) == 0:
            return
        self.low_state_buf.set(msg)
        self.latest_state_stamp_s = time.time()

        npos = min(len(msg.positions), max(ROS_ARM_STATE_INDICES) + 1)
        arm_q = np.zeros(14, dtype=np.float64)
        for i, idx in enumerate(ROS_ARM_STATE_INDICES):
            if idx < npos:
                arm_q[i] = float(msg.positions[idx])
        self.current_arm_q = arm_q

    def hand_state_callback(self, msg: AGIRosJointStateV2) -> None:
        if msg is None or len(msg.position) == 0:
            return
        self.hand_state_buf.set(msg)
        self.latest_state_stamp_s = time.time()

        raw = np.zeros(2, dtype=np.float32)
        n = min(2, len(msg.position))
        for i in range(n):
            raw[i] = float(msg.position[i])

        # Keep same semantics as old client: state hand = 1 - position / 120
        hand_norm = 1.0 - raw / 120.0
        self.current_hand_norm = np.clip(hand_norm, 0.0, 1.0).astype(np.float32)

    # -----------------------------------------------------------------
    # ZMQ state publication
    # -----------------------------------------------------------------
    def build_state_payload(self) -> dict[str, float]:
        state = {}
        for i, name in enumerate(ARM_ZMQ_NAMES):
            state[name] = float(self.current_arm_q[i])
        state["gripper_left"] = float(self.current_hand_norm[0])
        state["gripper_right"] = float(self.current_hand_norm[1])
        return state

    def publish_state_to_zmq(self) -> None:
        now = time.time()
        if self.latest_state_stamp_s > 0.0:
            age = now - self.latest_state_stamp_s
            if age > self.args.state_stale_warn_s and (now - self._last_state_warn_t) > 1.0:
                self.get_logger().warning(f"ROS state stale: {age:.3f}s")
                self._last_state_warn_t = now

        payload = json.dumps(self.build_state_payload(), ensure_ascii=False)
        self.state_pub.send_string(f"{self.args.state_topic} {payload}")

    # -----------------------------------------------------------------
    # ZMQ action receive
    # -----------------------------------------------------------------
    def zmq_action_loop(self) -> None:
        while self._running and rclpy.ok():
            try:
                events = dict(self.poller.poll(timeout=20))
            except zmq.ZMQError:
                break

            if self.action_sub in events and events[self.action_sub] == zmq.POLLIN:
                try:
                    raw = self.action_sub.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                except Exception as e:
                    self.get_logger().error(f"recv action failed: {e}")
                    continue

                _, _, payload = raw.partition(" ")
                try:
                    action_dict = json.loads(payload)
                except Exception as e:
                    self.get_logger().error(f"invalid action json: {e}")
                    continue

                self.handle_action_dict(action_dict)

    def handle_action_dict(self, action_dict: dict) -> None:
        action = np.zeros(16, dtype=np.float32)
        for i, name in enumerate(ZMQ_MOTOR_NAMES):
            if name in action_dict:
                action[i] = float(action_dict[name])
            else:
                # fall back to latest values to avoid accidental zeroing for partial actions
                if i < 14:
                    action[i] = float(self.current_arm_q[i])
                else:
                    action[i] = float(self.current_hand_norm[i - 14])

        arm_target = action[:14].astype(np.float64)
        hand_target = action[14:16].astype(np.float32)

        if self.use_kalman:
            for i in range(14):
                arm_target[i] = self.kalman_filters_arm[i].update(float(arm_target[i]))

        now_s = self._now_s()

        with self._action_lock:
            self.latest_model_action = action.copy()
            self.latest_arm_target = arm_target.copy()
            self.latest_hand_target = hand_target.copy()
            self.latest_action_stamp_s = now_s

        if not self._action_stream_active:
            self._action_stream_active = True
            self.get_logger().info("Model actions received; command publishing enabled.")
        self.maybe_start_rosbag_recording()

        # seed buffer with current arm state on first command so upsample starts from actual q
        if len(self.upsample_buffer) == 0:
            self.upsample_buffer.append(self.current_arm_q.copy(), now_s - (1.0 / max(self.args.input_action_hz, 1e-6)))
        self.upsample_buffer.append(arm_target, now_s)

        # publish raw model outputs for debugging
        self.publish_joint_command_model(arm_target)
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.pub_model_action.publish(msg)

    # -----------------------------------------------------------------
    # ROS bag recording
    # -----------------------------------------------------------------
    def _build_rosbag_record_command(self) -> list[str]:
        command = ["ros2", "bag", "record", "-s", self.args.rosbag_storage_id]
        topics = list(self.args.rosbag_topics or [])
        if not topics:
            raise ValueError("rosbag_topics is empty; refusing to start rosbag record without explicit topics.")
        command.extend(topics)
        if self.args.rosbag_output_dir:
            output_dir = Path(self.args.rosbag_output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            bag_name = f"{self.args.rosbag_name_prefix}_{timestamp}"
            command.extend(["-o", str(output_dir / bag_name)])
        return command

    def maybe_start_rosbag_recording(self) -> None:
        if not self.args.auto_record_rosbag_on_model_action:
            return

        with self._rosbag_lock:
            if self._rosbag_process is not None:
                if self._rosbag_process.poll() is None:
                    return
                self.get_logger().warning(
                    f"rosbag recorder exited unexpectedly with code {self._rosbag_process.poll()}; restarting."
                )
                self._rosbag_process = None

            try:
                command = self._build_rosbag_record_command()
            except ValueError as exc:
                self.get_logger().error(f"Failed to build rosbag recording command: {exc}")
                return
            try:
                self._rosbag_process = subprocess.Popen(command, start_new_session=True)  # nosec B603
            except FileNotFoundError:
                self.get_logger().error("Failed to start rosbag recording: `ros2` command not found.")
                return
            except Exception as exc:
                self.get_logger().error(f"Failed to start rosbag recording: {exc}")
                return

        self.get_logger().info(f"Started rosbag recording: {' '.join(command)}")

    def stop_rosbag_recording(self) -> None:
        with self._rosbag_lock:
            process = self._rosbag_process
            self._rosbag_process = None

        if process is None:
            return

        if process.poll() is not None:
            return

        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        except Exception as exc:
            self.get_logger().warning(f"Failed to signal rosbag recorder gracefully: {exc}")
            try:
                process.terminate()
            except Exception:
                return

        try:
            process.wait(timeout=self.args.rosbag_shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            self.get_logger().warning("rosbag recorder did not exit after SIGINT; forcing termination.")
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except Exception:
                process.kill()
            self.get_logger().warning("rosbag recorder was force-stopped.")

    # -----------------------------------------------------------------
    # Quintic upsample
    # -----------------------------------------------------------------
    @staticmethod
    def quintic_spline_interpolation(q0, v0, a0, qT, vT, aT, T, t, clip_t: bool = True) -> np.ndarray:
        T = float(T)
        if not np.isfinite(T) or T <= 0.0:
            raise ValueError("T must be a positive finite float.")

        q0 = np.asarray(q0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        a0 = np.asarray(a0, dtype=np.float64)
        qT = np.asarray(qT, dtype=np.float64)
        vT = np.asarray(vT, dtype=np.float64)
        aT = np.asarray(aT, dtype=np.float64)

        t = np.asarray(t, dtype=np.float64)
        if clip_t:
            t = np.clip(t, 0.0, T)

        s = t / T
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s

        h00 = 1 - 10 * s3 + 15 * s4 - 6 * s5
        h10 = s - 6 * s3 + 8 * s4 - 3 * s5
        h20 = 0.5 * (s2 - 3 * s3 + 3 * s4 - s5)
        h01 = 10 * s3 - 15 * s4 + 6 * s5
        h11 = -4 * s3 + 7 * s4 - 3 * s5
        h21 = 0.5 * (s3 - 2 * s4 + s5)

        return (
            q0 * h00
            + v0 * (T * h10)
            + a0 * (T ** 2 * h20)
            + qT * h01
            + vT * (T * h11)
            + aT * (T ** 2 * h21)
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def get_model_action_age_s(self, now_s: Optional[float] = None) -> Optional[float]:
        with self._action_lock:
            latest_action_stamp_s = self.latest_action_stamp_s

        if latest_action_stamp_s <= 0.0:
            return None

        if now_s is None:
            now_s = self._now_s()
        return max(0.0, now_s - latest_action_stamp_s)

    def has_fresh_model_action(self, now_s: Optional[float] = None) -> bool:
        age_s = self.get_model_action_age_s(now_s)
        if age_s is None:
            return False
        return age_s <= self.args.model_action_stale_timeout_s

    def release_action_control(self, age_s: float) -> None:
        if not self._action_stream_active:
            return

        self._action_stream_active = False
        self.stop_rosbag_recording()
        self.upsample_buffer.clear()
        self.kalman_filters_arm = [Kalman1D() for _ in range(14)]
        self.left_hand_cmd_count = 0
        self.right_hand_cmd_count = 0
        self.get_logger().warning(
            "Model action stale for "
            f"{age_s:.3f}s (> {self.args.model_action_stale_timeout_s:.3f}s); "
            "stop publishing robot commands to release control."
        )

    def process_upsampled_data(self, q_prev, q_next, ts_prev: float, ts_next: float) -> Optional[np.ndarray]:
        delta_t = ts_next - ts_prev + 1e-6
        if delta_t <= 0:
            self.get_logger().warning("Invalid delta_t, skip upsample")
            return None

        v_prev = np.clip((q_next - q_prev) / delta_t, -self.arm_velocity_limit, self.arm_velocity_limit)
        v_next = v_prev
        t_now = self.get_clock().now().nanoseconds / 1e9
        t_rel = np.clip(t_now - ts_next, 0.0, delta_t)
        a_prev = np.zeros_like(q_prev)
        a_next = np.zeros_like(q_next)

        return self.quintic_spline_interpolation(
            q_prev, v_prev, a_prev,
            q_next, v_next, a_next,
            delta_t, t_rel,
        )

    def upsample_callback(self) -> None:
        now_s = self._now_s()
        if not self.has_fresh_model_action(now_s):
            age_s = self.get_model_action_age_s(now_s)
            if age_s is not None:
                self.release_action_control(age_s)
            return

        data = self.upsample_buffer.get()
        if data is None:
            return

        q_prev, q_next, ts_prev, ts_next = data
        q_cmd = self.process_upsampled_data(q_prev, q_next, ts_prev, ts_next)
        if q_cmd is None:
            return

        cur_q = self.current_arm_q.copy()
        q_cmd_clipped = np.clip(q_cmd - cur_q, -self.arm_delta_clip, self.arm_delta_clip) + cur_q

        # publish arm joint command (skipped when debug_joint=True)
        if not self.debug_joint:
            self.publish_joint_command(q_cmd_clipped)
        else:
            self.get_logger().debug(f"[DEBUG_JOINT] skipping agiros_joint_commands: {q_cmd_clipped.tolist()}")
        self.publish_hand_command_if_needed()

    # -----------------------------------------------------------------
    # ROS command publication
    # -----------------------------------------------------------------
    def build_arm_command_msg(self, positions: np.ndarray) -> AGIRosJointCommand:
        msg = AGIRosJointCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(ROS_ARM_JOINT_NAMES)
        msg.control_modes = [int(self.args.control_mode)] * len(positions)
        msg.positions = [float(x) for x in positions]
        return msg

    def publish_joint_command(self, positions: np.ndarray) -> None:
        self.pub_joint_command.publish(self.build_arm_command_msg(positions))

    def publish_joint_command_model(self, positions: np.ndarray) -> None:
        self.pub_joint_command_model.publish(self.build_arm_command_msg(positions))

    def _make_hand_msg(self, side: str, position: float) -> AGIRosJointCommandV2:
        msg = AGIRosJointCommandV2()
        msg.header.stamp = self.get_clock().now().to_msg()
        if side == "left":
            msg.name = ["left_joint1"]
        else:
            msg.name = ["right_joint1"]
        msg.control_mode = [1]
        msg.position = [float(position)]
        msg.velocity = [100.0]
        msg.effort = [100.0]
        msg.param1 = [100.0]
        msg.param2 = [100.0]
        msg.param3 = [0.0]
        return msg

    def publish_hand_command_if_needed(self) -> None:
        with self._action_lock:
            hand_target = self.latest_hand_target.copy()

        # Same threshold logic as old client:
        #   >0.9 -> close (120)
        #   <0.4 -> open  (0)
        # left
        if hand_target[0] > self.args.hand_close_threshold and self.left_hand_cmd_open:
            self.left_hand_cmd_count += 1
            if self.left_hand_cmd_count > self.args.hand_debounce_count:
                self.left_hand_cmd_count = 0
                self.left_hand_cmd_open = False
                self.pub_hand_command.publish(self._make_hand_msg("left", 120.0))
        elif hand_target[0] < self.args.hand_open_threshold and not self.left_hand_cmd_open:
            self.left_hand_cmd_open = True
            self.left_hand_cmd_count = 0
            self.pub_hand_command.publish(self._make_hand_msg("left", 0.0))
        else:
            if hand_target[0] <= self.args.hand_close_threshold:
                self.left_hand_cmd_count = 0

        # right
        if hand_target[1] > self.args.hand_close_threshold and self.right_hand_cmd_open:
            self.right_hand_cmd_count += 1
            if self.right_hand_cmd_count > self.args.hand_debounce_count:
                self.right_hand_cmd_count = 0
                self.right_hand_cmd_open = False
                self.pub_hand_command.publish(self._make_hand_msg("right", 120.0))
        elif hand_target[1] < self.args.hand_open_threshold and not self.right_hand_cmd_open:
            self.right_hand_cmd_open = True
            self.right_hand_cmd_count = 0
            self.pub_hand_command.publish(self._make_hand_msg("right", 0.0))
        else:
            if hand_target[1] <= self.args.hand_close_threshold:
                self.right_hand_cmd_count = 0

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------
    def shutdown(self) -> None:
        self._running = False
        self.stop_rosbag_recording()
        try:
            self.action_sub.close(0)
        except Exception:
            pass
        try:
            self.state_pub.close(0)
        except Exception:
            pass
        try:
            if self._action_thread.is_alive():
                self._action_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.zmq_ctx.term()
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tianqing A2 ROS2 <-> ZMQ bridge")

    # ZMQ
    p.add_argument("--state_port", type=int, default=STATE_PORT)
    p.add_argument("--action_port", type=int, default=ACTION_PORT)
    p.add_argument("--state_topic", type=str, default=STATE_TOPIC)
    p.add_argument("--action_topic", type=str, default=ACTION_TOPIC)

    # ROS2 topics
    p.add_argument("--ros_joint_state_topic", type=str, default="/agiros_joint_states")
    p.add_argument("--ros_hand_state_topic", type=str, default="/agiros/hands/states")
    p.add_argument("--ros_joint_cmd_topic", type=str, default="/agiros_joint_commands")
    p.add_argument("--ros_hand_cmd_topic", type=str, default="/agiros/hands/commands")
    p.add_argument("--ros_joint_cmd_model_topic", type=str, default="/agiros_joint_commands_model")
    p.add_argument("--ros_model_action_topic", type=str, default="/lerobot/model_action")

    # Rates / smoothing
    p.add_argument("--state_pub_hz", type=float, default=200.0)
    p.add_argument("--command_pub_hz", type=float, default=100.0)
    p.add_argument("--input_action_hz", type=float, default=30.0)
    p.add_argument("--arm_velocity_limit", type=float, default=1.0)
    p.add_argument("--arm_delta_clip", type=float, default=0.15)
    p.add_argument("--control_mode", type=int, default=7)
    p.add_argument("--use_kalman", action="store_true")

    # Hand thresholds
    p.add_argument("--hand_close_threshold", type=float, default=0.9)
    p.add_argument("--hand_open_threshold", type=float, default=0.4)
    p.add_argument("--hand_debounce_count", type=int, default=10)

    # State health
    p.add_argument("--state_stale_warn_s", type=float, default=1.0)
    p.add_argument(
        "--model_action_stale_timeout_s",
        type=float,
        default=0.2,
        help="Stop publishing robot commands when model actions have been stale longer than this timeout.",
    )

    # rosbag
    p.add_argument(
        "--auto_record_rosbag_on_model_action",
        action="store_true",
        help="Start `ros2 bag record -s <storage> <topics...>` automatically on the first received model action.",
    )
    p.add_argument(
        "--rosbag_storage_id",
        type=str,
        default="mcap",
        help="Storage backend used by ros2 bag record.",
    )
    p.add_argument(
        "--rosbag_topics",
        nargs="+",
        default=[
            "/agiros_joint_states",
            "/agiros_joint_commands",
            "/lerobot/model_action",
            "/agiros_joint_commands_model",
        ],
        help="Explicit ROS topics to record instead of `-a`.",
    )
    p.add_argument(
        "--rosbag_output_dir",
        type=str,
        default="/home/ros/model_action_rosbags",
        help="Optional directory where bag folders are created. Empty means ros2 bag default behavior.",
    )
    p.add_argument(
        "--rosbag_name_prefix",
        type=str,
        default="tianqing_a2",
        help="Bag folder prefix used when --rosbag_output_dir is set.",
    )
    p.add_argument(
        "--rosbag_shutdown_timeout_s",
        type=float,
        default=10.0,
        help="How long to wait for rosbag recorder to flush and exit after SIGINT.",
    )

    # Debug
    p.add_argument(
        "--debug_joint",
        action="store_true",
        help="Dry-run mode: receive and process actions but do NOT publish to agiros_joint_commands.",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()
    rclpy.init()
    node = TianqingA2Bridge(args)

    stop_event = threading.Event()

    def _handle_signal(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
