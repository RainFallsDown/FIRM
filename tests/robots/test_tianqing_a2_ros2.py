import importlib
import signal
import sys
import threading
import types
from types import MethodType, SimpleNamespace

import numpy as np


def _load_tianqing_ros2_module(monkeypatch):
    class DummyNode:
        pass

    class DummyQoSProfile:
        def __init__(self, *args, **kwargs):
            pass

    dummy_rclpy = types.ModuleType("rclpy")
    dummy_rclpy.ok = lambda: True
    dummy_rclpy.init = lambda: None
    dummy_rclpy.shutdown = lambda: None
    dummy_rclpy.spin_once = lambda *args, **kwargs: None

    dummy_rclpy_node = types.ModuleType("rclpy.node")
    dummy_rclpy_node.Node = DummyNode

    dummy_rclpy_qos = types.ModuleType("rclpy.qos")
    dummy_rclpy_qos.DurabilityPolicy = SimpleNamespace(VOLATILE=0)
    dummy_rclpy_qos.HistoryPolicy = SimpleNamespace(KEEP_LAST=0)
    dummy_rclpy_qos.QoSProfile = DummyQoSProfile
    dummy_rclpy_qos.ReliabilityPolicy = SimpleNamespace(RELIABLE=0)

    dummy_std_msgs = types.ModuleType("std_msgs")
    dummy_std_msgs_msg = types.ModuleType("std_msgs.msg")
    dummy_std_msgs_msg.Float32MultiArray = type(
        "Float32MultiArray",
        (),
        {"__init__": lambda self: setattr(self, "data", [])},
    )

    dummy_bot = types.ModuleType("bot_custom_interface")
    dummy_bot_msg = types.ModuleType("bot_custom_interface.msg")
    for name in [
        "AGIRosJointCommand",
        "AGIRosJointCommandV2",
        "AGIRosJointState",
        "AGIRosJointStateV2",
    ]:
        setattr(dummy_bot_msg, name, type(name, (), {}))

    monkeypatch.setitem(sys.modules, "rclpy", dummy_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", dummy_rclpy_node)
    monkeypatch.setitem(sys.modules, "rclpy.qos", dummy_rclpy_qos)
    monkeypatch.setitem(sys.modules, "std_msgs", dummy_std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", dummy_std_msgs_msg)
    monkeypatch.setitem(sys.modules, "bot_custom_interface", dummy_bot)
    monkeypatch.setitem(sys.modules, "bot_custom_interface.msg", dummy_bot_msg)
    sys.modules.pop("lerobot.robots.tianqing_a2.tianqing_a2_ros2", None)

    return importlib.import_module("lerobot.robots.tianqing_a2.tianqing_a2_ros2")


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.debugs = []
        self.errors = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def debug(self, message):
        self.debugs.append(message)

    def error(self, message):
        self.errors.append(message)


class DummyNow:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds


class DummyClock:
    def __init__(self, now_s):
        self._now_s = now_s

    def now(self):
        return DummyNow(int(self._now_s * 1e9))


class DummyBuffer:
    def __init__(self, data=None):
        self.data = data
        self.clear_calls = 0
        self.get_calls = 0

    def clear(self):
        self.clear_calls += 1

    def get(self):
        self.get_calls += 1
        return self.data


class DummyAppendBuffer:
    def __init__(self):
        self.items = []

    def append(self, item, timestamp):
        self.items.append((item.copy(), timestamp))

    def __len__(self):
        return len(self.items)


def _make_bridge_like(module, *, now_s=0.0, latest_action_stamp_s=0.0, timeout_s=0.2, active=True, buffer_data=None):
    logger = DummyLogger()
    bridge = SimpleNamespace(
        args=SimpleNamespace(model_action_stale_timeout_s=timeout_s),
        _action_lock=threading.Lock(),
        latest_action_stamp_s=latest_action_stamp_s,
        _action_stream_active=active,
        upsample_buffer=DummyBuffer(buffer_data),
        left_hand_cmd_count=3,
        right_hand_cmd_count=4,
        current_arm_q=np.zeros(14, dtype=np.float64),
        arm_delta_clip=0.15,
        debug_joint=False,
        _logger=logger,
        _clock=DummyClock(now_s),
        stop_rosbag_calls=0,
    )

    bridge.get_clock = lambda: bridge._clock
    bridge.get_logger = lambda: bridge._logger
    bridge.stop_rosbag_recording = lambda: setattr(bridge, "stop_rosbag_calls", bridge.stop_rosbag_calls + 1)
    bridge._now_s = MethodType(module.TianqingA2Bridge._now_s, bridge)
    bridge.get_model_action_age_s = MethodType(module.TianqingA2Bridge.get_model_action_age_s, bridge)
    bridge.has_fresh_model_action = MethodType(module.TianqingA2Bridge.has_fresh_model_action, bridge)
    bridge.release_action_control = MethodType(module.TianqingA2Bridge.release_action_control, bridge)
    return bridge


def _make_rosbag_bridge(module, *, auto_record=True, output_dir="", process=None):
    logger = DummyLogger()
    bridge = SimpleNamespace(
        args=SimpleNamespace(
            auto_record_rosbag_on_model_action=auto_record,
            rosbag_storage_id="mcap",
            rosbag_output_dir=output_dir,
            rosbag_name_prefix="tianqing_a2",
            rosbag_shutdown_timeout_s=3.0,
        ),
        _rosbag_lock=threading.Lock(),
        _rosbag_process=process,
        _logger=logger,
    )
    bridge.get_logger = lambda: bridge._logger
    bridge._build_rosbag_record_command = MethodType(module.TianqingA2Bridge._build_rosbag_record_command, bridge)
    bridge.maybe_start_rosbag_recording = MethodType(module.TianqingA2Bridge.maybe_start_rosbag_recording, bridge)
    bridge.stop_rosbag_recording = MethodType(module.TianqingA2Bridge.stop_rosbag_recording, bridge)
    return bridge


def test_has_fresh_model_action_respects_timeout(monkeypatch):
    module = _load_tianqing_ros2_module(monkeypatch)
    bridge = _make_bridge_like(module, now_s=10.10, latest_action_stamp_s=10.0, timeout_s=0.2)

    assert bridge.has_fresh_model_action() is True

    bridge._clock = DummyClock(10.25)

    assert bridge.has_fresh_model_action() is False


def test_release_action_control_clears_buffer_and_logs(monkeypatch):
    module = _load_tianqing_ros2_module(monkeypatch)
    bridge = _make_bridge_like(module, now_s=1.0, latest_action_stamp_s=0.7, timeout_s=0.2, active=True)

    bridge.release_action_control(0.3)

    assert bridge._action_stream_active is False
    assert bridge.stop_rosbag_calls == 1
    assert bridge.upsample_buffer.clear_calls == 1
    assert bridge.left_hand_cmd_count == 0
    assert bridge.right_hand_cmd_count == 0
    assert len(bridge._logger.warnings) == 1

    bridge.release_action_control(0.4)

    assert bridge.stop_rosbag_calls == 1
    assert bridge.upsample_buffer.clear_calls == 1
    assert len(bridge._logger.warnings) == 1


def test_upsample_callback_skips_publish_when_action_is_stale(monkeypatch):
    module = _load_tianqing_ros2_module(monkeypatch)
    bridge = _make_bridge_like(module, now_s=5.0, latest_action_stamp_s=4.6, timeout_s=0.2, active=True)
    bridge.publish_joint_command_calls = []
    bridge.publish_hand_command_calls = 0
    bridge.publish_joint_command = lambda positions: bridge.publish_joint_command_calls.append(positions)
    bridge.publish_hand_command_if_needed = lambda: setattr(
        bridge, "publish_hand_command_calls", bridge.publish_hand_command_calls + 1
    )

    module.TianqingA2Bridge.upsample_callback(bridge)

    assert bridge.publish_joint_command_calls == []
    assert bridge.publish_hand_command_calls == 0
    assert bridge.upsample_buffer.get_calls == 0
    assert bridge.upsample_buffer.clear_calls == 1


def test_upsample_callback_publishes_when_action_is_fresh(monkeypatch):
    module = _load_tianqing_ros2_module(monkeypatch)
    q_prev = np.zeros(14, dtype=np.float64)
    q_next = np.ones(14, dtype=np.float64) * 0.5
    bridge = _make_bridge_like(
        module,
        now_s=5.0,
        latest_action_stamp_s=4.9,
        timeout_s=0.2,
        active=True,
        buffer_data=(q_prev, q_next, 4.8, 4.9),
    )
    bridge.publish_joint_command_calls = []
    bridge.publish_hand_command_calls = 0
    bridge.publish_joint_command = lambda positions: bridge.publish_joint_command_calls.append(positions.copy())
    bridge.publish_hand_command_if_needed = lambda: setattr(
        bridge, "publish_hand_command_calls", bridge.publish_hand_command_calls + 1
    )
    bridge.process_upsampled_data = lambda *args: np.ones(14, dtype=np.float64) * 0.5

    module.TianqingA2Bridge.upsample_callback(bridge)

    assert len(bridge.publish_joint_command_calls) == 1
    np.testing.assert_allclose(bridge.publish_joint_command_calls[0], np.ones(14) * 0.15)
    assert bridge.publish_hand_command_calls == 1


def test_handle_action_dict_starts_rosbag_once_on_first_action(monkeypatch):
    module = _load_tianqing_ros2_module(monkeypatch)
    logger = DummyLogger()
    bridge = SimpleNamespace(
        use_kalman=False,
        current_arm_q=np.zeros(14, dtype=np.float64),
        current_hand_norm=np.zeros(2, dtype=np.float32),
        _action_lock=threading.Lock(),
        latest_model_action=np.zeros(16, dtype=np.float32),
        latest_arm_target=np.zeros(14, dtype=np.float64),
        latest_hand_target=np.zeros(2, dtype=np.float32),
        latest_action_stamp_s=0.0,
        _action_stream_active=False,
        upsample_buffer=DummyAppendBuffer(),
        args=SimpleNamespace(input_action_hz=30.0),
        rosbag_start_calls=0,
        _rosbag_process=None,
        _logger=logger,
    )
    bridge._now_s = lambda: 1.0
    bridge.get_logger = lambda: bridge._logger
    bridge.publish_joint_command_model = lambda positions: None
    bridge.pub_model_action = SimpleNamespace(publish=lambda msg: None)
    def _maybe_start_rosbag():
        if bridge._rosbag_process is None:
            bridge.rosbag_start_calls += 1
            bridge._rosbag_process = object()

    bridge.maybe_start_rosbag_recording = _maybe_start_rosbag

    action_dict = {name: float(i) for i, name in enumerate(module.ZMQ_MOTOR_NAMES)}

    module.TianqingA2Bridge.handle_action_dict(bridge, action_dict)
    module.TianqingA2Bridge.handle_action_dict(bridge, action_dict)

    assert bridge._action_stream_active is True
    assert bridge.rosbag_start_calls == 1
    assert len(bridge.upsample_buffer.items) == 3


def test_maybe_start_rosbag_recording_uses_mcap_output_dir(monkeypatch, tmp_path):
    module = _load_tianqing_ros2_module(monkeypatch)
    bridge = _make_rosbag_bridge(module, output_dir=str(tmp_path))
    popen_calls = []

    class DummyProcess:
        def __init__(self):
            self.pid = 1234

        def poll(self):
            return None

    monkeypatch.setattr(module.time, "strftime", lambda fmt: "20260317_120000")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda cmd, start_new_session: popen_calls.append((cmd, start_new_session)) or DummyProcess(),
    )

    bridge.maybe_start_rosbag_recording()

    assert len(popen_calls) == 1
    command, start_new_session = popen_calls[0]
    assert start_new_session is True
    assert command[:6] == ["ros2", "bag", "record", "-a", "-s", "mcap"]
    assert command[6:] == ["-o", str(tmp_path / "tianqing_a2_20260317_120000")]
    assert len(bridge._logger.infos) == 1


def test_stop_rosbag_recording_sends_sigint(monkeypatch):
    module = _load_tianqing_ros2_module(monkeypatch)
    killpg_calls = []

    class DummyProcess:
        def __init__(self):
            self.pid = 4321
            self.wait_calls = []

        def poll(self):
            return None

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            return 0

    process = DummyProcess()
    bridge = _make_rosbag_bridge(module, process=process)

    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

    bridge.stop_rosbag_recording()

    assert bridge._rosbag_process is None
    assert killpg_calls == [(4321, signal.SIGINT)]
    assert process.wait_calls == [3.0]
