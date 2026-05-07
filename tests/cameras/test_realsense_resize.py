import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation


def _load_realsense_modules(monkeypatch):
    dummy_rs = types.ModuleType("pyrealsense2")
    dummy_rs.pipeline = type("pipeline", (), {})
    dummy_rs.pipeline_profile = type("pipeline_profile", (), {})
    dummy_rs.config = type("config", (), {})
    dummy_rs.stream = SimpleNamespace(color=1, depth=2)
    dummy_rs.format = SimpleNamespace(rgb8=1, z16=2)

    monkeypatch.setitem(sys.modules, "pyrealsense2", dummy_rs)
    sys.modules.pop("lerobot.cameras.realsense.camera_realsense", None)
    sys.modules.pop("lerobot.cameras.realsense.configuration_realsense", None)
    sys.modules.pop("lerobot.cameras.realsense", None)

    config_module = importlib.import_module("lerobot.cameras.realsense.configuration_realsense")
    camera_module = importlib.import_module("lerobot.cameras.realsense.camera_realsense")
    return config_module, camera_module


def test_realsense_config_requires_capture_dimensions_as_a_pair(monkeypatch):
    config_module, _ = _load_realsense_modules(monkeypatch)

    with pytest.raises(ValueError, match="capture_width"):
        config_module.RealSenseCameraConfig(
            serial_number_or_name="042",
            fps=30,
            width=640,
            height=480,
            capture_width=1280,
        )


def test_realsense_config_requires_fps_when_capture_dimensions_are_overridden(monkeypatch):
    config_module, _ = _load_realsense_modules(monkeypatch)

    with pytest.raises(ValueError, match="fps"):
        config_module.RealSenseCameraConfig(
            serial_number_or_name="042",
            capture_width=1280,
            capture_height=720,
        )


def test_realsense_camera_uses_capture_override_without_changing_output_shape(monkeypatch):
    config_module, camera_module = _load_realsense_modules(monkeypatch)
    config = config_module.RealSenseCameraConfig(
        serial_number_or_name="042",
        fps=30,
        width=640,
        height=480,
        capture_width=1280,
        capture_height=720,
    )
    camera = camera_module.RealSenseCamera(config)

    assert camera.capture_width == 1280
    assert camera.capture_height == 720
    assert camera.width == 640
    assert camera.height == 480

    raw_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    processed = camera._postprocess_image(raw_image)

    assert processed.shape == (480, 640, 3)


def test_realsense_camera_keeps_rotation_based_capture_defaults(monkeypatch):
    config_module, camera_module = _load_realsense_modules(monkeypatch)
    config = config_module.RealSenseCameraConfig(
        serial_number_or_name="042",
        fps=30,
        width=640,
        height=480,
        rotation=Cv2Rotation.ROTATE_90,
    )
    camera = camera_module.RealSenseCamera(config)

    assert camera.capture_width == 480
    assert camera.capture_height == 640
