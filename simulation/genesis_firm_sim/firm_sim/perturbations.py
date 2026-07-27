"""Deterministic perturbation schedules for FIRM-Sim evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PerturbationLevel:
    """Maximum physical and perception amplitudes for one evaluation level."""

    name: str
    object_translation_m: float
    fixture_translation_m: float
    object_yaw_deg: float
    pose_position_noise_m: float
    pose_rotation_noise_deg: float
    rgb_noise: float
    depth_noise_m: float


@dataclass(frozen=True)
class PerturbationSample:
    """One fully resolved perturbation condition that can be logged and replayed."""

    level: str
    axis: str
    seed: int
    object_translation_xy_m: tuple[float, float]
    fixture_translation_xy_m: tuple[float, float]
    object_yaw_deg: float
    pose_position_noise_m: float
    pose_rotation_noise_deg: float
    rgb_noise: float
    depth_noise_m: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


PERTURBATION_LEVELS = {
    "nominal": PerturbationLevel("nominal", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "low": PerturbationLevel("low", 0.010, 0.005, 5.0, 0.0025, 1.25, 2.5 / 255.0, 0.00125),
    "medium": PerturbationLevel("medium", 0.020, 0.010, 10.0, 0.005, 2.5, 5.0 / 255.0, 0.0025),
    "high": PerturbationLevel("high", 0.040, 0.020, 20.0, 0.010, 5.0, 10.0 / 255.0, 0.005),
}

PERTURBATION_AXES = (
    "none",
    "object_translation",
    "fixture_translation",
    "object_yaw",
    "pose_noise",
    "rgb_noise",
    "depth_noise",
    "combined",
)


def perturbation_levels() -> tuple[str, ...]:
    return tuple(PERTURBATION_LEVELS)


def perturbation_axes() -> tuple[str, ...]:
    return PERTURBATION_AXES


def get_perturbation_level(name: str) -> PerturbationLevel:
    try:
        return PERTURBATION_LEVELS[name]
    except KeyError as exc:
        available = ", ".join(perturbation_levels())
        raise KeyError(f"Unknown perturbation level '{name}'. Available levels: {available}") from exc


def sample_perturbation(
    level: str = "nominal",
    axis: str = "none",
    seed: int = 0,
) -> PerturbationSample:
    """Sample a replayable condition with the requested axis isolated.

    Translation and yaw use the exact level amplitude with a seeded direction or
    sign. Perception fields are maximum absolute noise amplitudes; individual
    pixel or pose noise is sampled by the downstream observation pipeline.
    """
    profile = get_perturbation_level(level)
    if axis not in PERTURBATION_AXES:
        available = ", ".join(PERTURBATION_AXES)
        raise KeyError(f"Unknown perturbation axis '{axis}'. Available axes: {available}")

    rng = np.random.default_rng(seed)
    enabled = lambda candidate: axis in {candidate, "combined"}

    object_xy = (
        _sample_planar_offset(rng, profile.object_translation_m)
        if enabled("object_translation")
        else (0.0, 0.0)
    )
    fixture_xy = (
        _sample_planar_offset(rng, profile.fixture_translation_m)
        if enabled("fixture_translation")
        else (0.0, 0.0)
    )
    yaw = (
        float(profile.object_yaw_deg * rng.choice((-1.0, 1.0)))
        if enabled("object_yaw") and profile.object_yaw_deg
        else 0.0
    )

    return PerturbationSample(
        level=level,
        axis=axis,
        seed=seed,
        object_translation_xy_m=object_xy,
        fixture_translation_xy_m=fixture_xy,
        object_yaw_deg=yaw,
        pose_position_noise_m=profile.pose_position_noise_m if enabled("pose_noise") else 0.0,
        pose_rotation_noise_deg=profile.pose_rotation_noise_deg if enabled("pose_noise") else 0.0,
        rgb_noise=profile.rgb_noise if enabled("rgb_noise") else 0.0,
        depth_noise_m=profile.depth_noise_m if enabled("depth_noise") else 0.0,
    )


def apply_rgb_noise(image: np.ndarray, max_amplitude: float, seed: int) -> np.ndarray:
    """Apply bounded uniform RGB noise while preserving the input dtype."""
    if max_amplitude <= 0.0:
        return image.copy()
    rng = np.random.default_rng(seed)
    is_integer = np.issubdtype(image.dtype, np.integer)
    normalized = image.astype(np.float32) / 255.0 if is_integer else image.astype(np.float32)
    noise = rng.uniform(-max_amplitude, max_amplitude, size=normalized.shape)
    noisy = np.clip(normalized + noise, 0.0, 1.0)
    return np.rint(noisy * 255.0).astype(image.dtype) if is_integer else noisy.astype(image.dtype)


def apply_depth_noise(depth_m: np.ndarray, max_amplitude_m: float, seed: int) -> np.ndarray:
    """Apply bounded uniform metric depth noise without creating negative depth."""
    if max_amplitude_m <= 0.0:
        return depth_m.copy()
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-max_amplitude_m, max_amplitude_m, size=depth_m.shape)
    return np.maximum(depth_m.astype(np.float32) + noise, 0.0).astype(depth_m.dtype)


def _sample_planar_offset(rng: np.random.Generator, radius_m: float) -> tuple[float, float]:
    if radius_m <= 0.0:
        return (0.0, 0.0)
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    return (radius_m * math.cos(angle), radius_m * math.sin(angle))
