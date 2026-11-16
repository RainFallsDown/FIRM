"""Replayable, isolated perturbations for the FIRM robustness protocol."""

from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
FAMILIES = ("object", "fixture", "scene")


@dataclass(frozen=True)
class Condition:
    task: str
    family: str
    delta: float
    trial: int
    seed: int
    object_xy: tuple
    object_yaw_deg: float
    fixture_xy: tuple
    fixture_yaw_deg: float
    camera_xyz: tuple
    camera_rotvec_deg: tuple
    rgb_sigma: float
    depth_bound_m: float
    distractors: tuple

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task.isidentifier():
            raise ValueError("Task must be a valid registry identifier")
        if self.family not in FAMILIES or self.delta not in LEVELS:
            raise ValueError("Unknown perturbation family or strength")
        if type(self.trial) is not int or self.trial not in range(20):
            raise ValueError("Trial must be an integer from 0 to 19")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31 - 1:
            raise ValueError("Seed must be a nonnegative signed 32-bit integer")
        vectors = (
            (self.object_xy, 2, 0.04 if self.family == "object" else 0),
            (self.fixture_xy, 2, 0.02 if self.family == "fixture" else 0),
            (self.camera_xyz, 3, 0.01 if self.family == "scene" else 0),
            (self.camera_rotvec_deg, 3, 5 if self.family == "scene" else 0),
        )
        for value, dimensions, bound in vectors:
            values = np.asarray(value)
            if (
                values.shape != (dimensions,)
                or not np.all(np.isfinite(values))
                or np.linalg.norm(values) > bound * self.delta + 1e-12
            ):
                raise ValueError("Perturbation vector exceeds protocol bounds")
        for value, bound in (
            (self.object_yaw_deg, 20 if self.family == "object" else 0),
            (self.fixture_yaw_deg, 10 if self.family == "fixture" else 0),
        ):
            if not np.isfinite(value) or abs(value) > bound * self.delta:
                raise ValueError("Yaw exceeds protocol bounds")
        scene = self.family == "scene"
        if not np.isclose(
            self.rgb_sigma, self.delta * 10 / 255 if scene else 0, atol=1e-12
        ):
            raise ValueError("RGB sigma does not match protocol")
        if not np.isclose(
            self.depth_bound_m, self.delta * 0.005 if scene else 0, atol=1e-12
        ):
            raise ValueError("Depth bound does not match protocol")
        if len(self.distractors) != (int(4 * self.delta) if scene else 0):
            raise ValueError("Distractor count does not match protocol")
        if self.distractors and (
            np.asarray(self.distractors).shape != (len(self.distractors), 3)
            or not np.all(np.isfinite(self.distractors))
        ):
            raise ValueError("Invalid distractor positions")

    def to_dict(self):
        return asdict(self)

    @property
    def id(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()


def _ball(rng, radius, dimensions):
    direction = rng.normal(size=dimensions)
    direction /= np.linalg.norm(direction)
    return tuple(
        float(x) for x in direction * radius * rng.random() ** (1 / dimensions)
    )


def generate_conditions(tasks, seed=0, trials=20):
    """Uniform translations in a disk/ball, uniform yaw, paired across methods."""
    if trials != 20:
        raise ValueError("The release protocol requires 20 trials per level")
    for task in sorted(set(tasks)):
        for family in FAMILIES:
            for delta in LEVELS:
                for trial in range(trials):
                    # Stable under task reordering and additions; no method identifier.
                    key = f"{seed}:{task}:{family}:{trial}".encode()
                    sample_seed = int.from_bytes(
                        hashlib.sha256(key).digest()[:4], "big"
                    ) % (2**31 - 1)
                    rng = np.random.default_rng(sample_seed)
                    obj = family == "object"
                    fixture = family == "fixture"
                    scene = family == "scene"
                    xy = _ball(rng, 0.04 * delta if obj else 0, 2)
                    yaw = float(rng.uniform(-20, 20) * delta) if obj else 0.0
                    fxy = _ball(rng, 0.02 * delta if fixture else 0, 2)
                    fyaw = float(rng.uniform(-10, 10) * delta) if fixture else 0.0
                    camera = _ball(rng, 0.01 * delta if scene else 0, 3)
                    rotation = _ball(rng, 5 * delta if scene else 0, 3)
                    # Peripheral slots are clear of the central task and spawn zone.
                    slots = [
                        (0.18, -0.19, 0.73),
                        (0.92, -0.19, 0.73),
                        (0.18, 0.19, 0.73),
                        (0.92, 0.19, 0.73),
                    ]
                    order = rng.permutation(4)
                    distractors = tuple(
                        slots[i] for i in order[: int(4 * delta) if scene else 0]
                    )
                    yield Condition(
                        task,
                        family,
                        delta,
                        trial,
                        sample_seed,
                        xy,
                        yaw,
                        fxy,
                        fyaw,
                        camera,
                        rotation,
                        delta * 10 / 255 if scene else 0.0,
                        delta * 0.005 if scene else 0.0,
                        distractors,
                    )


def noisy_observation(rgb, depth, condition, frame):
    """Gaussian RGB; bounded uniform metric depth noise; preserve invalid depth."""
    rng = np.random.default_rng(np.random.SeedSequence([condition.seed, frame, 917]))
    integer = np.issubdtype(rgb.dtype, np.integer)
    image = rgb.astype(np.float64) / (255 if integer else 1)
    image = np.clip(image + rng.normal(0, condition.rgb_sigma, image.shape), 0, 1)
    image = (
        np.rint(image * 255).astype(rgb.dtype) if integer else image.astype(rgb.dtype)
    )
    valid = np.isfinite(depth) & (depth > 0)
    result = depth.copy()
    noise = rng.uniform(-condition.depth_bound_m, condition.depth_bound_m, depth.shape)
    result[valid] = np.maximum(depth[valid] + noise[valid], 0)
    return image, result
