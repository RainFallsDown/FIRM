#!/usr/bin/env python3
"""
Extract raw DAP metrics from SAM object masks and target-region masks.

Expected episode structure:
    episodes/
    └── episode_000000/
        ├── meta.json
        ├── actions.npy                       optional
        ├── target_mask.png
        └── masks/
            └── object/
                ├── 000000.png
                ├── 000001.png
                └── ...

Output:
    raw_episode_metrics_mask.jsonl

The output is not final DAP score. It is raw task-level metrics for episode_scorer.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def list_mask_files(mask_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(mask_dir.glob(ext))
    return sorted(files)


def mask_area(mask: np.ndarray) -> float:
    return float((mask > 0).sum())


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(float(b)) < 1e-8:
        return default
    return float(a) / float(b)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def overlap_area(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    return float(np.logical_and(mask_a > 0, mask_b > 0).sum())


def normalized_centroid_error(obj_mask: np.ndarray, target_mask: np.ndarray) -> float:
    obj_c = mask_centroid(obj_mask)
    tgt_c = mask_centroid(target_mask)
    if obj_c is None or tgt_c is None:
        return 1e9

    h, w = obj_mask.shape[:2]
    diag = math.sqrt(h * h + w * w)
    return float(np.linalg.norm(obj_c - tgt_c) / max(diag, 1e-8))


def pca_orientation_deg(mask: np.ndarray) -> Optional[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = pts.T @ pts / max(len(pts) - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return float(math.degrees(math.atan2(axis[1], axis[0])))


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def mask_compactness(mask: np.ndarray) -> float:
    """
    Compactness = 4*pi*area / perimeter^2.
    Lower values indicate elongated or irregular shapes.
    """
    area = mask_area(mask)
    if area <= 1:
        return 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    if perimeter <= 1e-8:
        return 0.0

    return float(4.0 * math.pi * area / (perimeter * perimeter))


def skeleton_length_proxy(mask: np.ndarray) -> float:
    """
    Lightweight cable-length proxy.
    Uses cv2.ximgproc.thinning if available; otherwise falls back to foreground area.
    """
    binary = (mask > 0).astype(np.uint8) * 255

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        skel = cv2.ximgproc.thinning(binary)
        return float((skel > 0).sum())

    return float((binary > 0).sum())


def load_actions(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def infer_timeout(actions: Optional[np.ndarray], max_steps: Optional[int]) -> bool:
    if actions is None or max_steps is None:
        return False
    return int(actions.shape[0]) >= int(max_steps)


def infer_jammed(
    actions: Optional[np.ndarray],
    first_mask: np.ndarray,
    final_mask: np.ndarray,
    cfg: Dict[str, Any],
) -> bool:
    """
    Weak proxy:
    if robot actions are non-trivial but object centroid moves little, mark possible jamming.
    """
    if actions is None or actions.size == 0:
        return False

    min_action_norm = float(cfg.get("min_action_norm", 1e-3))
    min_motion_px = float(cfg.get("min_object_motion_px", 5.0))

    action_norm = float(np.mean(np.linalg.norm(actions.reshape(actions.shape[0], -1), axis=1)))
    if action_norm < min_action_norm:
        return False

    c0 = mask_centroid(first_mask)
    c1 = mask_centroid(final_mask)
    if c0 is None or c1 is None:
        return False

    return float(np.linalg.norm(c1 - c0)) < min_motion_px


def common_metrics(first_mask: np.ndarray, final_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, Any]:
    obj_area = mask_area(final_mask)
    tgt_area = mask_area(target_mask)
    ov = overlap_area(final_mask, target_mask)

    overlap_ratio = clamp01(safe_div(ov, obj_area))
    target_coverage = clamp01(safe_div(ov, tgt_area))
    pose_error = normalized_centroid_error(final_mask, target_mask)

    return {
        "object_area": obj_area,
        "target_area": tgt_area,
        "overlap_area": ov,
        "object_target_overlap_ratio": overlap_ratio,
        "target_region_coverage": target_coverage,
        "pose_error": pose_error,
        "position_error": pose_error,
        "initial_compactness": mask_compactness(first_mask),
        "final_compactness": mask_compactness(final_mask),
    }


def extract_manual(first_mask: np.ndarray, final_mask: np.ndarray, target_mask: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = common_metrics(first_mask, final_mask, target_mask)

    angle = pca_orientation_deg(final_mask)
    target_angle = float(cfg.get("target_orientation_deg", 0.0))
    orientation_error = angle_diff_deg(angle, target_angle) if angle is not None else 180.0

    m.update({
        "insertion_depth": m["object_target_overlap_ratio"],
        "alignment_error": m["pose_error"],
        "orientation_error": orientation_error,
        "severe_fold": m["final_compactness"] < float(cfg.get("manual_min_compactness", 0.05)),
    })
    return m


def extract_cable(first_mask: np.ndarray, final_mask: np.ndarray, target_mask: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = common_metrics(first_mask, final_mask, target_mask)

    cable_len = skeleton_length_proxy(final_mask)
    cable_in_target = np.logical_and(final_mask > 0, target_mask > 0).astype(np.uint8)
    cable_in_target_len = skeleton_length_proxy(cable_in_target)
    contained_ratio = clamp01(safe_div(cable_in_target_len, cable_len))

    m.update({
        "contained_cable_length": contained_ratio,
        "residual_tangling": clamp01(1.0 - m["final_compactness"]),
        "boundary_contact": float(contained_ratio < float(cfg.get("min_contained_ratio", 0.8))),
        "dropped": m["object_area"] < float(cfg.get("min_visible_area", 20.0)),
        "slip": False,
    })
    return m


def extract_sponge(first_mask: np.ndarray, final_mask: np.ndarray, target_mask: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = common_metrics(first_mask, final_mask, target_mask)

    first_area = mask_area(first_mask)
    final_area = mask_area(final_mask)
    area_ratio = safe_div(final_area, first_area, default=1.0)

    m.update({
        "target_region_coverage": m["target_region_coverage"],
        "pose_error": m["pose_error"],
        "residual_compression": clamp01(1.0 - area_ratio),
        "rebound_shift": m["pose_error"],
        "folded_corner": m["final_compactness"] < float(cfg.get("sponge_folded_compactness", 0.20)),
        "trapped_corner": False,
        "dropped": m["object_area"] < float(cfg.get("min_visible_area", 20.0)),
    })
    return m


def extract_tape(first_mask: np.ndarray, final_mask: np.ndarray, target_mask: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = common_metrics(first_mask, final_mask, target_mask)

    angle = pca_orientation_deg(final_mask)
    target_angle = float(cfg.get("target_orientation_deg", 0.0))
    orientation_error = angle_diff_deg(angle, target_angle) if angle is not None else 180.0

    m.update({
        "position_error": m["position_error"],
        "orientation_error": orientation_error,
        "outside_target": m["object_target_overlap_ratio"] < float(cfg.get("min_overlap_ratio", 0.3)),
        "rolling": False,
        "face_down": False,
        "on_lid": False,
        "slip": False,
    })
    return m


def extract_box(first_mask: np.ndarray, final_mask: np.ndarray, target_mask: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = common_metrics(first_mask, final_mask, target_mask)

    angle = pca_orientation_deg(final_mask)
    fold_angle = angle if angle is not None else 0.0

    m.update({
        "fold_angle": fold_angle,
        "hinge_jam": False,
        "collision": False,
        "severe_deformation": m["final_compactness"] < float(cfg.get("box_min_compactness", 0.05)),
    })
    return m


EXTRACTORS = {
    "Manual": extract_manual,
    "Cable": extract_cable,
    "Sponge": extract_sponge,
    "Tape": extract_tape,
    "Box": extract_box,
}


def extract_episode(episode_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    meta = load_json(episode_dir / "meta.json")
    task = meta.get("task")
    if task not in EXTRACTORS:
        raise ValueError(f"Unknown task `{task}` in {episode_dir}")

    mask_files = list_mask_files(episode_dir / "masks" / "object")
    if not mask_files:
        raise FileNotFoundError(f"No object masks found in {episode_dir / 'masks/object'}")

    target_mask_path = episode_dir / "target_mask.png"
    if not target_mask_path.exists():
        raise FileNotFoundError(f"Missing target_mask.png in {episode_dir}")

    first_mask = load_mask(mask_files[0])
    final_mask = load_mask(mask_files[-1])
    target_mask = load_mask(target_mask_path)

    task_cfg = config.get("tasks", {}).get(task, {})
    metrics = EXTRACTORS[task](first_mask, final_mask, target_mask, task_cfg)

    actions = load_actions(episode_dir / "actions.npy")
    metrics["timeout"] = infer_timeout(actions, task_cfg.get("max_steps"))
    metrics["jammed"] = infer_jammed(actions, first_mask, final_mask, task_cfg)

    # Optional manual/VLM/model override.
    for key in ("manual_flags", "manual_metrics", "model_annotations"):
        value = meta.get(key, {})
        if isinstance(value, dict):
            metrics.update(value)

    record = {
        "episode_id": meta.get("episode_id", episode_dir.name),
        "method": meta.get("method", "unknown_method"),
        "task": task,
        "metrics": metrics,
    }

    if "perturbation" in meta:
        record["perturbation"] = meta["perturbation"]

    return record


def find_episode_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()])


def write_jsonl(records: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    episode_dirs = find_episode_dirs(args.episodes_root)

    records = []
    for episode_dir in episode_dirs:
        try:
            records.append(extract_episode(episode_dir, config))
        except Exception as exc:
            print(f"[WARN] {episode_dir}: {exc}")

    write_jsonl(records, args.output)
    print(f"[OK] Processed {len(records)} / {len(episode_dirs)} episodes")
    print(f"[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()