#!/usr/bin/env python3
"""Validate whether episode target masks look real or synthetic.

This script flags masks that exactly match the old center-rectangle template
or are empty/missing, reports area anomalies, and can write overlay previews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HEAD_COLOR_CAMERA = "observation.images.head.color"


def center_template(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    ry, rx = height // 3, width // 3
    mask[cy - ry:cy + ry, cx - rx:cx + rx] = 255
    return mask


def load_preview_image(episode_dir: Path, camera_key: str) -> np.ndarray | None:
    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        image = cv2.imread(str(final_frame))
        if image is not None:
            return image

    sampled_dir = episode_dir / "sampled_frames" / camera_key
    frame_files = sorted(sampled_dir.glob("*.png")) if sampled_dir.exists() else []
    if frame_files:
        return cv2.imread(str(frame_files[-1]))
    return None


def write_overlay(image: np.ndarray, target: np.ndarray, output_path: Path, status: str) -> None:
    overlay = image.copy()
    mask = target > 0
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.putText(
        overlay,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if status != "ok" else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def classify_target_mask(path: Path, min_area_ratio: float, max_area_ratio: float) -> tuple[str, dict]:
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        return "missing_or_unreadable", {}

    target = ((target > 127).astype(np.uint8) * 255)
    nonzero = int((target > 0).sum())
    if nonzero == 0:
        return "empty", {"area_px": 0, "area_ratio": 0.0}

    area_ratio = float(nonzero) / float(target.shape[0] * target.shape[1])
    details = {"area_px": nonzero, "area_ratio": area_ratio}

    template = center_template(target.shape)
    if np.array_equal(target, template):
        return "center_template", details
    if area_ratio < min_area_ratio:
        return "too_small", details
    if area_ratio > max_area_ratio:
        return "too_large", details

    return "ok", details


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate target masks for FIRM episodes")
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON report path",
    )
    parser.add_argument(
        "--camera",
        default=HEAD_COLOR_CAMERA,
        help="Camera key for overlay preview frames",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Optional directory for target-mask overlay PNG previews",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.005,
        help="Flag target masks smaller than this image-area fraction",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.8,
        help="Flag target masks larger than this image-area fraction",
    )
    args = parser.parse_args()

    episodes = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    report = []
    summary = {
        "ok": 0,
        "center_template": 0,
        "empty": 0,
        "missing_or_unreadable": 0,
        "too_small": 0,
        "too_large": 0,
    }

    for ep in episodes:
        target_path = ep / "target_mask.png"
        status, details = classify_target_mask(
            target_path,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )
        summary[status] = summary.get(status, 0) + 1
        row = {"episode_id": ep.name, "status": status, **details}
        report.append(row)

        if args.overlay_dir is not None and target_path.exists():
            image = load_preview_image(ep, args.camera)
            target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            if image is not None and target is not None:
                target = ((target > 127).astype(np.uint8) * 255)
                write_overlay(image, target, args.overlay_dir / f"{ep.name}.png", status)

    print(json.dumps({"summary": summary, "episodes": report}, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps({"summary": summary, "episodes": report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
