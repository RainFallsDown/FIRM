#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def overlay(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = base.copy()
    idx = mask > 0
    out[idx] = (0.6 * out[idx] + 0.4 * np.array(color)).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", default="episode_000000")
    parser.add_argument("--camera", default="observation.images.hand.right.color")
    parser.add_argument("--mask-index", default="000008")
    parser.add_argument("--metrics-jsonl", default="annotation_workspace_sponge/raw_episode_metrics_mask_sam2_strict.jsonl")
    parser.add_argument("--output-dir", default="results_sponge_sam2_strict")
    args = parser.parse_args()

    root = Path.cwd()
    ep = root / "annotation_workspace_sponge" / "episodes" / args.episode_id
    img = imread_unicode(ep / "final_frames" / f"{args.camera}.png", cv2.IMREAD_COLOR)
    tgt = imread_unicode(ep / "target_mask.png", cv2.IMREAD_GRAYSCALE)
    pred = imread_unicode(ep / "masks" / "object" / f"{args.mask_index}.png", cv2.IMREAD_GRAYSCALE)
    if img is None or tgt is None or pred is None:
        raise FileNotFoundError("Failed to load one or more image files for visualization.")

    tgtb = (tgt > 127).astype(np.uint8)
    predb = (pred > 127).astype(np.uint8)
    ov = (tgtb & predb).astype(np.uint8)

    p1 = img.copy()
    p2 = overlay(img, tgtb, (0, 255, 0))
    p3 = overlay(img, predb, (255, 0, 0))
    p4 = overlay(img, tgtb, (0, 180, 0))
    p4 = overlay(p4, predb, (180, 0, 0))
    p4 = overlay(p4, ov, (0, 255, 255))

    labels = [
        "Final Frame",
        "Target Mask (Green)",
        "Pred Mask (Blue)",
        "Overlap (Yellow)",
    ]
    panels = [p1, p2, p3, p4]
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    canvas = np.vstack([top, bottom])

    metrics = None
    metrics_path = root / args.metrics_jsonl
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("episode_id") == args.episode_id:
            metrics = row["metrics"]
            break

    text = [
        f"{args.episode_id} | task=Sponge",
        f"camera={args.camera}",
        f"coverage={metrics['target_region_coverage']:.4f}",
        f"pose_error={metrics['pose_error']:.4f}",
        f"residual_compression={metrics['residual_compression']:.4f}",
        f"folded_corner={metrics['folded_corner']} | jammed={metrics['jammed']}",
    ]
    y = 40
    for line in text:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        y += 36

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    camera_tag = args.camera.replace(".", "_")
    out = outdir / f"example_{args.episode_id}_{camera_tag}.png"
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode visualization image.")
    buf.tofile(str(out))
    if not out.exists():
        raise RuntimeError(f"Failed to save visualization image: {out}")
    print(str(out))


if __name__ == "__main__":
    main()
