#!/usr/bin/env python3
"""Test SAM2 fixed-ROI prompts for target mask segmentation.

This script does NOT use an open-vocabulary detector.

Pipeline:
1) Load episode final head-camera frames.
2) Collect existing target_mask.png files.
3) Estimate a fixed ROI from target masks.
4) Sample 3x3 / 5x5 positive points inside the ROI.
5) Run SAM2 image predictor with fixed ROI / point prompts.
6) Compare predicted masks with target_mask.png.
7) Save summary.csv, episode_metrics.csv, and overlays.

Expected episode structure:
annotation_workspace_sponge/episodes/
└── episode_000000/
    ├── target_mask.png
    ├── final_frames/
    │   └── observation.images.head.color.png
    └── sampled_frames/
        └── observation.images.head.color/
            └── *.png

Example:
python firm_benchmark\\test_sam2_fixed_roi.py ^
  --episodes-root annotation_workspace_sponge\\episodes ^
  --checkpoint sam2\\checkpoints\\sam2.1_hiera_tiny.pt ^
  --output-dir results_sam2_fixed_roi ^
  --max-episodes 10 ^
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


HEAD_COLOR_CAMERA = "observation.images.head.color"


# -----------------------------
# Basic utilities
# -----------------------------


def list_episodes(episodes_root: Path) -> List[Path]:
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root does not exist: {episodes_root}")

    return sorted(
        p for p in episodes_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )


def load_preview_image(episode_dir: Path, camera_key: str) -> np.ndarray:
    """Load final frame first; fallback to the last sampled frame."""
    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        image = cv2.imread(str(final_frame), cv2.IMREAD_COLOR)
        if image is not None:
            return image

    sampled_dir = episode_dir / "sampled_frames" / camera_key
    frame_files = sorted(sampled_dir.glob("*.png")) if sampled_dir.exists() else []
    if frame_files:
        image = cv2.imread(str(frame_files[-1]), cv2.IMREAD_COLOR)
        if image is not None:
            return image

    raise FileNotFoundError(
        f"No usable preview frame for {episode_dir.name} with camera `{camera_key}`"
    )


def load_target_mask(episode_dir: Path) -> Optional[np.ndarray]:
    path = episode_dir / "target_mask.png"
    if not path.exists():
        return None

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = ((mask > 127).astype(np.uint8) * 255)
    if int((mask > 0).sum()) == 0:
        return None

    return mask


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def clip_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1 = int(round(max(0, min(width - 1, x1))))
    y1 = int(round(max(0, min(height - 1, y1))))
    x2 = int(round(max(0, min(width, x2))))
    y2 = int(round(max(0, min(height, y2))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def expand_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    dx = bw * margin
    dy = bh * margin

    return clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def compute_fixed_roi(
    episodes: List[Path],
    camera_key: str,
    roi_mode: str,
    margin: float,
) -> Tuple[int, int, int, int]:
    """Compute fixed ROI from all available target masks."""
    boxes: List[Tuple[int, int, int, int]] = []
    image_shape = None

    for episode_dir in episodes:
        mask = load_target_mask(episode_dir)
        if mask is None:
            continue

        bbox = mask_bbox(mask)
        if bbox is None:
            continue

        boxes.append(bbox)

        if image_shape is None:
            image = load_preview_image(episode_dir, camera_key)
            image_shape = image.shape[:2]

    if not boxes:
        raise RuntimeError(
            "No valid target_mask.png found. Please annotate target masks first."
        )

    if image_shape is None:
        raise RuntimeError("No image shape available.")

    height, width = image_shape

    if roi_mode == "union":
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        roi = (x1, y1, x2, y2)

    elif roi_mode == "median":
        centers_x = np.array([(b[0] + b[2]) / 2.0 for b in boxes], dtype=np.float32)
        centers_y = np.array([(b[1] + b[3]) / 2.0 for b in boxes], dtype=np.float32)
        widths = np.array([b[2] - b[0] for b in boxes], dtype=np.float32)
        heights = np.array([b[3] - b[1] for b in boxes], dtype=np.float32)

        cx = float(np.median(centers_x))
        cy = float(np.median(centers_y))
        bw = float(np.median(widths))
        bh = float(np.median(heights))

        roi = (
            int(round(cx - bw / 2.0)),
            int(round(cy - bh / 2.0)),
            int(round(cx + bw / 2.0)),
            int(round(cy + bh / 2.0)),
        )

    else:
        raise ValueError(f"Unsupported roi_mode: {roi_mode}")

    roi = expand_box(roi, width=width, height=height, margin=margin)
    return roi


def sample_grid_points(
    roi: Tuple[int, int, int, int],
    grid_size: int,
    inner_margin: float,
) -> np.ndarray:
    """Sample grid points inside ROI.

    Args:
        roi: x1, y1, x2, y2
        grid_size: 3 or 5
        inner_margin: avoid sampling exactly on box boundary

    Returns:
        point_coords: shape (N, 2), order x,y
    """
    x1, y1, x2, y2 = roi
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    px1 = x1 + inner_margin * w
    px2 = x2 - inner_margin * w
    py1 = y1 + inner_margin * h
    py2 = y2 - inner_margin * h

    if grid_size <= 1:
        xs = np.array([(x1 + x2) / 2.0], dtype=np.float32)
        ys = np.array([(y1 + y2) / 2.0], dtype=np.float32)
    else:
        xs = np.linspace(px1, px2, grid_size, dtype=np.float32)
        ys = np.linspace(py1, py2, grid_size, dtype=np.float32)

    points = []
    for y in ys:
        for x in xs:
            points.append([float(x), float(y)])

    return np.asarray(points, dtype=np.float32)


def compute_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask > 0
    target = target_mask > 0

    pred_area = int(pred.sum())
    target_area = int(target.sum())
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())

    coverage = inter / max(target_area, 1)
    iou = inter / max(union, 1)

    return {
        "coverage": float(coverage),
        "iou": float(iou),
        "intersection": float(inter),
        "union": float(union),
        "pred_area": float(pred_area),
        "target_area": float(target_area),
    }


def create_overlay(
    image_bgr: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    roi: Tuple[int, int, int, int],
    points: Optional[np.ndarray],
    title: str,
) -> np.ndarray:
    """Create overlay.

    Color convention:
    - target only: green
    - prediction only: red
    - overlap: yellow
    - ROI: blue box
    - positive points: cyan dots
    """
    image = image_bgr.copy()

    pred = pred_mask > 0
    target = target_mask > 0
    overlap = np.logical_and(pred, target)
    pred_only = np.logical_and(pred, ~target)
    target_only = np.logical_and(target, ~pred)

    overlay = image.copy()

    alpha = 0.45

    # target only: green
    overlay[target_only] = (
        (1 - alpha) * overlay[target_only]
        + alpha * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    # pred only: red
    overlay[pred_only] = (
        (1 - alpha) * overlay[pred_only]
        + alpha * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)

    # overlap: yellow
    overlay[overlap] = (
        (1 - alpha) * overlay[overlap]
        + alpha * np.array([0, 255, 255], dtype=np.float32)
    ).astype(np.uint8)

    x1, y1, x2, y2 = roi
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if points is not None:
        for x, y in points:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (255, 255, 0), -1)

    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


# -----------------------------
# SAM2 utilities
# -----------------------------


def infer_sam2_model_cfg(checkpoint: Path) -> str:
    """Infer SAM2 model config from checkpoint filename.

    You can override this by passing --model-cfg.
    """
    name = checkpoint.name.lower()

    if "sam2.1" in name:
        if "hiera_tiny" in name or "hiera_t" in name:
            return "configs/sam2.1/sam2.1_hiera_t.yaml"
        if "hiera_small" in name or "hiera_s" in name:
            return "configs/sam2.1/sam2.1_hiera_s.yaml"
        if "hiera_base_plus" in name or "hiera_b+" in name or "hiera_base" in name:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        if "hiera_large" in name or "hiera_l" in name:
            return "configs/sam2.1/sam2.1_hiera_l.yaml"

    if "hiera_tiny" in name or "hiera_t" in name:
        return "configs/sam2/sam2_hiera_t.yaml"
    if "hiera_small" in name or "hiera_s" in name:
        return "configs/sam2/sam2_hiera_s.yaml"
    if "hiera_base_plus" in name or "hiera_b+" in name or "hiera_base" in name:
        return "configs/sam2/sam2_hiera_b+.yaml"
    if "hiera_large" in name or "hiera_l" in name:
        return "configs/sam2/sam2_hiera_l.yaml"

    raise ValueError(
        f"Cannot infer SAM2 model config from checkpoint name: {checkpoint.name}. "
        f"Please pass --model-cfg explicitly."
    )


def load_sam2_predictor(
    checkpoint: Path,
    model_cfg: Optional[str],
    device: str,
):
    """Load SAM2 image predictor."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise ImportError(
            "Failed to import SAM2. Please make sure you are in the sam2_local "
            "environment and the SAM2 repo/package is installed."
        ) from exc

    if model_cfg is None:
        model_cfg = infer_sam2_model_cfg(checkpoint)

    print(f"[INFO] Loading SAM2")
    print(f"[INFO] model_cfg = {model_cfg}")
    print(f"[INFO] checkpoint = {checkpoint}")
    print(f"[INFO] device = {device}")

    sam2_model = build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint),
        device=device,
    )

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, model_cfg


def predict_with_sam2(
    predictor,
    image_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
    points: Optional[np.ndarray],
    use_box: bool,
    multimask_output: bool,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Run SAM2 prediction.

    Returns:
        selected_mask: uint8 HxW, foreground=255
        selected_score: float
        all_masks: boolean/float masks, shape KxHxW
        all_scores: shape K
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    box = None
    if use_box:
        x1, y1, x2, y2 = roi
        box = np.asarray([x1, y1, x2, y2], dtype=np.float32)

    point_coords = None
    point_labels = None
    if points is not None and len(points) > 0:
        point_coords = points.astype(np.float32)
        point_labels = np.ones((len(points),), dtype=np.int32)

    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask_output,
    )

    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    if masks.ndim == 2:
        masks = masks[None, ...]

    if len(scores) == 0:
        raise RuntimeError("SAM2 returned no scores.")

    best_idx = int(np.argmax(scores))
    selected = masks[best_idx]
    selected_mask = ((selected > 0).astype(np.uint8) * 255)

    selected_score = float(scores[best_idx])
    return selected_mask, selected_score, masks, scores


# -----------------------------
# Main evaluation
# -----------------------------


def build_prompt_specs(
    grid_sizes: List[int],
    include_box_only: bool,
    include_box_points: bool,
    include_points_only: bool,
) -> List[Dict]:
    specs = []

    if include_box_only:
        specs.append(
            {
                "name": "box_only",
                "grid_size": 0,
                "use_box": True,
                "use_points": False,
            }
        )

    for grid_size in grid_sizes:
        if include_points_only:
            specs.append(
                {
                    "name": f"points_{grid_size}x{grid_size}",
                    "grid_size": grid_size,
                    "use_box": False,
                    "use_points": True,
                }
            )

        if include_box_points:
            specs.append(
                {
                    "name": f"box_points_{grid_size}x{grid_size}",
                    "grid_size": grid_size,
                    "use_box": True,
                    "use_points": True,
                }
            )

    return specs


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    by_prompt: Dict[str, List[Dict]] = {}

    for row in rows:
        by_prompt.setdefault(row["prompt"], []).append(row)

    summary = []
    for prompt, items in by_prompt.items():
        n = len(items)

        def mean(key: str) -> float:
            vals = [float(x[key]) for x in items]
            if not vals:
                return 0.0
            return float(np.mean(vals))

        summary.append(
            {
                "prompt": prompt,
                "episodes": n,
                "coverage": mean("coverage"),
                "iou": mean("iou"),
                "oracle_iou": mean("oracle_iou"),
                "sam2_score": mean("sam2_score"),
                "pred_area": mean("pred_area"),
                "target_area": mean("target_area"),
            }
        )

    summary.sort(key=lambda x: (x["iou"], x["coverage"]), reverse=True)
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test SAM2 fixed-ROI positive point prompts."
    )

    parser.add_argument(
        "--episodes-root",
        required=True,
        type=Path,
        help="Root directory containing episode_* folders.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="SAM2 checkpoint path, e.g. sam2/checkpoints/sam2.1_hiera_tiny.pt.",
    )

    parser.add_argument(
        "--model-cfg",
        default=None,
        help=(
            "SAM2 model config. If omitted, inferred from checkpoint name. "
            "Example: configs/sam2.1/sam2.1_hiera_t.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save results.",
    )

    parser.add_argument(
        "--camera",
        default=HEAD_COLOR_CAMERA,
        help="Camera key, default observation.images.head.color.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this episode list index.",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to evaluate.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu.",
    )

    parser.add_argument(
        "--roi-mode",
        default="union",
        choices=["union", "median"],
        help="How to estimate fixed ROI from target masks.",
    )

    parser.add_argument(
        "--roi-margin",
        type=float,
        default=0.15,
        help="Expand fixed ROI by this fraction.",
    )

    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Grid sizes for positive point prompts.",
    )

    parser.add_argument(
        "--inner-margin",
        type=float,
        default=0.20,
        help="Inner margin for grid point sampling inside ROI.",
    )

    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Use SAM2 multimask output. Recommended for debugging.",
    )

    parser.add_argument(
        "--no-box-only",
        action="store_true",
        help="Disable box-only prompt.",
    )

    parser.add_argument(
        "--no-box-points",
        action="store_true",
        help="Disable box + points prompts.",
    )

    parser.add_argument(
        "--no-points-only",
        action="store_true",
        help="Disable points-only prompts.",
    )

    args = parser.parse_args()

    episodes_all = list_episodes(args.episodes_root)
    episodes = episodes_all[args.start_index :]

    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    if not episodes:
        raise RuntimeError("No episodes selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] episodes_root = {args.episodes_root}")
    print(f"[INFO] selected episodes = {len(episodes)}")
    print(f"[INFO] first episode = {episodes[0].name}")
    print(f"[INFO] last episode = {episodes[-1].name}")
    print(f"[INFO] output_dir = {args.output_dir}")

    roi = compute_fixed_roi(
        episodes=episodes,
        camera_key=args.camera,
        roi_mode=args.roi_mode,
        margin=args.roi_margin,
    )

    print(f"[INFO] fixed ROI = {roi}")

    roi_txt = args.output_dir / "fixed_roi.txt"
    roi_txt.write_text(
        f"roi_mode={args.roi_mode}\n"
        f"roi_margin={args.roi_margin}\n"
        f"x1,y1,x2,y2={roi[0]},{roi[1]},{roi[2]},{roi[3]}\n",
        encoding="utf-8",
    )

    predictor, resolved_model_cfg = load_sam2_predictor(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )

    prompt_specs = build_prompt_specs(
        grid_sizes=args.grid_sizes,
        include_box_only=not args.no_box_only,
        include_box_points=not args.no_box_points,
        include_points_only=not args.no_points_only,
    )

    print("[INFO] prompt specs:")
    for spec in prompt_specs:
        print(f"  - {spec['name']}")

    rows: List[Dict] = []

    with torch.inference_mode():
        for spec in prompt_specs:
            prompt_name = spec["name"]
            print(f"[PROMPT] {prompt_name}")

            overlay_dir = args.output_dir / prompt_name / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)

            for episode_dir in episodes:
                episode_id = episode_dir.name

                try:
                    image_bgr = load_preview_image(episode_dir, args.camera)
                    target_mask = load_target_mask(episode_dir)
                    if target_mask is None:
                        print(f"[WARN] Missing target mask: {episode_id}")
                        continue

                    points = None
                    if spec["use_points"]:
                        points = sample_grid_points(
                            roi=roi,
                            grid_size=int(spec["grid_size"]),
                            inner_margin=args.inner_margin,
                        )

                    pred_mask, sam2_score, all_masks, all_scores = predict_with_sam2(
                        predictor=predictor,
                        image_bgr=image_bgr,
                        roi=roi,
                        points=points,
                        use_box=bool(spec["use_box"]),
                        multimask_output=bool(args.multimask_output),
                    )

                    metrics = compute_metrics(pred_mask, target_mask)

                    # Oracle IoU among all SAM2 masks. This is only for diagnosis.
                    oracle_iou = metrics["iou"]
                    if all_masks is not None and len(all_masks) > 0:
                        oracle_ious = []
                        for m in all_masks:
                            m_uint8 = ((m > 0).astype(np.uint8) * 255)
                            oracle_ious.append(compute_metrics(m_uint8, target_mask)["iou"])
                        oracle_iou = float(max(oracle_ious)) if oracle_ious else metrics["iou"]

                    title = (
                        f"{episode_id} | {prompt_name} | "
                        f"IoU={metrics['iou']:.3f} Cov={metrics['coverage']:.3f}"
                    )
                    overlay = create_overlay(
                        image_bgr=image_bgr,
                        pred_mask=pred_mask,
                        target_mask=target_mask,
                        roi=roi,
                        points=points,
                        title=title,
                    )

                    cv2.imwrite(str(overlay_dir / f"{episode_id}.png"), overlay)

                    row = {
                        "episode": episode_id,
                        "prompt": prompt_name,
                        "coverage": metrics["coverage"],
                        "iou": metrics["iou"],
                        "oracle_iou": oracle_iou,
                        "sam2_score": sam2_score,
                        "intersection": metrics["intersection"],
                        "union": metrics["union"],
                        "pred_area": metrics["pred_area"],
                        "target_area": metrics["target_area"],
                        "roi_x1": roi[0],
                        "roi_y1": roi[1],
                        "roi_x2": roi[2],
                        "roi_y2": roi[3],
                    }
                    rows.append(row)

                except Exception as exc:
                    print(f"[ERROR] {prompt_name} | {episode_id}: {exc}")

    summary = summarize_rows(rows)

    episode_metrics_path = args.output_dir / "episode_metrics.csv"
    summary_path = args.output_dir / "summary.csv"

    write_csv(episode_metrics_path, rows)
    write_csv(summary_path, summary)

    print(f"[OK] Wrote {episode_metrics_path}")
    print(f"[OK] Wrote {summary_path}")

    print("[TOP]")
    for item in summary[:10]:
        print(
            f"{item['prompt']}: "
            f"coverage={item['coverage']:.4f}, "
            f"iou={item['iou']:.4f}, "
            f"oracle_iou={item['oracle_iou']:.4f}, "
            f"score={item['sam2_score']:.4f}, "
            f"episodes={item['episodes']}"
        )


if __name__ == "__main__":
    main()