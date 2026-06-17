#!/usr/bin/env python3
"""Debug whether SAM2 itself works on FIRM episode frames.

This script does NOT use OWLv2.
This script does NOT use fixed ROI from all episodes.

Instead, it uses each episode's existing target_mask.png as an oracle prompt source:
1) target bbox
2) target center point
3) 3x3 positive points inside target mask
4) 5x5 positive points inside target mask
5) bbox + center point
6) bbox + 3x3 points
7) bbox + 5x5 points

Then it runs SAM2 and compares the predicted mask with target_mask.png.

Interpretation:
- If oracle bbox / oracle points get reasonable IoU, SAM2 is basically working.
- If even oracle prompts get near-zero IoU, then SAM2 loading / image format / prompt coordinate / target mask definition may be wrong.

Example:
python firm_benchmark\\debug_sam2_basic.py ^
  --episodes-root annotation_workspace_sponge\\episodes ^
  --checkpoint sam2\\checkpoints\\sam2.1_hiera_tiny.pt ^
  --output-dir results_sam2_debug_basic ^
  --max-episodes 10 ^
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


HEAD_COLOR_CAMERA = "observation.images.head.color"


# ---------------------------------------------------------------------
# File / image utilities
# ---------------------------------------------------------------------


def list_episodes(episodes_root: Path) -> List[Path]:
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root does not exist: {episodes_root}")

    return sorted(
        p for p in episodes_root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )


def load_preview_image(episode_dir: Path, camera_key: str) -> np.ndarray:
    """Load final frame first; fallback to last sampled frame."""
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
        f"No usable image for {episode_dir.name} with camera `{camera_key}`"
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


def mask_center(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    cx = float(xs.mean())
    cy = float(ys.mean())
    return np.asarray([[cx, cy]], dtype=np.float32)


def nearest_foreground_point(mask: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Find nearest foreground pixel in mask to a given point."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    dist2 = dx * dx + dy * dy
    idx = int(np.argmin(dist2))

    return float(xs[idx]), float(ys[idx])


def sample_grid_points_from_mask(mask: np.ndarray, grid_size: int) -> Optional[np.ndarray]:
    """Sample grid points from target mask bbox.

    If a grid point is outside the foreground mask, it is snapped to the nearest
    foreground pixel.
    """
    bbox = mask_bbox(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # Avoid exact boundary points.
    margin_x = 0.18 * w
    margin_y = 0.18 * h

    if grid_size <= 1:
        xs = np.asarray([(x1 + x2) / 2.0], dtype=np.float32)
        ys = np.asarray([(y1 + y2) / 2.0], dtype=np.float32)
    else:
        xs = np.linspace(x1 + margin_x, x2 - margin_x, grid_size, dtype=np.float32)
        ys = np.linspace(y1 + margin_y, y2 - margin_y, grid_size, dtype=np.float32)

    points = []
    for yy in ys:
        for xx in xs:
            if mask[int(round(yy)), int(round(xx))] > 0:
                points.append([float(xx), float(yy)])
            else:
                nearest = nearest_foreground_point(mask, xx, yy)
                if nearest is not None:
                    points.append([nearest[0], nearest[1]])

    if not points:
        return None

    # Remove duplicates while keeping order.
    dedup = []
    seen = set()
    for x, y in points:
        key = (int(round(x)), int(round(y)))
        if key not in seen:
            seen.add(key)
            dedup.append([float(x), float(y)])

    return np.asarray(dedup, dtype=np.float32)


# ---------------------------------------------------------------------
# Metrics and visualization
# ---------------------------------------------------------------------


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
    points: Optional[np.ndarray],
    box: Optional[np.ndarray],
    title: str,
) -> np.ndarray:
    """Overlay predicted mask and target mask.

    Color:
    - target only: green
    - prediction only: red
    - overlap: yellow
    - prompt box: blue
    - positive points: cyan
    """
    overlay = image_bgr.copy()

    pred = pred_mask > 0
    target = target_mask > 0

    overlap = np.logical_and(pred, target)
    pred_only = np.logical_and(pred, ~target)
    target_only = np.logical_and(target, ~pred)

    alpha = 0.45

    # target only: green
    overlay[target_only] = (
        (1.0 - alpha) * overlay[target_only]
        + alpha * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    # pred only: red
    overlay[pred_only] = (
        (1.0 - alpha) * overlay[pred_only]
        + alpha * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)

    # overlap: yellow
    overlay[overlap] = (
        (1.0 - alpha) * overlay[overlap]
        + alpha * np.array([0, 255, 255], dtype=np.float32)
    ).astype(np.uint8)

    if box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if points is not None:
        for x, y in points:
            cv2.circle(
                overlay,
                (int(round(x)), int(round(y))),
                4,
                (255, 255, 0),
                -1,
            )

    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


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


def summarize(rows: List[Dict]) -> List[Dict]:
    by_prompt: Dict[str, List[Dict]] = {}

    for row in rows:
        by_prompt.setdefault(row["prompt"], []).append(row)

    summary = []

    for prompt, items in by_prompt.items():
        def mean(key: str) -> float:
            vals = [float(x[key]) for x in items]
            return float(np.mean(vals)) if vals else 0.0

        summary.append(
            {
                "prompt": prompt,
                "episodes": len(items),
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


# ---------------------------------------------------------------------
# SAM2 loading and prediction
# ---------------------------------------------------------------------


def infer_sam2_model_cfg(checkpoint: Path) -> str:
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
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise ImportError(
            "Failed to import SAM2. Please make sure sam2 is installed in this environment."
        ) from exc

    if model_cfg is None:
        model_cfg = infer_sam2_model_cfg(checkpoint)

    print("[INFO] Loading SAM2")
    print(f"[INFO] checkpoint = {checkpoint}")
    print(f"[INFO] model_cfg  = {model_cfg}")
    print(f"[INFO] device     = {device}")

    sam2_model = build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint),
        device=device,
    )

    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, model_cfg


def run_sam2_predict(
    predictor,
    image_bgr: np.ndarray,
    points: Optional[np.ndarray],
    box: Optional[np.ndarray],
    multimask_output: bool,
    image_format: str = "rgb",
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Run SAM2 predictor.

    Args:
        image_format:
            rgb: correct mode. Convert OpenCV BGR to RGB before SAM2.
            bgr: debug mode. Feed BGR directly to SAM2 to test color issue.

    Returns:
        selected_mask: selected by SAM2 score
        selected_score
        best_logit_dummy
        all_masks
        all_scores
    """
    if image_format == "rgb":
        image_input = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif image_format == "bgr":
        image_input = image_bgr.copy()
    else:
        raise ValueError(f"Unsupported image_format: {image_format}")

    predictor.set_image(image_input)

    point_coords = None
    point_labels = None

    if points is not None and len(points) > 0:
        point_coords = points.astype(np.float32)
        point_labels = np.ones((len(points),), dtype=np.int32)

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box.astype(np.float32) if box is not None else None,
        multimask_output=multimask_output,
    )

    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    if masks.ndim == 2:
        masks = masks[None, ...]

    if len(scores) == 0 or len(masks) == 0:
        raise RuntimeError("SAM2 returned no masks or scores.")

    best_idx = int(np.argmax(scores))
    selected = masks[best_idx]
    selected_mask = ((selected > 0).astype(np.uint8) * 255)
    selected_score = float(scores[best_idx])

    return selected_mask, selected_score, 0.0, masks, scores


def build_prompt_specs(target_mask: np.ndarray) -> List[Dict]:
    bbox = mask_bbox(target_mask)
    center = mask_center(target_mask)
    points_3 = sample_grid_points_from_mask(target_mask, grid_size=3)
    points_5 = sample_grid_points_from_mask(target_mask, grid_size=5)

    if bbox is None:
        return []

    x1, y1, x2, y2 = bbox
    box = np.asarray([x1, y1, x2, y2], dtype=np.float32)

    specs = []

    specs.append(
        {
            "name": "oracle_box_only",
            "box": box,
            "points": None,
        }
    )

    if center is not None:
        specs.append(
            {
                "name": "oracle_center_point",
                "box": None,
                "points": center,
            }
        )

        specs.append(
            {
                "name": "oracle_box_center_point",
                "box": box,
                "points": center,
            }
        )

    if points_3 is not None:
        specs.append(
            {
                "name": "oracle_points_3x3",
                "box": None,
                "points": points_3,
            }
        )

        specs.append(
            {
                "name": "oracle_box_points_3x3",
                "box": box,
                "points": points_3,
            }
        )

    if points_5 is not None:
        specs.append(
            {
                "name": "oracle_points_5x5",
                "box": None,
                "points": points_5,
            }
        )

        specs.append(
            {
                "name": "oracle_box_points_5x5",
                "box": box,
                "points": points_5,
            }
        )

    return specs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug SAM2 basic behavior.")

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
        help="SAM2 checkpoint path.",
    )

    parser.add_argument(
        "--model-cfg",
        default=None,
        help="SAM2 model config. If omitted, inferred from checkpoint filename.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--camera",
        default=HEAD_COLOR_CAMERA,
        help="Camera key.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start episode index.",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu.",
    )

    parser.add_argument(
        "--no-multimask",
        action="store_true",
        help="Disable SAM2 multimask output.",
    )

    parser.add_argument(
        "--include-bgr-test",
        action="store_true",
        help="Also test feeding BGR image directly to SAM2. This should usually be worse.",
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
    print(f"[INFO] output_dir    = {args.output_dir}")
    print(f"[INFO] camera        = {args.camera}")
    print(f"[INFO] selected eps  = {len(episodes)}")
    print(f"[INFO] first episode = {episodes[0].name}")
    print(f"[INFO] last episode  = {episodes[-1].name}")

    predictor, model_cfg = load_sam2_predictor(
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
    )

    image_formats = ["rgb"]
    if args.include_bgr_test:
        image_formats.append("bgr")

    rows: List[Dict] = []

    with torch.inference_mode():
        for episode_dir in episodes:
            episode_id = episode_dir.name

            try:
                image_bgr = load_preview_image(episode_dir, args.camera)
                target_mask = load_target_mask(episode_dir)

                if target_mask is None:
                    print(f"[WARN] Missing target_mask.png: {episode_id}")
                    continue

                prompt_specs = build_prompt_specs(target_mask)
                if not prompt_specs:
                    print(f"[WARN] No prompt specs: {episode_id}")
                    continue

                for image_format in image_formats:
                    for spec in prompt_specs:
                        prompt_name = spec["name"]
                        full_prompt_name = f"{image_format}_{prompt_name}"

                        try:
                            pred_mask, score, _, all_masks, all_scores = run_sam2_predict(
                                predictor=predictor,
                                image_bgr=image_bgr,
                                points=spec["points"],
                                box=spec["box"],
                                multimask_output=not args.no_multimask,
                                image_format=image_format,
                            )

                            metrics = compute_metrics(pred_mask, target_mask)

                            # Oracle-best IoU among SAM2 multimask outputs.
                            oracle_iou = metrics["iou"]
                            oracle_score = score

                            if all_masks is not None and len(all_masks) > 0:
                                best_iou = -1.0
                                best_score = 0.0

                                for m, s in zip(all_masks, all_scores):
                                    m_uint8 = ((m > 0).astype(np.uint8) * 255)
                                    cur = compute_metrics(m_uint8, target_mask)
                                    if cur["iou"] > best_iou:
                                        best_iou = cur["iou"]
                                        best_score = float(s)

                                oracle_iou = float(best_iou)
                                oracle_score = float(best_score)

                            title = (
                                f"{episode_id} | {full_prompt_name} | "
                                f"IoU={metrics['iou']:.3f} "
                                f"Oracle={oracle_iou:.3f} "
                                f"Score={score:.3f}"
                            )

                            overlay = create_overlay(
                                image_bgr=image_bgr,
                                pred_mask=pred_mask,
                                target_mask=target_mask,
                                points=spec["points"],
                                box=spec["box"],
                                title=title,
                            )

                            overlay_dir = args.output_dir / full_prompt_name / "overlays"
                            overlay_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(overlay_dir / f"{episode_id}.png"), overlay)

                            row = {
                                "episode": episode_id,
                                "prompt": full_prompt_name,
                                "coverage": metrics["coverage"],
                                "iou": metrics["iou"],
                                "oracle_iou": oracle_iou,
                                "sam2_score": score,
                                "oracle_score": oracle_score,
                                "intersection": metrics["intersection"],
                                "union": metrics["union"],
                                "pred_area": metrics["pred_area"],
                                "target_area": metrics["target_area"],
                            }
                            rows.append(row)

                        except Exception as exc:
                            print(f"[ERROR] {episode_id} | {full_prompt_name}: {exc}")

            except Exception as exc:
                print(f"[ERROR] {episode_id}: {exc}")

    summary_rows = summarize(rows)

    episode_metrics_path = args.output_dir / "episode_metrics.csv"
    summary_path = args.output_dir / "summary.csv"

    write_csv(episode_metrics_path, rows)
    write_csv(summary_path, summary_rows)

    print(f"[OK] Wrote {episode_metrics_path}")
    print(f"[OK] Wrote {summary_path}")

    print("[TOP]")
    for item in summary_rows[:20]:
        print(
            f"{item['prompt']}: "
            f"coverage={item['coverage']:.4f}, "
            f"iou={item['iou']:.4f}, "
            f"oracle_iou={item['oracle_iou']:.4f}, "
            f"score={item['sam2_score']:.4f}, "
            f"episodes={item['episodes']}"
        )

    print("")
    print("[INTERPRETATION]")
    if summary_rows:
        best_iou = max(float(x["iou"]) for x in summary_rows)
        best_oracle_iou = max(float(x["oracle_iou"]) for x in summary_rows)

        print(f"best selected IoU = {best_iou:.4f}")
        print(f"best oracle IoU   = {best_oracle_iou:.4f}")

        if best_oracle_iou >= 0.30:
            print(
                "SAM2 is probably working. If selected IoU is much lower than oracle IoU, "
                "the issue is mask selection / prompt strategy, not SAM2 loading."
            )
        elif best_oracle_iou >= 0.10:
            print(
                "SAM2 gives weak but non-zero masks. Check whether target_mask is a target region "
                "rather than the exact visible sponge mask."
            )
        else:
            print(
                "Even oracle prompts give very low IoU. Possible issues: wrong SAM2 config, "
                "wrong checkpoint, RGB/BGR problem, target_mask does not correspond to visible object, "
                "or SAM2 is segmenting a different object boundary."
            )


if __name__ == "__main__":
    main()