#!/usr/bin/env python3
"""Generate SAM2 object masks for FIRM episodes.

Important:
- This script NEVER creates or edits `target_mask.png`.
- Each episode must already contain a trustworthy, human-provided target mask.
- Target-mask geometry is never used as a SAM2 prompt or candidate selector.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
HEAD_COLOR_CAMERA = "observation.images.head.color"


def require_target_mask(episode_dir: Path) -> None:
    """Require a real target mask for downstream metric extraction."""
    target_path = episode_dir / "target_mask.png"
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(
            f"Missing or unreadable target mask: {target_path}. "
            "Provide real target masks before running SAM2 generation."
        )
    target = ((target > 127).astype(np.uint8) * 255)
    if int((target > 0).sum()) == 0:
        raise ValueError(f"Empty target mask: {target_path}")


def object_box_from_norm(
    image_shape: tuple[int, int],
    object_box_norm: tuple[float, float, float, float],
) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] object ROI to SAM2 xyxy pixels."""
    height, width = image_shape
    cx, cy, box_w, box_h = object_box_norm
    x0 = max(0.0, (cx - box_w / 2.0) * width)
    y0 = max(0.0, (cy - box_h / 2.0) * height)
    x1 = min(float(width - 1), (cx + box_w / 2.0) * width)
    y1 = min(float(height - 1), (cy + box_h / 2.0) * height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid object box after scaling: {(x0, y0, x1, y1)}")
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def build_sam2_predictor(checkpoint_path: Path, config_path: str, device: str):
    """Build a prompted SAM2 predictor."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "SAM2 import failed. Install SAM2 in the selected Python environment."
        ) from exc

    logger.info("Loading SAM2 checkpoint from %s", checkpoint_path)
    model = build_sam2(
        config_file=config_path,
        ckpt_path=str(checkpoint_path),
        device=device,
        apply_postprocessing=False,
    )
    return SAM2ImagePredictor(model)


def choose_best_mask(
    masks: np.ndarray,
    iou_scores: np.ndarray,
    image_shape: tuple[int, int],
    box_prompt: np.ndarray,
) -> np.ndarray:
    """Select the SAM2 candidate that best agrees with the object prompt box."""
    if masks is None or len(masks) == 0:
        return np.zeros(image_shape, dtype=np.uint8)

    x0, y0, x1, y1 = [int(round(v)) for v in box_prompt]
    x0 = max(0, min(x0, image_shape[1] - 1))
    x1 = max(0, min(x1, image_shape[1] - 1))
    y0 = max(0, min(y0, image_shape[0] - 1))
    y1 = max(0, min(y1, image_shape[0] - 1))

    best_idx = 0
    best_score = -1e9
    for idx, mask in enumerate(masks):
        candidate = mask > 0
        area = float(candidate.sum())
        if area <= 0.0:
            continue
        in_box = float(candidate[y0 : y1 + 1, x0 : x1 + 1].sum())
        box_area = max(float((y1 - y0 + 1) * (x1 - x0 + 1)), 1.0)
        box_coverage = in_box / box_area
        mask_in_box_ratio = in_box / area
        area_ratio = area / box_area
        area_penalty = max(0.0, area_ratio - 1.5)
        score = 2.0 * mask_in_box_ratio + 0.5 * box_coverage - area_penalty + 0.1 * float(iou_scores[idx])
        if score > best_score:
            best_score = score
            best_idx = idx

    return (masks[best_idx] > 0).astype(np.uint8) * 255


def generate_mask_for_frame(
    predictor,
    frame_path: Path,
    object_box_norm: tuple[float, float, float, float],
    object_point_norm: tuple[float, float] | None,
    clip_to_object_box: bool,
    use_cuda: bool,
) -> np.ndarray | None:
    """Generate a prompted mask for a single frame."""
    import torch

    image = cv2.imread(str(frame_path))
    if image is None:
        logger.warning("Cannot read image: %s", frame_path)
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    box_prompt = object_box_from_norm(image.shape[:2], object_box_norm)
    point_coords = None
    point_labels = None
    if object_point_norm is not None:
        px = object_point_norm[0] * image.shape[1]
        py = object_point_norm[1] * image.shape[0]
        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

    with torch.inference_mode():
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image_rgb)
                masks, iou_scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_prompt,
                    multimask_output=True,
                    return_logits=False,
                    normalize_coords=False,
                )
        else:
            predictor.set_image(image_rgb)
            masks, iou_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_prompt,
                multimask_output=True,
                return_logits=False,
                normalize_coords=False,
            )

    best_mask = choose_best_mask(masks, iou_scores, image.shape[:2], box_prompt)
    if clip_to_object_box:
        clipped = np.zeros_like(best_mask)
        x0, y0, x1, y1 = [int(round(v)) for v in box_prompt]
        clipped[y0 : y1 + 1, x0 : x1 + 1] = best_mask[y0 : y1 + 1, x0 : x1 + 1]
        best_mask = clipped
    return best_mask


def process_episode(
    episode_dir: Path,
    predictor,
    camera_key: str,
    object_box_norm: tuple[float, float, float, float],
    object_point_norm: tuple[float, float] | None,
    clip_to_object_box: bool,
    use_cuda: bool,
) -> int:
    """Generate masks for all sampled frames plus the final frame."""
    sampled_dir = episode_dir / "sampled_frames" / camera_key
    if not sampled_dir.exists():
        logger.warning("%s: no sampled frames for %s", episode_dir.name, camera_key)
        return 0

    frame_files = sorted(sampled_dir.glob("*.png"))
    if not frame_files:
        logger.warning("%s: no frames found", episode_dir.name)
        return 0

    require_target_mask(episode_dir)

    output_dir = episode_dir / "masks" / "object"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_count = 0
    for index, frame_path in enumerate(frame_files):
        mask = generate_mask_for_frame(
            predictor,
            frame_path,
            object_box_norm,
            object_point_norm,
            clip_to_object_box,
            use_cuda,
        )
        if mask is None:
            continue
        cv2.imwrite(str(output_dir / f"{index:06d}.png"), mask)
        mask_count += 1

    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        mask = generate_mask_for_frame(
            predictor,
            final_frame,
            object_box_norm,
            object_point_norm,
            clip_to_object_box,
            use_cuda,
        )
        if mask is not None:
            cv2.imwrite(str(output_dir / f"{len(frame_files):06d}.png"), mask)
            mask_count += 1

    logger.info("%s: generated %d masks", episode_dir.name, mask_count)
    return mask_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompted SAM2 masks for FIRM episodes")
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 config path relative to the SAM2 repository",
    )
    parser.add_argument(
        "--camera",
        default=HEAD_COLOR_CAMERA,
        help="Camera key to use. Only observation.images.head.color is allowed.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--object-box-norm",
        nargs=4,
        type=float,
        metavar=("CX", "CY", "W", "H"),
        default=(0.5, 0.5, 0.4, 0.4),
        help=(
            "Normalized object ROI used as the SAM2 box prompt, in [cx cy w h]. "
            "This is independent of target_mask.png. Default: 0.5 0.5 0.4 0.4"
        ),
    )
    parser.add_argument(
        "--object-point-norm",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help=(
            "Optional normalized positive point on the object, in [x y]. "
            "This is independent of target_mask.png."
        ),
    )
    parser.add_argument(
        "--clip-to-object-box",
        action="store_true",
        help="Zero out predicted mask pixels outside the object prompt box.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.checkpoint}")
    if args.camera != HEAD_COLOR_CAMERA:
        raise ValueError(
            f"Only `{HEAD_COLOR_CAMERA}` is supported for detection. "
            f"Got `{args.camera}`."
        )

    use_cuda = args.device == "cuda"
    import torch

    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if any(value <= 0.0 or value > 1.0 for value in args.object_box_norm):
        raise ValueError("--object-box-norm values must be in (0, 1]")
    if args.object_point_norm is not None and any(
        value < 0.0 or value > 1.0 for value in args.object_point_norm
    ):
        raise ValueError("--object-point-norm values must be in [0, 1]")

    predictor = build_sam2_predictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    episode_dirs = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    total_masks = 0
    for idx, episode_dir in enumerate(episode_dirs, start=1):
        logger.info("[%d/%d] Processing %s", idx, len(episode_dirs), episode_dir.name)
        total_masks += process_episode(
            episode_dir,
            predictor,
            args.camera,
            tuple(args.object_box_norm),
            tuple(args.object_point_norm) if args.object_point_norm is not None else None,
            args.clip_to_object_box,
            use_cuda,
        )

    logger.info("Completed. Generated %d masks across %d episodes", total_masks, len(episode_dirs))


if __name__ == "__main__":
    main()
