#!/usr/bin/env python3
"""Generate local SAM3 object masks for FIRM episodes.

This is the SAM3 replacement for the previous prompted SAM2 path.

Important:
- This script NEVER creates or edits `target_mask.png`.
- Each episode must already contain a trustworthy, human-provided target mask.
- Target-mask geometry is never used as a SAM3 prompt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEAD_COLOR_CAMERA = "observation.images.head.color"
DEFAULT_MODEL_DIR = Path.home() / "sam3"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "sam3.pt"
DEFAULT_SAM3_REPO = Path.home() / "cap-x" / "capx" / "third_party" / "sam3"
DEFAULT_TEXT_PROMPT = "white paper"


def require_target_mask(episode_dir: Path) -> None:
    """Require a real target mask for downstream metric extraction."""
    target_path = episode_dir / "target_mask.png"
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(
            f"Missing or unreadable target mask: {target_path}. "
            "Provide real target masks before running SAM3 generation."
        )
    target = ((target > 127).astype(np.uint8) * 255)
    if int((target > 0).sum()) == 0:
        raise ValueError(f"Empty target mask: {target_path}")


def load_target_mask_binary(episode_dir: Path) -> np.ndarray:
    """Load target_mask.png as a binary uint8 mask in {0,255}."""
    target_path = episode_dir / "target_mask.png"
    target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(f"Missing or unreadable target mask: {target_path}")
    return ((target > 127).astype(np.uint8) * 255)


def object_box_from_norm(
    image_shape: tuple[int, int],
    object_box_norm: tuple[float, float, float, float],
) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] object ROI to xyxy pixels."""
    height, width = image_shape
    cx, cy, box_w, box_h = object_box_norm
    x0 = max(0.0, (cx - box_w / 2.0) * width)
    y0 = max(0.0, (cy - box_h / 2.0) * height)
    x1 = min(float(width - 1), (cx + box_w / 2.0) * width)
    y1 = min(float(height - 1), (cy + box_h / 2.0) * height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid object box after scaling: {(x0, y0, x1, y1)}")
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def choose_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    object_roi: np.ndarray | None,
    selection_mask: np.ndarray | None,
) -> np.ndarray:
    """Select a text-prompt candidate by score, optionally biased toward an ROI."""
    if masks is None or len(masks) == 0:
        return np.zeros(image_shape, dtype=np.uint8)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    best_idx = 0
    best_score = -1e9
    for idx, mask in enumerate(masks):
        candidate = np.asarray(mask) > 0
        area = float(candidate.sum())
        if area <= 0.0:
            continue
        model_score = float(scores[idx]) if scores is not None and len(scores) > idx else 0.0
        score = model_score
        if selection_mask is not None:
            selection = selection_mask > 0
            in_selection = float(np.logical_and(candidate, selection).sum())
            selection_area = max(float(selection.sum()), 1.0)
            selection_coverage = in_selection / selection_area
            mask_in_selection_ratio = in_selection / area
            area_ratio = area / selection_area
            area_penalty = max(0.0, area_ratio - 1.5)
            score = 2.5 * mask_in_selection_ratio + 0.75 * selection_coverage - area_penalty + 0.1 * model_score
        elif object_roi is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in object_roi]
            x0 = max(0, min(x0, image_shape[1] - 1))
            x1 = max(0, min(x1, image_shape[1] - 1))
            y0 = max(0, min(y0, image_shape[0] - 1))
            y1 = max(0, min(y1, image_shape[0] - 1))
            in_box = float(candidate[y0 : y1 + 1, x0 : x1 + 1].sum())
            box_area = max(float((y1 - y0 + 1) * (x1 - x0 + 1)), 1.0)
            box_coverage = in_box / box_area
            mask_in_box_ratio = in_box / area
            area_ratio = area / box_area
            area_penalty = max(0.0, area_ratio - 1.5)
            score = 2.0 * mask_in_box_ratio + 0.5 * box_coverage - area_penalty + 0.1 * model_score
        if score > best_score:
            best_score = score
            best_idx = idx

    return (np.asarray(masks[best_idx]) > 0).astype(np.uint8) * 255


def union_text_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple[int, int],
    min_mask_score: float,
) -> np.ndarray:
    """Union text-prompt masks whose SAM3 score passes the threshold."""
    if masks is None or len(masks) == 0:
        return np.zeros(image_shape, dtype=np.uint8)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    union = np.zeros(image_shape, dtype=bool)
    used = 0
    for idx, mask in enumerate(masks):
        score = float(scores[idx]) if scores is not None and len(scores) > idx else 1.0
        if score < min_mask_score:
            continue
        union |= np.asarray(mask) > 0
        used += 1

    if used == 0:
        return (np.asarray(masks[0]) > 0).astype(np.uint8) * 255
    return union.astype(np.uint8) * 255


def contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    if area <= 1.0:
        return 0.0
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def refine_ring_mask(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Keep the connected component that most resembles a tape-like ring."""
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_mask = binary
    best_score = -1e9

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        component = (labels == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            continue

        hierarchy = hierarchy[0]
        outer_indices = [idx for idx, h in enumerate(hierarchy) if h[3] == -1]
        if not outer_indices:
            continue

        outer_idx = max(outer_indices, key=lambda idx: cv2.contourArea(contours[idx]))
        outer = contours[outer_idx]
        outer_area = float(cv2.contourArea(outer))
        if outer_area <= 1.0:
            continue

        hole_area = 0.0
        child_idx = hierarchy[outer_idx][2]
        while child_idx != -1:
            hole_area += float(cv2.contourArea(contours[child_idx]))
            child_idx = hierarchy[child_idx][0]

        hole_ratio = hole_area / max(outer_area, 1.0)
        circularity = contour_circularity(outer)
        fill_ratio = area / max(outer_area, 1.0)
        bbox_w = max(int(stats[label, cv2.CC_STAT_WIDTH]), 1)
        bbox_h = max(int(stats[label, cv2.CC_STAT_HEIGHT]), 1)
        aspect_ratio = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)

        ring_score = (
            3.0 * min(hole_ratio, 0.45)
            + 1.5 * circularity
            + 1.0 * aspect_ratio
            - 2.0 * abs(fill_ratio - 0.75)
        )
        if hole_area <= 1.0:
            ring_score -= 2.0

        if ring_score > best_score:
            best_score = ring_score
            best_mask = component

    return (best_mask > 0).astype(np.uint8) * 255


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def build_sam3_model(checkpoint: Path, sam3_repo: Path, device: str):
    """Load local SAM3 from the CaP-X vendored repo and local checkpoint."""
    if sam3_repo.exists():
        sys.path.insert(0, str(sam3_repo))

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise ImportError(
            "Could not import local SAM3. Activate the CaP-X environment and/or pass "
            f"--sam3-repo {DEFAULT_SAM3_REPO}."
        ) from exc

    if "cuda" in device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.set_device(device_idx)

    logger.info("Loading local SAM3 checkpoint from %s", checkpoint)
    model = build_sam3_image_model(
        enable_inst_interactivity=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
    )
    model = model.to(device).eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
    return model, processor


def generate_mask_for_frame(
    model,
    processor,
    frame_path: Path,
    text_prompt: str,
    object_box_norm: tuple[float, float, float, float],
    use_roi_selection: bool,
    clip_to_object_box: bool,
    selection_mask: np.ndarray | None,
    clip_to_selection_mask: bool,
    apply_ring_filter: bool,
    min_ring_component_area: int,
    union_masks: bool,
    min_mask_score: float,
    device: str,
) -> np.ndarray | None:
    """Generate a prompted mask for a single frame."""
    image = cv2.imread(str(frame_path))
    if image is None:
        logger.warning("Cannot read image: %s", frame_path)
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    object_roi = object_box_from_norm(image.shape[:2], object_box_norm) if use_roi_selection or clip_to_object_box else None

    from PIL import Image

    pil_image = Image.fromarray(image_rgb)
    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_enabled = device_type == "cuda"
    with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None or len(masks) == 0:
        logger.warning("No masks generated for %s with prompt %r", frame_path, text_prompt)
        return None

    masks_np = _to_numpy(masks)
    scores_np = _to_numpy(scores) if scores is not None else np.zeros(len(masks_np), dtype=np.float32)
    if union_masks:
        best_mask = union_text_masks(masks_np, scores_np, image.shape[:2], min_mask_score)
    else:
        best_mask = choose_best_mask(
            masks_np,
            scores_np,
            image.shape[:2],
            object_roi,
            selection_mask,
        )

    if clip_to_object_box:
        clipped = np.zeros_like(best_mask)
        x0, y0, x1, y1 = [int(round(v)) for v in object_roi]
        clipped[y0 : y1 + 1, x0 : x1 + 1] = best_mask[y0 : y1 + 1, x0 : x1 + 1]
        best_mask = clipped

    if clip_to_selection_mask and selection_mask is not None:
        best_mask = np.where(selection_mask > 0, best_mask, 0).astype(np.uint8)

    if apply_ring_filter:
        best_mask = refine_ring_mask(best_mask, min_component_area=min_ring_component_area)

    return best_mask


def process_episode(
    episode_dir: Path,
    model,
    processor,
    camera_key: str,
    text_prompt: str,
    object_box_norm: tuple[float, float, float, float],
    use_roi_selection: bool,
    clip_to_object_box: bool,
    use_target_mask_selection: bool,
    clip_to_target_mask: bool,
    apply_ring_filter: bool,
    min_ring_component_area: int,
    union_masks: bool,
    min_mask_score: float,
    device: str,
) -> int:
    """Generate masks for all sampled frames plus the final frame."""
    sampled_dir = episode_dir / "sampled_frames" / camera_key
    if not sampled_dir.exists():
        logger.warning("%s: no sampled frames for %s", episode_dir.name, camera_key)
        return 0

    frame_files = sorted(sampled_dir.glob("*.png"))
    if not frame_files:
        logger.warning("%s: no frames found in %s", episode_dir.name, sampled_dir)
        return 0

    require_target_mask(episode_dir)
    target_mask = load_target_mask_binary(episode_dir) if (use_target_mask_selection or clip_to_target_mask) else None

    output_dir = episode_dir / "masks" / "object"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_count = 0
    for index, frame_path in enumerate(frame_files):
        mask = generate_mask_for_frame(
            model,
            processor,
            frame_path,
            text_prompt,
            object_box_norm,
            use_roi_selection,
            clip_to_object_box,
            target_mask,
            clip_to_target_mask,
            apply_ring_filter,
            min_ring_component_area,
            union_masks,
            min_mask_score,
            device,
        )
        if mask is None:
            continue
        cv2.imwrite(str(output_dir / f"{index:06d}.png"), mask)
        mask_count += 1

    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        mask = generate_mask_for_frame(
            model,
            processor,
            final_frame,
            text_prompt,
            object_box_norm,
            use_roi_selection,
            clip_to_object_box,
            target_mask,
            clip_to_target_mask,
            apply_ring_filter,
            min_ring_component_area,
            union_masks,
            min_mask_score,
            device,
        )
        if mask is not None:
            cv2.imwrite(str(output_dir / f"{len(frame_files):06d}.png"), mask)
            mask_count += 1

    logger.info("%s: generated %d masks", episode_dir.name, mask_count)
    return mask_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompted local SAM3 masks for FIRM episodes")
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam3-repo", type=Path, default=DEFAULT_SAM3_REPO)
    parser.add_argument(
        "--camera",
        default=HEAD_COLOR_CAMERA,
        help="Camera key to use. Only observation.images.head.color is allowed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--text-prompt",
        default=DEFAULT_TEXT_PROMPT,
        help="SAM3 text prompt used to segment the object. Default: white paper",
    )
    parser.add_argument(
        "--object-box-norm",
        nargs=4,
        type=float,
        metavar=("CX", "CY", "W", "H"),
        default=(0.5, 0.5, 0.4, 0.4),
        help=(
            "Optional normalized object ROI for candidate selection/clipping, in [cx cy w h]. "
            "Default: 0.5 0.5 0.4 0.4"
        ),
    )
    parser.add_argument(
        "--use-roi-selection",
        action="store_true",
        help=(
            "Bias selection among SAM3 text-prompt candidates toward --object-box-norm. "
            "The ROI is not used as a SAM3 prompt."
        ),
    )
    parser.add_argument(
        "--clip-to-object-box",
        action="store_true",
        help="Zero out predicted mask pixels outside the object ROI.",
    )
    parser.add_argument(
        "--use-target-mask-selection",
        action="store_true",
        help=(
            "Bias selection among SAM3 text-prompt candidates toward the existing "
            "target_mask.png region. The target mask is not used as a SAM3 prompt."
        ),
    )
    parser.add_argument(
        "--clip-to-target-mask",
        action="store_true",
        help="Zero out predicted mask pixels outside target_mask.png.",
    )
    parser.add_argument(
        "--apply-ring-filter",
        action="store_true",
        help="Post-process the selected mask and keep the most tape-ring-like component.",
    )
    parser.add_argument(
        "--min-ring-component-area",
        type=int,
        default=80,
        help="Minimum connected-component area kept by --apply-ring-filter. Default: 80",
    )
    parser.add_argument(
        "--union-text-masks",
        action="store_true",
        help="Union all SAM3 text-prompt masks above --min-mask-score.",
    )
    parser.add_argument(
        "--min-mask-score",
        type=float,
        default=0.0,
        help="Minimum SAM3 text mask score used with --union-text-masks.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {args.checkpoint}")
    if not args.sam3_repo.exists():
        raise FileNotFoundError(f"Local SAM3 repo not found: {args.sam3_repo}")
    if args.camera != HEAD_COLOR_CAMERA:
        raise ValueError(
            f"Only `{HEAD_COLOR_CAMERA}` is supported for detection. "
            f"Got `{args.camera}`."
        )
    if any(value <= 0.0 or value > 1.0 for value in args.object_box_norm):
        raise ValueError("--object-box-norm values must be in (0, 1]")
    if not args.text_prompt.strip():
        raise ValueError("--text-prompt must be non-empty")

    model, processor = build_sam3_model(args.checkpoint, args.sam3_repo, args.device)

    episode_dirs = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    logger.info("Found %d episodes to process", len(episode_dirs))

    total_masks = 0
    for idx, episode_dir in enumerate(episode_dirs, start=1):
        logger.info("[%d/%d] Processing %s", idx, len(episode_dirs), episode_dir.name)
        total_masks += process_episode(
            episode_dir,
            model,
            processor,
            args.camera,
            args.text_prompt,
            tuple(args.object_box_norm),
            args.use_roi_selection,
            args.clip_to_object_box,
            args.use_target_mask_selection,
            args.clip_to_target_mask,
            args.apply_ring_filter,
            args.min_ring_component_area,
            args.union_text_masks,
            args.min_mask_score,
            args.device,
        )

    logger.info("Completed. Generated %d masks across %d episodes", total_masks, len(episode_dirs))


if __name__ == "__main__":
    main()
