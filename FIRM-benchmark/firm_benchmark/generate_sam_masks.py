#!/usr/bin/env python3
"""
Generate SAM object masks for FIRM episodes.

This script uses Segment Anything Model (SAM) to segment objects in episode frames.

Requirements:
    pip install segment-anything opencv-python torch torchvision

Usage:
    python generate_sam_masks.py \
        --episodes-root annotation_workspace_sponge/episodes \
        --camera observation.images.hand.right.color \
        --checkpoint sam_vit_h_4b8939.pth

Output:
    episodes/episode_XXXXXX/masks/object/NNNNNN.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


def load_sam_model(checkpoint_path: Path, model_type: str = "vit_h", device: str = "cuda"):
    """Load SAM model."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "segment-anything not installed. Install with: pip install segment-anything"
        )

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def find_largest_mask(masks: List[dict]) -> np.ndarray:
    """Find the largest mask (assumed to be the object of interest)."""
    if not masks:
        return np.zeros((480, 640), dtype=np.uint8)

    largest = max(masks, key=lambda x: x["area"])
    return largest["segmentation"].astype(np.uint8) * 255


def process_episode(
    episode_dir: Path,
    mask_generator,
    camera_key: str,
    sample_interval: int = 10,
) -> None:
    """Process one episode and generate masks."""

    # Find sampled frames
    sampled_dir = episode_dir / "sampled_frames" / camera_key
    if not sampled_dir.exists():
        print(f"[SKIP] {episode_dir.name}: no sampled frames for {camera_key}")
        return

    frame_files = sorted(sampled_dir.glob("*.png"))
    if not frame_files:
        print(f"[SKIP] {episode_dir.name}: no frames found")
        return

    # Sample frames
    sampled_frames = frame_files[::sample_interval]
    if not sampled_frames:
        sampled_frames = [frame_files[-1]]

    # Add final frame
    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        sampled_frames.append(final_frame)

    # Create output directory
    output_dir = episode_dir / "masks" / "object"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each frame
    for i, frame_path in enumerate(sampled_frames):
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"[WARN] Cannot read {frame_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate masks
        masks = mask_generator.generate(image_rgb)

        # Find largest mask (object)
        object_mask = find_largest_mask(masks)

        # Save mask
        output_path = output_dir / f"{i:06d}.png"
        cv2.imwrite(str(output_path), object_mask)

    print(f"[OK] {episode_dir.name}: generated {len(sampled_frames)} masks")


def create_dummy_target_mask(episode_dir: Path, size: tuple = (480, 640)) -> None:
    """Create a dummy target mask (center region)."""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create a rectangular target region in the center
    cy, cx = h // 2, w // 2
    rh, rw = h // 3, w // 3
    mask[cy - rh:cy + rh, cx - rw:cx + rw] = 255

    output_path = episode_dir / "target_mask.png"
    cv2.imwrite(str(output_path), mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAM masks for FIRM episodes")
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="SAM checkpoint file")
    parser.add_argument("--camera", default="observation.images.hand.right.color", help="Camera key to use")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--sample-interval", type=int, default=10, help="Sample every N frames")
    parser.add_argument("--create-dummy-target", action="store_true", help="Create dummy target masks")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.checkpoint}")

    print(f"[INFO] Loading SAM model from {args.checkpoint}")
    mask_generator = load_sam_model(args.checkpoint, args.model_type, args.device)

    episode_dirs = sorted([d for d in args.episodes_root.iterdir() if d.is_dir()])
    print(f"[INFO] Found {len(episode_dirs)} episodes")

    for episode_dir in episode_dirs:
        process_episode(episode_dir, mask_generator, args.camera, args.sample_interval)

        if args.create_dummy_target:
            create_dummy_target_mask(episode_dir)

    print(f"[OK] Processed {len(episode_dirs)} episodes")


if __name__ == "__main__":
    main()
