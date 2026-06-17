#!/usr/bin/env python3
"""Generate object/target masks with English text prompts + SAM2.

Pipeline:
1) Open-vocabulary detector (text -> box)
2) SAM2 (box -> mask)

Object mask text prompt example: "sponge pad"
Target mask text prompt example: "box"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    buf.tofile(str(path))


class TextBoxDetector:
    """Text-guided detector based on OWLv2."""

    def __init__(self, model_name: str, device: str, box_threshold: float):
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise ImportError(
                "transformers is required for text-guided detection. "
                "Install with: pip install transformers"
            ) from exc

        self.device = torch.device(device)
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.box_threshold = float(box_threshold)

    @torch.inference_mode()
    def detect_best_box(self, image_rgb: np.ndarray, text_prompt: str) -> Optional[np.ndarray]:
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(text=[[text_prompt]], images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)  # (h, w)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.box_threshold,
        )[0]

        if len(results["scores"]) == 0:
            return None

        best_idx = int(torch.argmax(results["scores"]).item())
        box = results["boxes"][best_idx].detach().cpu().numpy().astype(np.float32)  # xyxy
        return box


def build_sam2_predictor(checkpoint_path: Path, config_path: str, device: str):
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


def choose_best_sam2_mask(masks: np.ndarray, iou_scores: np.ndarray) -> np.ndarray:
    if masks is None or len(masks) == 0:
        return np.zeros((480, 640), dtype=np.uint8)
    best = int(np.argmax(iou_scores))
    return (masks[best] > 0).astype(np.uint8) * 255


def segment_with_box(
    predictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    use_cuda: bool,
) -> np.ndarray:
    with torch.inference_mode():
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image_rgb)
                masks, iou_scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_xyxy,
                    multimask_output=True,
                    return_logits=False,
                    normalize_coords=False,
                )
        else:
            predictor.set_image(image_rgb)
            masks, iou_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy,
                multimask_output=True,
                return_logits=False,
                normalize_coords=False,
            )
    return choose_best_sam2_mask(masks, iou_scores)


def load_frames(episode_dir: Path, camera_key: str) -> list[Path]:
    sampled_dir = episode_dir / "sampled_frames" / camera_key
    frames = sorted(sampled_dir.glob("*.png")) if sampled_dir.exists() else []
    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        frames.append(final_frame)
    return frames


def generate_episode_masks(
    episode_dir: Path,
    predictor,
    detector: TextBoxDetector,
    object_text: str,
    target_text: str,
    camera_key: str,
    use_cuda: bool,
) -> tuple[int, bool]:
    frame_paths = load_frames(episode_dir, camera_key)
    if not frame_paths:
        logger.warning("%s: no frames for %s", episode_dir.name, camera_key)
        return 0, False

    # Build target mask from final frame and target text prompt.
    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if not final_frame.exists():
        logger.warning("%s: missing final frame, skip target mask", episode_dir.name)
        return 0, False

    final_img = imread_unicode(final_frame, cv2.IMREAD_COLOR)
    if final_img is None:
        logger.warning("%s: cannot read final frame", episode_dir.name)
        return 0, False
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    target_box = detector.detect_best_box(final_rgb, target_text)
    if target_box is None:
        logger.warning("%s: no target box detected for text '%s'", episode_dir.name, target_text)
        return 0, False

    target_mask = segment_with_box(predictor, final_rgb, target_box, use_cuda)
    imwrite_unicode(episode_dir / "target_mask.png", target_mask)

    output_dir = episode_dir / "masks" / "object"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_count = 0
    for idx, frame_path in enumerate(frame_paths):
        image = imread_unicode(frame_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        object_box = detector.detect_best_box(image_rgb, object_text)
        if object_box is None:
            continue
        object_mask = segment_with_box(predictor, image_rgb, object_box, use_cuda)
        imwrite_unicode(output_dir / f"{idx:06d}.png", object_mask)
        mask_count += 1

    logger.info("%s: generated %d object masks", episode_dir.name, mask_count)
    return mask_count, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text-guided SAM2 object/target masks")
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 config path relative to the SAM2 repository",
    )
    parser.add_argument(
        "--camera",
        default="observation.images.head.color",
        help="Camera key",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--object-text", default="sponge pad", help="English text for object")
    parser.add_argument("--target-text", default="box", help="English text for target region")
    parser.add_argument(
        "--detector-model",
        default="google/owlv2-base-patch16-ensemble",
        help="HuggingFace detector model id",
    )
    parser.add_argument("--detector-threshold", type=float, default=0.10)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    predictor = build_sam2_predictor(args.checkpoint, args.config, args.device)
    detector = TextBoxDetector(args.detector_model, args.device, args.detector_threshold)

    episode_dirs = sorted([p for p in args.episodes_root.iterdir() if p.is_dir()])
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    total_masks = 0
    total_target_ok = 0
    for i, episode_dir in enumerate(episode_dirs, start=1):
        logger.info("[%d/%d] %s", i, len(episode_dirs), episode_dir.name)
        count, target_ok = generate_episode_masks(
            episode_dir=episode_dir,
            predictor=predictor,
            detector=detector,
            object_text=args.object_text,
            target_text=args.target_text,
            camera_key=args.camera,
            use_cuda=(args.device == "cuda"),
        )
        total_masks += count
        total_target_ok += int(target_ok)

    logger.info(
        "Done. episodes=%d, target_mask_ok=%d, object_masks=%d",
        len(episode_dirs),
        total_target_ok,
        total_masks,
    )


if __name__ == "__main__":
    main()

