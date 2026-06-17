#!/usr/bin/env python3
"""Compare text prompts for OWLv2-box + SAM2 object segmentation.

This script is intentionally read-only for the annotation workspace:
- It reads final frames and existing target_mask.png files.
- It writes candidate object masks, overlays, and summary files under output-dir.
- It never modifies episode target masks or existing masks/object files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_PROMPTS = [
    # "sponge",
    # "sponge pad",
    "white paper"
    # "rectangular sponge",
    # "foam pad",
    # "white foam pad",
    # "sponge in the box",
    # "white rectangular foam",
    # "white rectangle"
]


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "prompt"


def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    buf.tofile(str(path))


def load_binary_mask(path: Path) -> np.ndarray:
    mask = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def mask_area(mask: np.ndarray) -> float:
    return float((mask > 0).sum())


def mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def normalized_centroid_error(obj_mask: np.ndarray, target_mask: np.ndarray) -> float:
    obj_c = mask_centroid(obj_mask)
    tgt_c = mask_centroid(target_mask)
    if obj_c is None or tgt_c is None:
        return 1e9
    h, w = obj_mask.shape[:2]
    diag = float(np.hypot(h, w))
    return float(np.linalg.norm(obj_c - tgt_c) / max(diag, 1e-8))


def compute_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict:
    pred = pred_mask > 0
    target = target_mask > 0
    overlap = float(np.logical_and(pred, target).sum())
    pred_area = float(pred.sum())
    target_area = float(target.sum())
    union = float(np.logical_or(pred, target).sum())
    return {
        "object_area": pred_area,
        "target_area": target_area,
        "overlap_area": overlap,
        "target_region_coverage": overlap / target_area if target_area > 0 else 0.0,
        "object_target_overlap_ratio": overlap / pred_area if pred_area > 0 else 0.0,
        "iou": overlap / union if union > 0 else 0.0,
        "pose_error": normalized_centroid_error(pred.astype(np.uint8), target.astype(np.uint8)),
    }


def overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = base_bgr.copy()
    idx = mask > 0
    if np.any(idx):
        out[idx] = (0.55 * out[idx] + 0.45 * np.array(color_bgr)).astype(np.uint8)
    return out


def make_overlay(image_bgr: np.ndarray, target: np.ndarray, pred: np.ndarray, label: str, metrics: dict) -> np.ndarray:
    target_panel = overlay_mask(image_bgr, target, (0, 255, 0))
    pred_panel = overlay_mask(image_bgr, pred, (255, 0, 0))
    both = overlay_mask(image_bgr, target, (0, 180, 0))
    both = overlay_mask(both, pred, (180, 0, 0))
    both = overlay_mask(both, np.logical_and(target > 0, pred > 0).astype(np.uint8), (0, 255, 255))

    panels = [image_bgr.copy(), target_panel, pred_panel, both]
    titles = ["Final Frame", "Target (Green)", "Pred (Blue)", "Overlap (Yellow)"]
    for panel, title in zip(panels, titles):
        cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    canvas = np.vstack([top, bottom])

    lines = [
        label,
        f"coverage={metrics['target_region_coverage']:.4f}  iou={metrics['iou']:.4f}",
        f"obj_overlap={metrics['object_target_overlap_ratio']:.4f}  pose={metrics['pose_error']:.4f}",
    ]
    y = 40
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34
    return canvas


class TextBoxDetector:
    def __init__(self, model_name: str, device: str, threshold: float):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = torch.device(device)
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.threshold = float(threshold)

    @torch.inference_mode()
    def detect_best_box(self, image_rgb: np.ndarray, prompt: str) -> tuple[Optional[np.ndarray], float]:
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(text=[[prompt]], images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        if hasattr(self.processor, "post_process_object_detection"):
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.threshold,
            )[0]
        else:
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.threshold,
            )[0]
        if len(results["scores"]) == 0:
            return None, 0.0
        best_idx = int(torch.argmax(results["scores"]).item())
        score = float(results["scores"][best_idx].detach().cpu().item())
        box = results["boxes"][best_idx].detach().cpu().numpy().astype(np.float32)
        return box, score


def build_sam2_predictor(checkpoint_path: Path, config_path: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        config_file=config_path,
        ckpt_path=str(checkpoint_path),
        device=device,
        apply_postprocessing=False,
    )
    return SAM2ImagePredictor(model)


def segment_with_box(predictor, image_rgb: np.ndarray, box_xyxy: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        if device == "cuda":
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
    if masks is None or len(masks) == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    best = int(np.argmax(iou_scores))
    return (masks[best] > 0).astype(np.uint8) * 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OWLv2 text prompts for SAM2 object masks")
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--camera", default="observation.images.head.color")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default=Path("results_sam2_prompt_test"), type=Path)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--detector-model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--detector-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.checkpoint}")

    episode_dirs = sorted(path for path in args.episodes_root.iterdir() if path.is_dir())
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.json").write_text(
        json.dumps({"prompts": args.prompts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] Loading detector: {args.detector_model}")
    detector = TextBoxDetector(args.detector_model, args.device, args.detector_threshold)
    print("[INFO] Loading SAM2")
    predictor = build_sam2_predictor(args.checkpoint, args.config, args.device)

    rows: list[dict] = []
    for prompt in args.prompts:
        prompt_slug = slugify(prompt)
        print(f"[PROMPT] {prompt}")
        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            frame_path = episode_dir / "final_frames" / f"{args.camera}.png"
            target_path = episode_dir / "target_mask.png"
            image_bgr = imread_unicode(frame_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[WARN] {episode_id}: missing frame {frame_path}")
                continue
            target = load_binary_mask(target_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            box, detector_score = detector.detect_best_box(image_rgb, prompt)
            if box is None:
                pred = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            else:
                pred = segment_with_box(predictor, image_rgb, box, args.device)

            metrics = compute_metrics(pred, target)
            row = {
                "prompt": prompt,
                "prompt_slug": prompt_slug,
                "episode_id": episode_id,
                "detector_score": detector_score,
                "box_xyxy": box.tolist() if box is not None else None,
                **metrics,
            }
            rows.append(row)

            prompt_dir = args.output_dir / prompt_slug
            imwrite_unicode(prompt_dir / "masks" / f"{episode_id}.png", pred)
            overlay = make_overlay(image_bgr, target, pred, f"{episode_id} | {prompt}", metrics)
            imwrite_unicode(prompt_dir / "overlays" / f"{episode_id}.png", overlay)

    csv_path = args.output_dir / "episode_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = []
    for prompt in args.prompts:
        subset = [r for r in rows if r["prompt"] == prompt]
        if not subset:
            continue
        summary.append({
            "prompt": prompt,
            "episodes": len(subset),
            "mean_target_region_coverage": float(np.mean([r["target_region_coverage"] for r in subset])),
            "mean_object_target_overlap_ratio": float(np.mean([r["object_target_overlap_ratio"] for r in subset])),
            "mean_iou": float(np.mean([r["iou"] for r in subset])),
            "mean_pose_error": float(np.mean([r["pose_error"] for r in subset])),
            "detections": int(sum(r["box_xyxy"] is not None for r in subset)),
        })
    summary.sort(key=lambda r: (r["mean_target_region_coverage"], r["mean_iou"]), reverse=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {args.output_dir / 'summary.csv'}")
    print("[TOP]")
    for row in summary[:5]:
        print(
            f"{row['prompt']}: coverage={row['mean_target_region_coverage']:.4f}, "
            f"iou={row['mean_iou']:.4f}, pose={row['mean_pose_error']:.4f}, "
            f"detections={row['detections']}/{row['episodes']}"
        )


if __name__ == "__main__":
    main()
