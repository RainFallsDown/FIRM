#!/usr/bin/env python3
"""Interactively annotate real target masks for episodes.

Workflow per episode:
1) ROI window pops up.
2) Drag rectangle.
3) Press Enter/Space in the ROI window to accept.
4) Press `c` in the ROI window to cancel the rectangle.

Then in terminal:
- Enter: save this rectangle to `target_mask.png`
- `r`: redraw this episode
- `s`: skip this episode
- `q`: quit this session

Outputs per episode:
- target_mask.png
- target_mask_overlay.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


HEAD_COLOR_CAMERA = "observation.images.head.color"


def list_episodes(episodes_root: Path) -> List[Path]:
    """List episode directories.

    Only directories whose names start with `episode_` are treated as episodes.
    """
    if not episodes_root.exists():
        raise FileNotFoundError(f"Episodes root does not exist: {episodes_root}")

    return sorted(
        path
        for path in episodes_root.iterdir()
        if path.is_dir() and path.name.startswith("episode_")
    )


def center_template(shape: Tuple[int, int]) -> np.ndarray:
    """Create the default center-template mask used by earlier initialization."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    ry, rx = height // 3, width // 3
    mask[cy - ry : cy + ry, cx - rx : cx + rx] = 255
    return mask


def classify_target_mask(path: Path) -> str:
    """Classify an existing target_mask.png.

    Returns:
        missing_or_unreadable
        empty
        center_template
        ok
    """
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        return "missing_or_unreadable"

    target = ((target > 127).astype(np.uint8) * 255)

    if int((target > 0).sum()) == 0:
        return "empty"

    if np.array_equal(target, center_template(target.shape)):
        return "center_template"

    return "ok"


def load_preview_image(episode_dir: Path, camera_key: str) -> np.ndarray:
    """Load the preview frame used for ROI annotation.

    Priority:
    1) final_frames/{camera_key}.png
    2) last sampled frame from sampled_frames/{camera_key}/
    """
    final_frame = episode_dir / "final_frames" / f"{camera_key}.png"
    if final_frame.exists():
        image = cv2.imread(str(final_frame))
        if image is not None:
            return image

    sampled_dir = episode_dir / "sampled_frames" / camera_key
    frame_files = sorted(sampled_dir.glob("*.png")) if sampled_dir.exists() else []
    if frame_files:
        image = cv2.imread(str(frame_files[-1]))
        if image is not None:
            return image

    raise FileNotFoundError(
        f"No usable preview frame for {episode_dir.name} with camera `{camera_key}`"
    )


def draw_overlay(image: np.ndarray, text: str) -> np.ndarray:
    """Draw instruction text on the image without changing image size."""
    view = image.copy()

    bar_h = 48
    cv2.rectangle(view, (0, 0), (view.shape[1], bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        view,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return view


def make_rect_mask(shape: Tuple[int, int], rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Create a binary rectangle mask from an ROI rectangle.

    Args:
        shape: (height, width)
        rect: (x, y, w, h)

    Returns:
        uint8 mask with foreground = 255.
    """
    x, y, w, h = rect
    height, width = shape

    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)

    mask = np.zeros(shape, dtype=np.uint8)

    if w <= 0 or h <= 0:
        return mask

    x2 = min(x + w, width)
    y2 = min(y + h, height)

    if x2 <= x or y2 <= y:
        return mask

    mask[y:y2, x:x2] = 255
    return mask


def make_target_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a visualization overlay for the target mask."""
    overlay = image.copy()

    mask = ((mask > 127).astype(np.uint8) * 255)
    mask_bool = mask > 0

    green = np.zeros_like(image)
    green[:, :, 1] = 255

    alpha = 0.35
    overlay[mask_bool] = (
        (1.0 - alpha) * overlay[mask_bool] + alpha * green[mask_bool]
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def save_target_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    """Save target mask overlay image."""
    overlay = make_target_overlay(image, mask)
    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise RuntimeError(f"Failed to write target overlay: {output_path}")


def load_status(path: Path) -> Dict[str, str]:
    """Load annotation status JSON."""
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        print(f"[WARN] Failed to read status file {path}: {exc}")

    return {}


def save_status(path: Path, status: Dict[str, str]) -> None:
    """Save annotation status JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_destroy_window(window_name: str) -> None:
    """Destroy an OpenCV window safely."""
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def annotate_episode(
    episode_dir: Path,
    camera_key: str,
    overwrite: bool,
    window_name: str,
    force_existing: bool = False,
) -> str:
    """Interactively annotate one episode.

    Args:
        episode_dir: episode directory.
        camera_key: camera name.
        overwrite: overwrite target_mask.png if it exists.
        window_name: OpenCV ROI window name.
        force_existing: annotate even if target_mask.png exists. Useful for
            --only-center-template.

    Returns:
        saved
        skipped
        exists
        quit
    """
    target_path = episode_dir / "target_mask.png"
    overlay_path = episode_dir / "target_mask_overlay.png"

    if target_path.exists() and not overwrite and not force_existing:
        return "exists"

    image = load_preview_image(episode_dir, camera_key)

    help_text = (
        f"{episode_dir.name} | Drag ROI | Enter/Space=confirm | c=cancel"
    )

    while True:
        view = draw_overlay(image, help_text)

        print(f"\n[ANNOTATE] {episode_dir.name}")
        print("[ROI WINDOW] Drag target rectangle, then press Enter/Space.")
        print("[ROI WINDOW] Press c to cancel current rectangle.")

        rect = cv2.selectROI(
            window_name,
            view,
            showCrosshair=True,
            fromCenter=False,
        )
        safe_destroy_window(window_name)

        x, y, w, h = [int(v) for v in rect]

        if w <= 0 or h <= 0:
            choice = input(
                f"{episode_dir.name}: empty ROI, redraw/skip/quit [r/s/q]: "
            ).strip().lower()

            if choice == "s":
                return "skipped"
            if choice == "q":
                return "quit"

            continue

        mask = make_rect_mask((image.shape[0], image.shape[1]), (x, y, w, h))

        if int((mask > 0).sum()) == 0:
            print(f"[WARN] Empty mask after clipping for {episode_dir.name}, redraw.")
            continue

        preview = make_target_overlay(image, mask)
        preview_name = f"Preview Target {episode_dir.name}"
        cv2.imshow(preview_name, preview)
        cv2.waitKey(1)

        print(f"[RECT] x={x}, y={y}, w={w}, h={h}")
        print(f"[PREVIEW] Green overlay should cover the real target region.")
        choice = input(
            f"{episode_dir.name}: save this target mask? "
            "[Enter=save / r=redraw / s=skip / q=quit]: "
        ).strip().lower()

        safe_destroy_window(preview_name)

        if choice == "q":
            return "quit"

        if choice == "s":
            return "skipped"

        if choice == "r":
            continue

        ok = cv2.imwrite(str(target_path), mask)
        if not ok:
            raise RuntimeError(f"Failed to write target mask: {target_path}")

        save_target_overlay(image, mask, overlay_path)

        print(f"[WRITE] {target_path}")
        print(f"[WRITE] {overlay_path}")

        return "saved"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive annotation of real target masks."
    )

    parser.add_argument(
        "--episodes-root",
        required=True,
        type=Path,
        help="Root directory containing episode_* folders.",
    )

    parser.add_argument(
        "--camera",
        default=HEAD_COLOR_CAMERA,
        help="Camera key for preview frames.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "Episode list index to start from. "
            "For example, if folders are episode_000000, episode_000001, "
            "episode_000002, then --start-index 2 starts from episode_000002."
        ),
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process in this session.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target_mask.png files.",
    )

    parser.add_argument(
        "--only-center-template",
        action="store_true",
        help=(
            "Only annotate episodes whose current target_mask.png is classified "
            "as the default center_template."
        ),
    )

    parser.add_argument(
        "--status-json",
        type=Path,
        default=Path("annotation_workspace_sponge/target_annotation_status.json"),
        help="Progress status file for resumable annotation sessions.",
    )

    args = parser.parse_args()

    episodes_all = list_episodes(args.episodes_root)

    if not episodes_all:
        raise RuntimeError(f"No episode_* directories found under {args.episodes_root}")

    episodes = episodes_all[args.start_index :]

    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"[INFO] episodes_root = {args.episodes_root}")
    print(f"[INFO] camera = {args.camera}")
    print(f"[INFO] total episode_* dirs = {len(episodes_all)}")
    print(f"[INFO] selected episodes = {len(episodes)}")
    print(f"[INFO] start_index = {args.start_index}")
    print(f"[INFO] max_episodes = {args.max_episodes}")
    print(f"[INFO] overwrite = {args.overwrite}")
    print(f"[INFO] only_center_template = {args.only_center_template}")
    print(f"[INFO] status_json = {args.status_json}")

    if episodes:
        print(f"[INFO] first selected episode = {episodes[0].name}")
        print(f"[INFO] last selected episode = {episodes[-1].name}")

    status_map = load_status(args.status_json)

    saved = 0
    skipped = 0
    exists = 0
    filtered = 0
    failed = 0

    for idx, episode_dir in enumerate(episodes, start=1):
        episode_id = episode_dir.name

        if (
            episode_id in status_map
            and status_map[episode_id] == "saved"
            and not args.overwrite
            and not args.only_center_template
        ):
            filtered += 1
            print(f"[FILTERED:STATUS_SAVED] {episode_id}")
            continue

        if args.only_center_template:
            cls = classify_target_mask(episode_dir / "target_mask.png")
            if cls != "center_template":
                filtered += 1
                print(f"[FILTERED:{cls}] {episode_id}")
                continue
            print(f"[CENTER_TEMPLATE] {episode_id}")

        try:
            result = annotate_episode(
                episode_dir=episode_dir,
                camera_key=args.camera,
                overwrite=args.overwrite,
                window_name=f"Annotate Target {idx}/{len(episodes)}",
                force_existing=args.only_center_template,
            )
        except Exception as exc:
            result = "failed"
            failed += 1
            print(f"[FAILED] {episode_id}: {exc}")

        status_map[episode_id] = result
        save_status(args.status_json, status_map)

        if result == "saved":
            saved += 1
            print(f"[SAVED] {episode_id}")
            continue

        if result == "exists":
            exists += 1
            print(f"[EXISTS] {episode_id}")
            continue

        if result == "skipped":
            skipped += 1
            print(f"[SKIP] {episode_id}")
            continue

        if result == "quit":
            print("[QUIT] Annotation interrupted by user.")
            break

        if result == "failed":
            continue

    cv2.destroyAllWindows()

    print(
        f"[DONE] saved={saved}, skipped={skipped}, "
        f"exists={exists}, filtered={filtered}, failed={failed}"
    )
    print(f"[STATUS] {args.status_json}")


if __name__ == "__main__":
    main()