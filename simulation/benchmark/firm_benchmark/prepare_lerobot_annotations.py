#!/usr/bin/env python3
"""
Prepare FIRM annotation inputs from a LeRobot-v3-style dataset.

This script reads:
    data/chunk-*/file-*.parquet
    meta/info.json
    meta/tasks.parquet
    videos/observation.images.*

It exports:
    annotation_workspace/
    ├── episodes/
    │   ├── episode_000000/
    │   │   ├── meta.json
    │   │   ├── actions.npy
    │   │   ├── sampled_frames/
    │   │   └── final_frames/
    └── raw_episode_metrics_template.jsonl

The generated raw_episode_metrics_template.jsonl is NOT final DAP metrics.
It is a template to be completed by manual/model annotation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


CAMERAS = [
    "observation.images.head.color",
    "observation.images.head.depth",
    "observation.images.hand.left.color",
    "observation.images.hand.left.depth",
    "observation.images.hand.right.color",
    "observation.images.hand.right.depth",
]


TASK_NAME_MAP = {
    "manual": "Manual",
    "instruction": "Manual",
    "cable": "Cable",
    "mouse": "Cable",
    "box": "Box",
    "sponge": "Sponge",
    "haimian": "Sponge",
    "haimiandian": "Sponge",
    "tape": "Tape",
}


def infer_task_name(text: str) -> str:
    low = text.lower()
    for key, task in TASK_NAME_MAP.items():
        if key in low:
            return task
    return "Unknown"


def load_all_parquets(dataset_root: Path) -> pd.DataFrame:
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    dfs = []
    for p in parquet_files:
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_tasks(dataset_root: Path) -> Dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}

    tasks_df = pd.read_parquet(tasks_path)
    tasks = {}

    # Common LeRobot forms:
    # task_index, task
    # index, task
    # task_index, task_name
    possible_id_cols = ["task_index", "index", "id"]
    possible_text_cols = ["task", "task_name", "description", "instruction"]

    id_col = next((c for c in possible_id_cols if c in tasks_df.columns), None)
    text_col = next((c for c in possible_text_cols if c in tasks_df.columns), None)

    if id_col is None or text_col is None:
        return {}

    for _, row in tasks_df.iterrows():
        tasks[int(row[id_col])] = str(row[text_col])

    return tasks


def detect_episode_col(df: pd.DataFrame) -> str:
    for c in ["episode_index", "episode_id", "episode"]:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find episode column. Columns: {list(df.columns)}")


def detect_frame_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["frame_index", "index", "timestamp"]:
        if c in df.columns:
            return c
    return None


def detect_action_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "action",
        "actions",
        "action.delta",
        "action.eef",
        "action_token",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        if c.startswith("action"):
            return c

    return None


def detect_task_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["task_index", "task", "task_name"]:
        if c in df.columns:
            return c
    return None


def to_numpy_action(values: List[Any]) -> np.ndarray:
    arrs = []
    for v in values:
        if isinstance(v, np.ndarray):
            arrs.append(v.astype(np.float32).reshape(-1))
        elif isinstance(v, list):
            arrs.append(np.array(v, dtype=np.float32).reshape(-1))
        else:
            try:
                arrs.append(np.array(v, dtype=np.float32).reshape(-1))
            except Exception:
                continue

    if not arrs:
        return np.zeros((0, 0), dtype=np.float32)

    max_dim = max(a.shape[0] for a in arrs)
    padded = []
    for a in arrs:
        if a.shape[0] < max_dim:
            a = np.pad(a, (0, max_dim - a.shape[0]))
        padded.append(a)

    return np.stack(padded, axis=0)


def find_video_files(dataset_root: Path, camera_name: str) -> List[Path]:
    video_root = dataset_root / "videos"
    direct = video_root / camera_name

    files = []
    if direct.exists():
        files.extend(sorted(direct.glob("**/*.mp4")))
        files.extend(sorted(direct.glob("**/*.avi")))

    if files:
        return files

    # Fallback: search anywhere under videos
    pattern_safe = camera_name.replace(".", r"\.")
    for p in sorted(video_root.glob("**/*")):
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi"}:
            if re.search(pattern_safe, str(p)):
                files.append(p)

    return files


def select_video_for_episode(video_files: List[Path], episode_index: int) -> Optional[Path]:
    """
    Try to match an episode video by filename.
    Supports names like:
        episode_000001.mp4
        episode_1.mp4
        file-001.mp4
    If no match is found and there is only one video, return it.
    """
    if not video_files:
        return None

    patterns = [
        f"{episode_index:06d}",
        f"{episode_index:05d}",
        f"{episode_index:04d}",
        f"{episode_index:03d}",
        f"{episode_index}",
    ]

    for p in video_files:
        stem = p.stem
        if any(token in stem for token in patterns):
            return p

    if len(video_files) == 1:
        return video_files[0]

    # Fallback: sorted order by episode index if possible
    if 0 <= episode_index < len(video_files):
        return video_files[episode_index]

    return None


def read_video_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    frame_index = max(0, min(frame_index, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    return frame


def read_video_last_frame(video_path: Path) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def export_episode_frames(
    dataset_root: Path,
    out_dir: Path,
    episode_index: int,
    num_frames: int,
    sample_count: int,
) -> Dict[str, Any]:
    """
    Export sampled and final frames for all available cameras.
    """
    result = {}

    sample_indices = np.linspace(0, max(num_frames - 1, 0), sample_count).astype(int).tolist()

    for camera in CAMERAS:
        video_files = find_video_files(dataset_root, camera)
        video_path = select_video_for_episode(video_files, episode_index)
        if video_path is None:
            continue

        result[camera] = str(video_path)

        # Final frame
        frame = read_video_last_frame(video_path)
        if frame is not None:
            save_frame(out_dir / "final_frames" / f"{camera}.png", frame)

        # Sampled frames
        for idx in sample_indices:
            frame = read_video_frame(video_path, idx)
            if frame is not None:
                save_frame(out_dir / "sampled_frames" / camera / f"{idx:06d}.png", frame)

    return result


def infer_episode_task(
    episode_df: pd.DataFrame,
    dataset_root: Path,
    task_col: Optional[str],
    tasks: Dict[int, str],
    fallback_text: str,
) -> Tuple[str, str]:
    """
    Return:
        firm_task_category, raw_task_text
    """
    raw_task_text = fallback_text

    if task_col is not None:
        value = episode_df.iloc[0][task_col]

        if task_col == "task_index":
            idx = int(value)
            raw_task_text = tasks.get(idx, str(idx))
        else:
            raw_task_text = str(value)

    firm_task = infer_task_name(raw_task_text)
    return firm_task, raw_task_text


def build_episode_package(
    dataset_root: Path,
    output_root: Path,
    episode_index: int,
    episode_df: pd.DataFrame,
    action_col: Optional[str],
    task_col: Optional[str],
    tasks: Dict[int, str],
    sample_count: int,
) -> Dict[str, Any]:
    episode_name = f"episode_{episode_index:06d}"
    episode_out = output_root / "episodes" / episode_name
    episode_out.mkdir(parents=True, exist_ok=True)

    num_frames = len(episode_df)

    # Save actions.
    if action_col is not None:
        actions = to_numpy_action(episode_df[action_col].tolist())
    else:
        actions = np.zeros((num_frames, 0), dtype=np.float32)

    np.save(episode_out / "actions.npy", actions)

    # Infer task.
    fallback_text = dataset_root.name
    firm_task, raw_task_text = infer_episode_task(
        episode_df=episode_df,
        dataset_root=dataset_root,
        task_col=task_col,
        tasks=tasks,
        fallback_text=fallback_text,
    )

    # Export video frames.
    video_map = export_episode_frames(
        dataset_root=dataset_root,
        out_dir=episode_out,
        episode_index=episode_index,
        num_frames=num_frames,
        sample_count=sample_count,
    )

    meta = {
        "episode_id": episode_name,
        "episode_index": int(episode_index),
        "method": "unknown_method",
        "task": firm_task,
        "raw_task_text": raw_task_text,
        "num_frames": int(num_frames),
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
        "videos": video_map,
        "manual_flags": {},
        "manual_metrics": {}
    }

    (episode_out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    template = {
        "episode_id": episode_name,
        "method": "unknown_method",
        "task": firm_task,
        "metrics": {
            "timeout": False

            # /*
            # Fill these fields after annotation or metric extraction.

            # Manual:
            #     "insertion_depth": 0.0,
            #     "alignment_error": 0.0,
            #     "severe_fold": false,
            #     "jammed": false

            # Cable:
            #     "contained_cable_length": 0.0,
            #     "residual_tangling": 0.0,
            #     "boundary_contact": 0.0,
            #     "slip": false,
            #     "dropped": false

            # Box:
            #     "fold_angle": 90.0,
            #     "hinge_jam": false,
            #     "collision": false,
            #     "severe_deformation": false

            # Sponge:
            #     "target_region_coverage": 0.0,
            #     "pose_error": 0.0,
            #     "folded_corner": false,
            #     "trapped_corner": false,
            #     "residual_compression": 0.0,
            #     "rebound_shift": 0.0

            # Tape:
            #     "position_error": 0.0,
            #     "orientation_error": 0.0,
            #     "rolling": false,
            #     "face_down": false,
            #     "outside_target": false,
            #     "on_lid": false
            # */
        },
    }

    return template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare annotation workspace from a LeRobot-v3-style FIRM dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root of LeRobot dataset, e.g. A2p_dataset_haimiandian_0414_50_merged.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output annotation workspace.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Number of frames to sample per episode per camera.",
    )
    args = parser.parse_args()

    df = load_all_parquets(args.dataset_root)
    episode_col = detect_episode_col(df)
    frame_col = detect_frame_col(df)
    action_col = detect_action_col(df)
    task_col = detect_task_col(df)
    tasks = load_tasks(args.dataset_root)

    if frame_col is not None:
        df = df.sort_values([episode_col, frame_col])
    else:
        df = df.sort_values([episode_col])

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Columns:", list(df.columns))
    print("[INFO] episode_col:", episode_col)
    print("[INFO] frame_col:", frame_col)
    print("[INFO] action_col:", action_col)
    print("[INFO] task_col:", task_col)

    templates = []
    grouped = df.groupby(episode_col, sort=True)

    for episode_index, episode_df in grouped:
        print(f"[EPISODE] {episode_index} frames={len(episode_df)}")
        template = build_episode_package(
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            episode_index=int(episode_index),
            episode_df=episode_df,
            action_col=action_col,
            task_col=task_col,
            tasks=tasks,
            sample_count=args.sample_count,
        )
        templates.append(template)

    # JSON does not support comments, so write a clean template.
    cleaned_templates = []
    for t in templates:
        cleaned_templates.append({
            "episode_id": t["episode_id"],
            "method": t["method"],
            "task": t["task"],
            "metrics": t["metrics"],
        })

    out_jsonl = args.output_root / "raw_episode_metrics_template.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in cleaned_templates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Exported annotation workspace to: {args.output_root}")
    print(f"[OK] Wrote template: {out_jsonl}")


if __name__ == "__main__":
    main()
