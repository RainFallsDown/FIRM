#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    root = args.dataset_root
    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    print("[DATA PARQUET FILES]")
    for p in parquet_files:
        print(" ", p)

    df = pd.read_parquet(parquet_files[0])
    print("\n[COLUMNS]")
    for c in df.columns:
        print(" ", c)

    print("\n[HEAD]")
    print(df.head(args.n))

    info_path = root / "meta" / "info.json"
    if info_path.exists():
        print("\n[INFO.JSON]")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        print(json.dumps(info, indent=2, ensure_ascii=False)[:3000])

    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        print("\n[TASKS]")
        tasks = pd.read_parquet(tasks_path)
        print(tasks.head(20))

    print("\n[VIDEO FOLDERS]")
    video_root = root / "videos"
    if video_root.exists():
        for p in sorted(video_root.iterdir()):
            if p.is_dir():
                mp4s = list(p.glob("**/*.mp4"))
                print(f" {p.name}: {len(mp4s)} mp4 files")


if __name__ == "__main__":
    main()