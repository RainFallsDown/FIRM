#!${CONDA_ROOT:-/path/to/miniconda3}/envs/act-grpo/bin/python
"""Build a mixed LeRobot v3 dataset from the configured Tianqing datasets."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE_KEYS = ("jiaodai", "box_task_id_15", "mouse_1", "mouse_2", "mouse_3")
DEFAULT_OUTPUT_KEY = "tianqing_mixed"
DEFAULT_OUTPUT_REPO_ID = "tianqing/tianqing_mixed"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    repo_id: str
    root: Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        parts = shlex.split(raw_value, posix=True)
        values[key] = parts[0] if parts else ""
    return values


def load_dataset_config(act_grpo_root: Path, key: str) -> DatasetConfig:
    config_path = act_grpo_root / "configs" / "datasets" / f"{key}.env"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing dataset config: {config_path}")
    values = parse_env_file(config_path)
    dataset_key = values.get("DATASET_KEY")
    repo_id = values.get("DATASET_REPO_ID")
    root = values.get("DATASET_ROOT")
    if dataset_key != key:
        raise ValueError(f"{config_path} has DATASET_KEY={dataset_key!r}, expected {key!r}")
    if not repo_id:
        raise ValueError(f"{config_path} is missing DATASET_REPO_ID")
    if not root:
        raise ValueError(f"{config_path} is missing DATASET_ROOT")
    return DatasetConfig(key=key, repo_id=repo_id, root=Path(root))


def require_lerobot_v3_dataset(config: DatasetConfig) -> None:
    required = (
        config.root / "meta" / "info.json",
        config.root / "meta" / "stats.json",
        config.root / "meta" / "tasks.parquet",
        config.root / "meta" / "episodes",
        config.root / "data",
        config.root / "videos",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{config.key} is not a complete LeRobot v3 dataset: {missing}")


def ensure_lerobot_import_path(act_grpo_root: Path) -> None:
    lerobot_src = act_grpo_root / "resources" / "lerobot" / "src"
    if not (lerobot_src / "lerobot").is_dir():
        raise FileNotFoundError(f"missing LeRobot source: {lerobot_src}")
    sys.path.insert(0, str(lerobot_src))


def normalized_feature(feature: dict) -> dict:
    clean = dict(feature)
    clean.pop("fps", None)
    return clean


def common_features(configs: list[DatasetConfig]) -> dict:
    infos = [read_json(config.root / "meta" / "info.json") for config in configs]
    feature_maps = [info["features"] for info in infos]
    common_keys = [
        key
        for key in feature_maps[0]
        if all(key in features for features in feature_maps[1:])
    ]
    features = {
        key: normalized_feature(feature_maps[0][key])
        for key in common_keys
    }
    for feature_map in feature_maps[1:]:
        for key in list(features):
            if normalized_feature(feature_map[key]) != features[key]:
                raise ValueError(f"common feature is not compatible after normalization: {key}")
    for required in ("action", "observation.state"):
        if required not in features:
            raise ValueError(f"missing required common feature: {required}")
    if not any(key.startswith("observation.images.") for key in features):
        raise ValueError("mixed dataset has no common observation image features")
    return features


def safe_replace_view_dir(act_grpo_root: Path, view_root: Path) -> None:
    view_base = (act_grpo_root / "work" / "mixed_dataset_views").resolve()
    resolved = view_root.resolve(strict=False)
    if resolved == view_base or not resolved.is_relative_to(view_base):
        raise ValueError(f"refusing metadata view outside {view_base}: {view_root}")
    if view_root.is_symlink() or view_root.is_file():
        view_root.unlink()
    elif view_root.exists():
        shutil.rmtree(view_root)
    view_root.mkdir(parents=True)


def filtered_episode_columns(columns: list[str], feature_keys: set[str]) -> list[str]:
    keep = []
    for column in columns:
        if column.startswith("videos/"):
            video_key = column.split("/", 2)[1]
            if video_key in feature_keys:
                keep.append(column)
            continue
        if column.startswith("stats/"):
            feature_key = column.split("/", 2)[1]
            if feature_key in feature_keys:
                keep.append(column)
            continue
        keep.append(column)
    return keep


def create_metadata_view(
    act_grpo_root: Path,
    config: DatasetConfig,
    features: dict,
) -> DatasetConfig:
    import pandas as pd

    view_root = act_grpo_root / "work" / "mixed_dataset_views" / config.key
    safe_replace_view_dir(act_grpo_root, view_root)
    (view_root / "meta").mkdir(parents=True, exist_ok=True)

    source_info = read_json(config.root / "meta" / "info.json")
    source_info["features"] = features
    write_json(view_root / "meta" / "info.json", source_info)

    feature_keys = set(features)
    source_stats = read_json(config.root / "meta" / "stats.json")
    filtered_stats = {
        key: value
        for key, value in source_stats.items()
        if key in feature_keys
    }
    write_json(view_root / "meta" / "stats.json", filtered_stats)

    shutil.copy2(config.root / "meta" / "tasks.parquet", view_root / "meta" / "tasks.parquet")

    for source_episode_path in sorted((config.root / "meta" / "episodes").glob("**/*.parquet")):
        relative_path = source_episode_path.relative_to(config.root / "meta" / "episodes")
        target_path = view_root / "meta" / "episodes" / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        episodes = pd.read_parquet(source_episode_path)
        episodes = episodes[filtered_episode_columns(list(episodes.columns), feature_keys)]
        episodes.to_parquet(target_path, index=False)

    for dirname in ("data", "videos"):
        (view_root / dirname).symlink_to(config.root / dirname, target_is_directory=True)

    return DatasetConfig(key=config.key, repo_id=config.repo_id, root=view_root)


def create_metadata_views(
    act_grpo_root: Path,
    configs: list[DatasetConfig],
    features: dict,
) -> list[DatasetConfig]:
    return [create_metadata_view(act_grpo_root, config, features) for config in configs]


def load_metadata(configs: list[DatasetConfig], act_grpo_root: Path):
    ensure_lerobot_import_path(act_grpo_root)
    from lerobot.datasets.aggregate import validate_all_metadata
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = [
        LeRobotDatasetMetadata(config.repo_id, root=config.root)
        for config in configs
    ]
    validate_all_metadata(metadata)
    return metadata


def safe_output_root(act_grpo_root: Path, output_root: Path) -> Path:
    allowed_root = (act_grpo_root / "resources" / "datasets").resolve()
    resolved = output_root.resolve(strict=False)
    if resolved == allowed_root or not resolved.is_relative_to(allowed_root):
        raise ValueError(f"refusing output outside {allowed_root}: {output_root}")
    return resolved


def remove_existing_output(act_grpo_root: Path, output_root: Path, force: bool) -> None:
    safe_output_root(act_grpo_root, output_root)
    if not output_root.exists():
        return
    if not force:
        raise FileExistsError(f"output already exists, pass --force to replace it: {output_root}")
    if output_root.is_symlink() or output_root.is_file():
        raise ValueError(f"refusing to remove non-directory output: {output_root}")
    shutil.rmtree(output_root)


def write_source_manifest(
    output_root: Path,
    source_configs: list[DatasetConfig],
    build_configs: list[DatasetConfig],
    metadata: list[object],
) -> None:
    manifest = {
        "source_keys": [config.key for config in source_configs],
        "source_repo_ids": [config.repo_id for config in source_configs],
        "source_roots": [str(config.root) for config in source_configs],
        "metadata_view_roots": [str(config.root) for config in build_configs],
        "source_total_episodes": [int(meta.total_episodes) for meta in metadata],
        "source_total_frames": [int(meta.total_frames) for meta in metadata],
        "total_episodes": int(sum(meta.total_episodes for meta in metadata)),
        "total_frames": int(sum(meta.total_frames for meta in metadata)),
    }
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "mixed_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def print_plan(
    source_configs: list[DatasetConfig],
    build_configs: list[DatasetConfig],
    metadata: list[object],
    output_key: str,
    output_repo_id: str,
    output_root: Path,
    features: dict,
) -> None:
    print(f"output_key={output_key}")
    print(f"output_repo_id={output_repo_id}")
    print(f"output_root={output_root}")
    print("common_features=" + ",".join(features))
    for source_config, build_config, meta in zip(source_configs, build_configs, metadata, strict=True):
        print(
            "source="
            f"{source_config.key},repo_id={source_config.repo_id},episodes={meta.total_episodes},"
            f"frames={meta.total_frames},metadata_view={build_config.root}"
        )
    print(f"total_episodes={sum(meta.total_episodes for meta in metadata)}")
    print(f"total_frames={sum(meta.total_frames for meta in metadata)}")


def build_dataset(
    source_configs: list[DatasetConfig],
    build_configs: list[DatasetConfig],
    metadata: list[object],
    output_repo_id: str,
    output_root: Path,
    data_files_size_in_mb: float,
    video_files_size_in_mb: float,
    chunk_size: int,
) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[config.repo_id for config in build_configs],
        roots=[config.root for config in build_configs],
        aggr_repo_id=output_repo_id,
        aggr_root=output_root,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
        chunk_size=chunk_size,
    )
    write_source_manifest(output_root, source_configs, build_configs, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-grpo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--source-key", action="append", dest="source_keys")
    parser.add_argument("--output-key", default=DEFAULT_OUTPUT_KEY)
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-files-size-in-mb", type=float, default=100.0)
    parser.add_argument("--video-files-size-in-mb", type=float, default=200.0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    act_grpo_root = args.act_grpo_root.resolve()
    source_keys = args.source_keys or list(DEFAULT_SOURCE_KEYS)
    output_root = args.output_root or (
        act_grpo_root / "resources" / "datasets" / args.output_key
    )

    source_configs = [load_dataset_config(act_grpo_root, key) for key in source_keys]
    for config in source_configs:
        require_lerobot_v3_dataset(config)

    features = common_features(source_configs)
    build_configs = create_metadata_views(act_grpo_root, source_configs, features)
    metadata = load_metadata(build_configs, act_grpo_root)
    print_plan(
        source_configs,
        build_configs,
        metadata,
        args.output_key,
        args.output_repo_id,
        output_root,
        features,
    )

    if args.dry_run:
        print("MIXED_DATASET_DRY_RUN_OK")
        return 0

    remove_existing_output(act_grpo_root, output_root, args.force)
    build_dataset(
        source_configs=source_configs,
        build_configs=build_configs,
        metadata=metadata,
        output_repo_id=args.output_repo_id,
        output_root=output_root,
        data_files_size_in_mb=args.data_files_size_in_mb,
        video_files_size_in_mb=args.video_files_size_in_mb,
        chunk_size=args.chunk_size,
    )
    print(f"MIXED_DATASET_BUILD_OK root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
