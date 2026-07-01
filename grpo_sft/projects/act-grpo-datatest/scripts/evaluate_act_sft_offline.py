#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ACT-SFT checkpoint reconstruction check.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint/pretrained_model directory.")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--compare-random", action="store_true")
    return parser.parse_args()


def evenly_spaced_indices(total: int, num_samples: int) -> list[int]:
    if num_samples <= 0 or num_samples >= total:
        return list(range(total))
    if num_samples == 1:
        return [0]
    return sorted({round(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)})


def make_processors(cfg: TrainPipelineConfig, policy, dataset, device: torch.device):
    processor_kwargs = {
        "preprocessor_overrides": {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
    }
    postprocessor_kwargs = {
        "postprocessor_overrides": {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        }
    }
    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )


def evaluate_policy(policy, preprocessor, dataloader: DataLoader) -> dict[str, float]:
    # ACT's variational loss only returns mu/log_sigma in training mode. We still use
    # inference_mode(), so this computes the training objective without updating weights.
    policy.train()
    totals: dict[str, float] = {"loss": 0.0}
    count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, output = policy.forward(batch)
            batch_size = next(iter(batch.values())).shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            if output:
                for key, value in output.items():
                    if isinstance(value, (float, int)) and math.isfinite(float(value)):
                        totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            count += batch_size
    elapsed = time.perf_counter() - started
    result = {f"{key}_mean": value / max(count, 1) for key, value in totals.items()}
    result["num_eval_samples"] = float(count)
    result["elapsed_s"] = elapsed
    return result


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint.name != "pretrained_model":
        checkpoint = checkpoint / "pretrained_model"
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {checkpoint}")

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    cfg = TrainPipelineConfig.from_pretrained(
        checkpoint,
        cli_args=[
            f"--policy.device={device.type}",
            "--dataset.image_transforms.enable=false",
            f"--batch_size={args.batch_size}",
            f"--num_workers={args.num_workers}",
        ],
    )
    cfg.policy.pretrained_path = str(checkpoint)
    cfg.policy.device = device.type
    cfg.dataset.image_transforms.enable = False
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size

    dataset = make_dataset(cfg)
    indices = evenly_spaced_indices(len(dataset), args.num_samples)
    subset = Subset(dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_processors(cfg, policy, dataset, device)
    trained = evaluate_policy(policy, preprocessor, dataloader)

    result = {
        "checkpoint": str(checkpoint),
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root),
        "dataset_num_frames": len(dataset),
        "sample_strategy": "evenly_spaced",
        "requested_num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "device": device.type,
        "trained": trained,
    }

    if args.compare_random:
        random_cfg = copy.deepcopy(cfg)
        random_cfg.policy.pretrained_path = None
        random_policy = make_policy(random_cfg.policy, ds_meta=dataset.meta, rename_map=random_cfg.rename_map)
        random_preprocessor, _ = make_pre_post_processors(
            policy_cfg=random_cfg.policy,
            dataset_stats=dataset.meta.stats,
        )
        random_result = evaluate_policy(random_policy, random_preprocessor, dataloader)
        result["random_init"] = random_result
        if trained["loss_mean"] > 0:
            result["random_to_trained_loss_ratio"] = random_result["loss_mean"] / trained["loss_mean"]

    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
