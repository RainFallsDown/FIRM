#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES


LOWER_IS_BETTER = [
    "val_loss",
    "val_action_l1_loss",
    "val_kld_loss",
    "action_mse",
    "action_rmse",
    "action_mae",
    "action_l2_per_step",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ACT zero-shot, SFT/BC, and GRPO checkpoints with one offline metric path."
    )
    parser.add_argument(
        "--zero-shot",
        action="append",
        default=[],
        help="Reference checkpoint spec label=/path. Config is loaded, weights are random initialized.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint spec label=/path/to/checkpoint_or_pretrained_model.",
    )
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.zero_shot and not args.checkpoint:
        parser.error("at least one --zero-shot or --checkpoint is required")
    return args


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Checkpoint must be label=path, got: {spec}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Checkpoint label is empty in: {spec}")
    path = Path(raw_path).expanduser().resolve()
    if path.name != "pretrained_model":
        path = path / "pretrained_model"
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {path}")
    return label, path


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


def set_forward_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(
    checkpoint: Path,
    device: torch.device,
    num_workers: int,
    dataset_root: Path | None,
) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig.from_pretrained(
        checkpoint,
        cli_args=[
            f"--policy.device={device.type}",
            "--dataset.image_transforms.enable=false",
            "--batch_size=1",
            f"--num_workers={num_workers}",
        ],
    )
    cfg.policy.pretrained_path = str(checkpoint)
    cfg.policy.device = device.type
    cfg.dataset.image_transforms.enable = False
    if dataset_root is not None:
        cfg.dataset.root = dataset_root
    cfg.num_workers = num_workers
    cfg.batch_size = 1
    return cfg


@torch.inference_mode()
def evaluate_batch_pi05(policy, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Pi0.5 uses diffusion-style sampling via predict_action_chunk, not ACT-style VAE outputs."""
    policy.eval()
    actions_hat = policy.predict_action_chunk(batch)
    actions_gt = batch[ACTION]
    t = min(actions_hat.shape[1], actions_gt.shape[1])
    actions_hat = actions_hat[:, :t]
    actions_gt = actions_gt[:, :t]
    action_is_pad = batch["action_is_pad"].bool()
    if action_is_pad.shape[1] >= t:
        action_is_pad = action_is_pad[:, :t]
    else:
        pad_extra = t - action_is_pad.shape[1]
        action_is_pad = torch.cat(
            [action_is_pad, torch.ones(action_is_pad.shape[0], pad_extra, dtype=torch.bool, device=action_is_pad.device)],
            dim=1,
        )

    valid_steps = ~action_is_pad
    valid_action_mask = valid_steps.unsqueeze(-1).expand_as(actions_hat)
    error = (actions_hat - actions_gt) * valid_action_mask

    denom = valid_action_mask.sum().clamp_min(1)
    action_mae = error.abs().sum() / denom
    action_mse = error.square().sum() / denom
    action_rmse = action_mse.sqrt()

    per_step_l2 = error.square().sum(dim=-1).sqrt()
    action_l2_per_step = per_step_l2[valid_steps].mean() if valid_steps.any() else action_mse.new_tensor(0.0)

    kld_loss = action_mae.new_tensor(0.0)
    val_loss = action_mae
    return {
        "val_loss": float(val_loss.item()),
        "val_action_l1_loss": float(action_mae.item()),
        "val_kld_loss": float(kld_loss.item()),
        "action_mse": float(action_mse.item()),
        "action_rmse": float(action_rmse.item()),
        "action_mae": float(action_mae.item()),
        "action_l2_per_step": float(action_l2_per_step.item()),
    }


@torch.inference_mode()
def evaluate_batch(policy, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    if getattr(policy.config, "type", None) == "pi05":
        return evaluate_batch_pi05(policy, batch)

    policy.train()
    model_batch = dict(batch)
    if policy.config.image_features:
        model_batch[OBS_IMAGES] = [model_batch[key] for key in policy.config.image_features]

    actions_hat, (mu_hat, log_sigma_x2_hat) = policy.model(model_batch)
    action_is_pad = model_batch["action_is_pad"].bool()
    valid_steps = ~action_is_pad
    valid_action_mask = valid_steps.unsqueeze(-1).expand_as(actions_hat)
    error = (actions_hat - model_batch[ACTION]) * valid_action_mask

    denom = valid_action_mask.sum().clamp_min(1)
    action_mae = error.abs().sum() / denom
    action_mse = error.square().sum() / denom
    action_rmse = action_mse.sqrt()

    per_step_l2 = error.square().sum(dim=-1).sqrt()
    action_l2_per_step = per_step_l2[valid_steps].mean() if valid_steps.any() else action_mse.new_tensor(0.0)

    if policy.config.use_vae:
        kld_per_sample = (
            -0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())
        ).sum(-1)
        kld_loss = kld_per_sample.mean()
    else:
        kld_loss = action_mae.new_tensor(0.0)

    val_loss = action_mae + kld_loss * policy.config.kl_weight
    return {
        "val_loss": float(val_loss.item()),
        "val_action_l1_loss": float(action_mae.item()),
        "val_kld_loss": float(kld_loss.item()),
        "action_mse": float(action_mse.item()),
        "action_rmse": float(action_rmse.item()),
        "action_mae": float(action_mae.item()),
        "action_l2_per_step": float(action_l2_per_step.item()),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate_cfg(
    label: str,
    cfg: TrainPipelineConfig,
    checkpoint_description: str,
    device: torch.device,
    num_workers: int,
    num_samples: int,
    seed: int,
) -> dict[str, object]:
    dataset = make_dataset(cfg)
    if len(dataset) == 0:
        raise RuntimeError("Dataset has zero frames.")

    indices = [idx % len(dataset) for idx in range(num_samples)]
    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_processors(cfg, policy, dataset, device)

    rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for sample_idx, batch in enumerate(dataloader):
        set_forward_seed(seed + sample_idx)
        batch = preprocessor(batch)
        rows.append(evaluate_batch(policy, batch))
    elapsed = time.perf_counter() - started

    metrics = mean_metrics(rows)
    metrics["num_eval_samples"] = float(len(rows))
    metrics["elapsed_s"] = elapsed
    return {
        "label": label,
        "checkpoint": checkpoint_description,
        "metrics": metrics,
        "per_sample": rows,
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root),
        "dataset_num_frames": len(dataset),
    }


def evaluate_checkpoint(
    label: str,
    checkpoint: Path,
    device: torch.device,
    num_workers: int,
    dataset_root: Path | None,
    num_samples: int,
    seed: int,
) -> tuple[dict[str, object], TrainPipelineConfig]:
    cfg = load_cfg(checkpoint, device, num_workers, dataset_root)
    result = evaluate_cfg(
        label=label,
        cfg=cfg,
        checkpoint_description=str(checkpoint),
        device=device,
        num_workers=num_workers,
        num_samples=num_samples,
        seed=seed,
    )
    return result, cfg


def evaluate_zero_shot(
    label: str,
    reference_checkpoint: Path,
    device: torch.device,
    num_workers: int,
    dataset_root: Path | None,
    num_samples: int,
    seed: int,
) -> tuple[dict[str, object], TrainPipelineConfig]:
    cfg = load_cfg(reference_checkpoint, device, num_workers, dataset_root)
    cfg.policy.pretrained_path = None
    result = evaluate_cfg(
        label=label,
        cfg=cfg,
        checkpoint_description=f"zero_shot_config_from={reference_checkpoint}",
        device=device,
        num_workers=num_workers,
        num_samples=num_samples,
        seed=seed,
    )
    return result, cfg


def build_pair_delta(current: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    current_metrics = current["metrics"]
    baseline_metrics = baseline["metrics"]
    delta: dict[str, float] = {}
    for key in LOWER_IS_BETTER:
        base_value = float(baseline_metrics[key])
        value = float(current_metrics[key])
        delta[f"{key}_delta"] = value - base_value
        delta[f"{key}_ratio"] = value / base_value if base_value else math.nan
        delta[f"{key}_relative_change_pct"] = ((value - base_value) / base_value) * 100.0 if base_value else math.nan
    return delta


def build_deltas(results: dict[str, dict[str, object]], zero_labels: list[str], checkpoint_labels: list[str]):
    deltas: dict[str, dict[str, float]] = {}
    if zero_labels and checkpoint_labels:
        zero = zero_labels[0]
        sft = checkpoint_labels[0]
        deltas[f"{sft}_vs_{zero}"] = build_pair_delta(results[sft], results[zero])
        deltas["sft_vs_zero"] = deltas[f"{sft}_vs_{zero}"]
        grpo = checkpoint_labels[-1]
        deltas[f"{grpo}_vs_{zero}"] = build_pair_delta(results[grpo], results[zero])
        deltas["grpo_vs_zero"] = deltas[f"{grpo}_vs_{zero}"]
    if len(checkpoint_labels) >= 2:
        sft = checkpoint_labels[0]
        grpo = checkpoint_labels[-1]
        deltas[f"{grpo}_vs_{sft}"] = build_pair_delta(results[grpo], results[sft])
        deltas["grpo_vs_sft"] = deltas[f"{grpo}_vs_{sft}"]
    return deltas


def write_summary(path: Path, payload: dict[str, object]) -> None:
    labels = list(payload["results"])
    title = str(payload.get("report_title", "ACT Offline Zero-Shot vs SFT vs GRPO Evaluation"))
    lines = [
        f"# {title}",
        "",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Dataset frames: {payload['dataset_num_frames']}",
        f"- Samples evaluated: {payload['num_samples']}",
        f"- Seed: {payload['seed']}",
        f"- Sample strategy: `{payload['sample_strategy']}`",
        "",
        "## Metrics",
        "",
        "| metric | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    for metric in LOWER_IS_BETTER:
        values = [f"{float(payload['results'][label]['metrics'][metric]):.8f}" for label in labels]
        lines.append(f"| `{metric}` | " + " | ".join(values) + " |")

    lines.extend(["", "## Deltas", ""])
    if payload["deltas"]:
        lines.append("| comparison | metric | delta | ratio | relative_change_pct |")
        lines.append("|---|---|---:|---:|---:|")
        for comparison, metrics in payload["deltas"].items():
            for metric in LOWER_IS_BETTER:
                lines.append(
                    f"| `{comparison}` | `{metric}` | "
                    f"{metrics[metric + '_delta']:.8f} | "
                    f"{metrics[metric + '_ratio']:.8f} | "
                    f"{metrics[metric + '_relative_change_pct']:.4f} |"
                )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are offline reconstruction / action-error metrics computed on fixed samples.",
            "- Lower values are better for all listed metrics.",
            "- This is useful for same-script A/B comparison, but it is not a real robot success rate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zero_specs = [parse_checkpoint_spec(spec) for spec in args.zero_shot]
    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoint]
    zero_labels = [label for label, _ in zero_specs]
    checkpoint_labels = [label for label, _ in checkpoint_specs]

    results: dict[str, dict[str, object]] = {}
    first_cfg: TrainPipelineConfig | None = None
    for label, reference_checkpoint in zero_specs:
        result, cfg = evaluate_zero_shot(
            label=label,
            reference_checkpoint=reference_checkpoint,
            device=device,
            num_workers=args.num_workers,
            dataset_root=args.dataset_root,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        results[label] = result
        first_cfg = first_cfg or cfg

    for label, checkpoint in checkpoint_specs:
        result, cfg = evaluate_checkpoint(
            label=label,
            checkpoint=checkpoint,
            device=device,
            num_workers=args.num_workers,
            dataset_root=args.dataset_root,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        results[label] = result
        first_cfg = first_cfg or cfg

    if first_cfg is None:
        raise RuntimeError("No evaluation targets were provided.")

    policy_type = first_cfg.policy.type
    report_title = (
        "Pi0.5 (pi05) Offline Zero-Shot vs BC-SFT vs GRPO Evaluation"
        if policy_type == "pi05"
        else "ACT Offline Zero-Shot vs SFT vs GRPO Evaluation"
    )

    payload: dict[str, object] = {
        "mode": "act_zero_sft_grpo_offline",
        "report_title": report_title,
        "policy_type": policy_type,
        "results": results,
        "deltas": build_deltas(results, zero_labels, checkpoint_labels),
        "zero_labels": zero_labels,
        "checkpoint_labels": checkpoint_labels,
        "dataset_repo_id": first_cfg.dataset.repo_id,
        "dataset_root": str(first_cfg.dataset.root),
        "dataset_num_frames": next(iter(results.values()))["dataset_num_frames"],
        "num_samples": args.num_samples,
        "seed": args.seed,
        "sample_strategy": "first_n_samples_with_seed_plus_index_forward_rng",
        "device": device.type,
        "lower_is_better": LOWER_IS_BETTER,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(args.output_dir / "summary.md", payload)
    print(json.dumps({label: results[label]["metrics"] for label in results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
