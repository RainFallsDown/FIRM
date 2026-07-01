#!/usr/bin/env python3
"""Offline LingBot-VA haimiandian zero-shot/SFT evaluator.

This evaluator compares checkpoints through the training forward path:
video/action flow-matching target prediction losses on the same latent dataset.
It does not require a robot/simulation client and is intended as a stable
zero-shot vs SFT A/B metric.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.modules import load_transformer
from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, sample_timestep_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["zeroshot", "sft"], required=True)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--config-name", default="haimiandian_train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--attn-mode", default="torch")
    parser.add_argument("--max-new-metas", type=int, default=None)
    return parser.parse_args()


def to_device_batch(sample: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not torch.is_tensor(value):
            continue
        batch[key] = value.unsqueeze(0).to(device)
    return batch


class ForwardEvaluator:
    def __init__(self, config, model_root: Path, device: torch.device, attn_mode: str):
        self.config = config
        self.device = device
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        transformer_path = model_root / "transformer"
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=self.dtype,
            torch_device=device,
            attn_mode=attn_mode,
        )
        self.transformer.eval().requires_grad_(False)

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=None, action_mode=False, noisy_cond_prob=0.0):
        batch_size, _channels, frames, height, width = latent.shape
        timestep_ids = sample_timestep_id(
            batch_size=frames,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)
        grid_id = grid_id[None].repeat(batch_size, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=frames,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            mask = action_mask.float()
            noisy_latents *= mask
            targets *= mask
            latent *= mask

        return {
            "timesteps": timesteps[None].repeat(batch_size, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
            "grid_id": grid_id,
        }

    @torch.no_grad()
    def prepare_input_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, object]:
        latent_dict = self._add_noise(
            latent=batch["latents"],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.0,
        )
        action_dict = self._add_noise(
            latent=batch["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )
        latent_dict["text_emb"] = batch["text_emb"]
        action_dict["text_emb"] = batch["text_emb"]
        action_dict["actions_mask"] = batch["actions_mask"]
        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": 1,
            "window_size": 16,
        }

    @torch.no_grad()
    def evaluate_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        input_dict = self.prepare_input_dict(batch)
        latent_pred, action_pred = self.transformer(input_dict, train_mode=True)
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_loss = self._weighted_frame_mse(
            latent_pred,
            input_dict["latent_dict"]["targets"],
            input_dict["latent_dict"]["timesteps"],
            self.train_scheduler_latent,
            mask=None,
        )
        action_loss = self._weighted_frame_mse(
            action_pred,
            input_dict["action_dict"]["targets"],
            input_dict["action_dict"]["timesteps"],
            self.train_scheduler_action,
            mask=input_dict["action_dict"]["actions_mask"],
        )

        action_mask = input_dict["action_dict"]["actions_mask"].float()
        action_err = (action_pred.float() - input_dict["action_dict"]["targets"].float()) * action_mask
        denom = action_mask.sum().clamp_min(1.0)
        action_mse = (action_err.pow(2).sum() / denom).item()
        action_mae = (action_err.abs().sum() / denom).item()
        action_rmse = math.sqrt(action_mse)

        per_step = action_err.pow(2).sum(dim=(1, 3, 4)).sqrt()
        valid_step = action_mask.sum(dim=(1, 3, 4)) > 0
        l2_step = per_step[valid_step].mean().item() if valid_step.any() else 0.0

        video_err = latent_pred.float() - input_dict["latent_dict"]["targets"].float()
        video_mse = video_err.pow(2).mean().item()
        video_mae = video_err.abs().mean().item()

        return {
            "val_loss": latent_loss + action_loss,
            "val_dynamics_loss": latent_loss,
            "val_action_loss": action_loss,
            "action_noise_mse": action_mse,
            "action_noise_rmse": action_rmse,
            "action_noise_mae": action_mae,
            "action_noise_l2_per_step": l2_step,
            "video_noise_mse": video_mse,
            "video_noise_mae": video_mae,
        }

    def _weighted_frame_mse(self, pred, target, timesteps, scheduler, mask=None) -> float:
        batch_frames, frames = timesteps.shape
        weight = scheduler.training_weight(timesteps.flatten()).reshape(batch_frames, frames)
        loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
        loss = loss * weight[:, None, :, None, None]
        if mask is not None:
            mask_f = mask.float()
            loss = loss * mask_f
            denom = mask_f.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
        else:
            denom = torch.ones_like(loss).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
        loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).sum(dim=1)
        return (loss / (denom + 1e-6)).mean().item()


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_summary(path: Path, args: argparse.Namespace, metrics: dict[str, float], dataset_len: int) -> None:
    lines = [
        "# LingBot-VA Haimiandian Offline Evaluation",
        "",
        f"- Mode: `{args.mode}`",
        f"- Model root: `{args.model_root}`",
        f"- Checkpoint: `{args.checkpoint or 'pretrained'}`",
        f"- Dataset: `{args.dataset_path}`",
        f"- Dataset segments: {dataset_len}",
        f"- Samples evaluated: {args.num_samples}",
        f"- Seed: {args.seed}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in sorted(metrics):
        lines.append(f"| {key} | {metrics[key]:.8f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics are computed through the LingBot-VA training forward path on extracted latents.",
            "- `action_noise_*` compares predicted flow-matching action noise target under the action mask.",
            "- This is an offline A/B metric, not a real-robot success-rate evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.dataset_path = str(args.dataset_path)
    config.empty_emb_path = str(args.dataset_path / "empty_emb.pt")
    config.wan22_pretrained_model_name_or_path = str(args.model_root)
    config.cfg_prob = 0.0
    config.load_worker = 0
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=1)
    if args.max_new_metas is not None:
        for dset in dataset._datasets:
            dset.new_metas = dset.new_metas[: args.max_new_metas]

    if len(dataset) == 0:
        raise RuntimeError("Dataset has zero usable latent segments. Check latents/ and empty_emb.pt.")

    evaluator = ForwardEvaluator(config, args.model_root, device, args.attn_mode)

    rows: list[dict[str, float]] = []
    for idx in tqdm(range(args.num_samples), desc=f"eval-{args.mode}"):
        torch.manual_seed(args.seed + idx)
        sample = dataset[idx % len(dataset)]
        batch = to_device_batch(sample, device)
        rows.append(evaluator.evaluate_batch(batch))

    metrics = mean_metrics(rows)
    payload = {
        "mode": args.mode,
        "model_root": str(args.model_root),
        "checkpoint": args.checkpoint,
        "dataset_path": str(args.dataset_path),
        "dataset_segments": len(dataset),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "metrics": metrics,
        "per_sample": rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.md", args, metrics, len(dataset))
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
