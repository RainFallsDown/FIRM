#!/usr/bin/env python3
"""Offline zero-shot/SFT/GRPO comparison for Haimiandian.

Zero-shot in this script means:
  same Wan2.2 SFT config, load Wan/T5/CLIP/VAE pretrained components, do not
  load the SFT model.safetensors.

Outputs:
  - metrics.json: overall/per-dim action metrics and validation loss summary
  - predictions.npz: raw GT/pred arrays for each evaluated model
  - plots/*.png: pred-vs-GT trajectory plots and metric bars
  - summary.md: short human-readable report
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tianshou.data import Batch
from transformers.feature_extraction_utils import BatchFeature

try:
    from groot.vla.common.utils import get_frames_by_timestamps
except ImportError:
    from groot.vla.common.utils.misc.video_utils import get_frames_by_timestamps


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
DEFAULT_SFT = PROJECT_ROOT / "outputs/wam_firm_sft_20260428/checkpoint-3500"
DEFAULT_DATA = PROJECT_ROOT / "data/haimiandian_50"
DEFAULT_OUT = Path(__file__).resolve().parent / "results"
ACTION_KEYS = ("left_arm", "right_arm", "gripper")
ACTION_SLICES = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "gripper": slice(14, 16),
}
ACTION_DIM = 16


def setup_runtime(disable_compile: bool) -> None:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "offline")

    import transformers.utils.import_utils as transformers_import_utils

    if hasattr(transformers_import_utils, "check_torch_load_is_safe"):
        transformers_import_utils.check_torch_load_is_safe = lambda: None

    if disable_compile:
        torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)


def patch_current_vae_to_sft_run_behavior() -> None:
    """Match the VAE chunk-cache behavior used by the SFT run.

    The current bc-grpo VAE keeps one feat_idx list across chunks; for T > 1 this
    can overrun feat_cache. The SFT-run code resets feat_idx=[0] per chunk.
    """

    from groot.vla.model.dreamzero.modules import wan_video_vae as vae_mod

    def encode_16(self, x, scale):
        feat_map = [None] * self._enc_conv_num
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = self.encoder(x[:, :, :1], feat_cache=feat_map, feat_idx=[0])
        for i in range(1, iter_):
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i],
                feat_cache=feat_map,
                feat_idx=[0],
            )
            out = torch.cat([out, out_], dim=2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def encode_38(self, x, scale):
        feat_map = [None] * self._enc_conv_num
        x = vae_mod.patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = self.encoder(x[:, :, :1], feat_cache=feat_map, feat_idx=[0])
        for i in range(1, iter_):
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i],
                feat_cache=feat_map,
                feat_idx=[0],
            )
            out = torch.cat([out, out_], dim=2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    vae_mod.VideoVAE_.encode = encode_16
    vae_mod.VideoVAE38_.encode = encode_38


class HaimiandianDataset:
    def __init__(self, root: Path, action_horizon: int, loss_video_frames: int):
        self.root = root
        self.action_horizon = action_horizon
        self.loss_video_frames = loss_video_frames
        self.info = json.loads((root / "meta/info.json").read_text())
        self.tasks = self._load_tasks()
        self.episode_lengths = self._load_episode_lengths()
        self._tables: dict[int, pd.DataFrame] = {}

    def _load_tasks(self) -> dict[int, str]:
        tasks: dict[int, str] = {}
        with open(self.root / "meta/tasks.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                tasks[int(item["task_index"])] = str(item.get("task", ""))
        return tasks

    def _load_episode_lengths(self) -> dict[int, int]:
        lengths: dict[int, int] = {}
        with open(self.root / "meta/episodes.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                lengths[int(item["episode_index"])] = int(item["length"])
        return lengths

    def table(self, episode: int) -> pd.DataFrame:
        if episode not in self._tables:
            rel = self.info["data_path"].format(episode_index=episode, episode_chunk=episode // 1000)
            self._tables[episode] = pd.read_parquet(self.root / rel)
        return self._tables[episode]

    def video_path(self, episode: int, video_key: str = "observation.images.head.color") -> Path:
        rel = self.info["video_path"].format(
            video_key=video_key,
            episode_index=episode,
            episode_chunk=episode // int(self.info.get("chunks_size", 1000)),
        )
        return self.root / rel

    def read_frames(self, episode: int, frame_indices: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        path = self.video_path(episode)
        timestamps = df["timestamp"].to_numpy()[frame_indices]
        try:
            frames = get_frames_by_timestamps(
                path.as_posix(),
                timestamps=timestamps,
                video_backend="torchvision_av",
                video_backend_kwargs={},
            )
            if frames.shape[0] != len(timestamps):
                raise RuntimeError(
                    f"torchvision_av returned {frames.shape[0]} frames for {len(timestamps)} timestamps"
                )
        except Exception as exc:
            print(f"[video] torchvision_av failed for {path}: {exc}; falling back to ffmpeg", flush=True)
            frames = get_frames_by_timestamps(
                path.as_posix(),
                timestamps=timestamps,
                video_backend="ffmpeg",
                video_backend_kwargs={},
            )
        return frames.astype(np.uint8)

    def sample_plan(self, num_samples: int, seed: int, stride: int) -> list[dict[str, int]]:
        rng = random.Random(seed)
        candidates: list[dict[str, int]] = []
        margin = max(self.action_horizon, self.loss_video_frames)
        for episode, length in sorted(self.episode_lengths.items()):
            max_start = length - margin - 1
            if max_start <= 0:
                continue
            for frame in range(0, max_start, stride):
                candidates.append({"episode": episode, "frame": frame})
        rng.shuffle(candidates)
        return candidates[:num_samples]

    def raw_obs(self, episode: int, frame: int, mode: str) -> dict[str, Any]:
        df = self.table(episode)
        if mode == "inference":
            video_indices = np.array([frame], dtype=int)
            action_indices = np.arange(frame, frame + self.action_horizon, dtype=int)
        elif mode == "loss":
            video_indices = np.arange(frame, frame + self.loss_video_frames, dtype=int)
            action_indices = np.arange(frame, frame + self.action_horizon, dtype=int)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        task_index = int(df["task_index"].iloc[frame]) if "task_index" in df else 0
        state_parts = {
            "state.left_arm": state[[frame], ACTION_SLICES["left_arm"]],
            "state.right_arm": state[[frame], ACTION_SLICES["right_arm"]],
            "state.gripper": state[[frame], ACTION_SLICES["gripper"]],
        }
        if mode == "inference":
            state_parts = {key: value[None, ...] for key, value in state_parts.items()}
        obs = {
            "video.head": self.read_frames(episode, video_indices, df),
            **state_parts,
            "annotation.task": [self.tasks.get(task_index, "haimiandian_manipulation")],
        }
        if mode == "loss":
            obs.update(
                {
                    "action.left_arm": action[action_indices, ACTION_SLICES["left_arm"]],
                    "action.right_arm": action[action_indices, ACTION_SLICES["right_arm"]],
                    "action.gripper": action[action_indices, ACTION_SLICES["gripper"]],
                }
            )
        return obs

    def gt_action(self, episode: int, frame: int) -> np.ndarray:
        df = self.table(episode)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        return action[frame : frame + self.action_horizon, :ACTION_DIM]


def nested_to_device(batch: Any, device: str, bf16: bool) -> Any:
    if isinstance(batch, dict):
        return {k: nested_to_device(v, device, bf16) for k, v in batch.items()}
    if isinstance(batch, BatchFeature):
        return BatchFeature(data=nested_to_device(dict(batch), device, bf16))
    if torch.is_tensor(batch):
        if bf16 and torch.is_floating_point(batch):
            return batch.to(device=device, dtype=torch.bfloat16)
        return batch.to(device=device)
    return batch


def make_loss_transform(eval_transform):
    loss_transform = copy.deepcopy(eval_transform)
    loss_transform.eval()
    for transform in loss_transform.transforms:
        if transform.__class__.__name__ == "DreamTransform":
            transform.train()
    return loss_transform


def unnormalize_action(eval_transform, action_pred: torch.Tensor) -> np.ndarray:
    split = eval_transform.unapply({"action": action_pred.detach().cpu()})
    parts = []
    for key in ACTION_KEYS:
        value = split[f"action.{key}"]
        if torch.is_tensor(value):
            value = value.detach().cpu().float().numpy()
        parts.append(np.asarray(value))
    out = np.concatenate(parts, axis=-1)
    if out.ndim == 3:
        out = out[0]
    return out[:, :ACTION_DIM].astype(np.float32)


def prepare_inference_input(eval_transform, obs: dict[str, Any], device: str, bf16: bool):
    eval_transform.eval()
    normalized = eval_transform(copy.deepcopy(obs))
    return nested_to_device(normalized, device, bf16)


def prepare_loss_input(loss_transform, obs: dict[str, Any], device: str, bf16: bool):
    normalized = loss_transform(copy.deepcopy(obs))
    for transform in getattr(loss_transform, "transforms", []):
        if transform.__class__.__name__ == "DreamTransform":
            from groot.vla.model.dreamzero.transform.dreamzero_cotrain import collate

            normalized = collate(
                [normalized],
                transform.tokenizer,
                transform.num_views,
                transform.embodiment_tag_mapping,
            )
            break
    return nested_to_device(normalized, device, bf16)


def load_policy_model(args, model_path: Path):
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.HAIMIANDIAN,
        model_path=str(model_path),
        device=args.device,
        tokenizer_path_override=str(REPO_ROOT / "checkpoints/umt5-xxl"),
    )
    model = policy.trained_model
    model.action_head.num_inference_steps = args.inference_steps
    model.action_head.seed = args.seed
    return model, policy.eval_transform, bool(policy.eval_bf16)


def load_sft_model(args):
    return load_policy_model(args, args.sft_model)


def load_grpo_model(args):
    if args.grpo_model is None:
        raise ValueError("--grpo-model is required when evaluating the grpo model")
    return load_policy_model(args, args.grpo_model)


def load_zeroshot_model(args):
    from groot.vla.data.schema import DatasetMetadata
    from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig

    config_dict = json.loads((args.sft_model / "config.json").read_text())
    model_config = VLAConfig.from_dict(config_dict)
    model = VLA(model_config)
    model.eval().requires_grad_(False)

    train_cfg = OmegaConf.load(args.sft_model / "experiment_cfg/conf.yaml")
    metadata = DatasetMetadata.model_validate(
        json.loads((args.sft_model / "experiment_cfg/metadata.json").read_text())["haimiandian"]
    )
    eval_transform = instantiate(train_cfg.transforms.haimiandian)
    eval_transform.set_metadata(metadata)
    eval_transform.eval()

    eval_bf16 = bool(train_cfg.get("eval_bf16", True))
    if eval_bf16:
        model = model.to(dtype=torch.bfloat16)
    model.to(device=args.device)
    model.post_initialize()
    model.action_head.num_inference_steps = args.inference_steps
    model.action_head.seed = args.seed
    return model, eval_transform, eval_bf16


def evaluate_model(label: str, load_fn, args, dataset: HaimiandianDataset, samples: list[dict[str, int]]):
    print(f"[load] {label}", flush=True)
    model, eval_transform, eval_bf16 = load_fn(args)
    loss_transform = make_loss_transform(eval_transform)
    model.eval()

    preds = []
    gts = []
    losses = []
    sample_records = []

    for i, sample in enumerate(samples):
        episode, frame = sample["episode"], sample["frame"]
        gt = dataset.gt_action(episode, frame)
        infer_obs = dataset.raw_obs(episode, frame, mode="inference")
        loss_obs = dataset.raw_obs(episode, frame, mode="loss")

        torch.manual_seed(args.seed + i)
        torch.cuda.manual_seed_all(args.seed + i)
        with torch.no_grad():
            normalized = prepare_inference_input(eval_transform, infer_obs, args.device, eval_bf16)
            pred = model.lazy_joint_video_action_causal(normalized)["action_pred"].float()
            pred_np = unnormalize_action(eval_transform, pred)

        torch.manual_seed(args.seed + 10000 + i)
        torch.cuda.manual_seed_all(args.seed + 10000 + i)
        with torch.no_grad():
            loss_input = prepare_loss_input(loss_transform, loss_obs, args.device, eval_bf16)
            loss_out = model(loss_input)
            loss_item = {
                "loss": float(loss_out["loss"].detach().cpu()),
                "dynamics_loss": float(loss_out["dynamics_loss"].detach().cpu()),
                "action_loss": float(loss_out["action_loss"].detach().cpu()),
            }

        preds.append(pred_np)
        gts.append(gt)
        losses.append(loss_item)
        sample_records.append({"episode": episode, "frame": frame, **loss_item})
        print(
            f"[{label}] {i + 1}/{len(samples)} ep={episode} frame={frame} "
            f"loss={loss_item['loss']:.6f}",
            flush=True,
        )

    del model, eval_transform, loss_transform
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "pred": np.stack(preds),
        "gt": np.stack(gts),
        "losses": losses,
        "samples": sample_records,
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray, losses: list[dict[str, float]]) -> dict[str, Any]:
    err = pred - gt
    mse_per_dim = np.mean(err**2, axis=(0, 1))
    mae_per_dim = np.mean(np.abs(err), axis=(0, 1))
    rmse_per_dim = np.sqrt(mse_per_dim)
    l2_per_step = np.linalg.norm(err, axis=-1)
    loss_summary = {}
    for key in ("loss", "dynamics_loss", "action_loss"):
        values = np.array([x[key] for x in losses], dtype=np.float64)
        loss_summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "overall": {
            "mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "l2_mean_per_timestep": float(l2_per_step.mean()),
            "l2_std_per_timestep": float(l2_per_step.std()),
            "l2_mean_per_trajectory": float(np.linalg.norm(err.reshape(err.shape[0], -1), axis=-1).mean()),
        },
        "per_dim": {
            "mse": mse_per_dim.tolist(),
            "mae": mae_per_dim.tolist(),
            "rmse": rmse_per_dim.tolist(),
            "l2": np.sqrt(np.sum(err**2, axis=(0, 1))).tolist(),
        },
        "by_group": {
            key: {
                "mse": float(np.mean(err[..., sl] ** 2)),
                "mae": float(np.mean(np.abs(err[..., sl]))),
                "rmse": float(np.sqrt(np.mean(err[..., sl] ** 2))),
                "l2_mean_per_timestep": float(np.linalg.norm(err[..., sl], axis=-1).mean()),
            }
            for key, sl in ACTION_SLICES.items()
        },
        "validation_loss": loss_summary,
    }


def plot_trajectories(results: dict[str, dict[str, Any]], out_dir: Path, max_samples: int) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = list(results.keys())
    n_samples = min(max_samples, next(iter(results.values()))["pred"].shape[0])
    for sample_idx in range(n_samples):
        gt = next(iter(results.values()))["gt"][sample_idx]
        fig, axes = plt.subplots(4, 4, figsize=(18, 12), squeeze=False)
        fig.suptitle(f"Sample {sample_idx}: pred vs GT action trajectory", fontsize=14)
        for dim in range(ACTION_DIM):
            ax = axes[dim // 4][dim % 4]
            ax.plot(gt[:, dim], color="black", linewidth=1.2, label="GT")
            for label in labels:
                ax.plot(results[label]["pred"][sample_idx, :, dim], linewidth=1.0, alpha=0.8, label=label)
            ax.set_title(f"dim {dim}", fontsize=9)
            ax.grid(alpha=0.25)
            if dim == 0:
                ax.legend(fontsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(plot_dir / f"sample_{sample_idx:03d}_pred_vs_gt.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    x = np.arange(ACTION_DIM)
    for offset, label in zip(np.linspace(-width / 2, width / 2, len(labels)), labels):
        mse = np.array(results[label]["metrics"]["per_dim"]["mse"])
        ax.bar(x + offset, mse, width / max(len(labels), 1), label=label)
    ax.set_xlabel("action dim")
    ax.set_ylabel("MSE")
    ax.set_title("Per-dimension action MSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "per_dim_mse_compare.png", dpi=180)
    plt.close(fig)


def write_summary(out_dir: Path, args, metrics: dict[str, Any], samples: list[dict[str, int]]) -> None:
    lines = [
        "# Zero-shot vs SFT vs GRPO Comparison",
        "",
        f"- SFT model: `{args.sft_model}`",
        f"- GRPO model: `{args.grpo_model}`",
        "- Zero-shot: same Wan2.2 SFT config, pretrained Wan/T5/CLIP/VAE components only, no SFT weights loaded.",
        f"- Dataset: `{args.dataset}`",
        f"- Samples: {len(samples)}",
        f"- Models evaluated: {', '.join(metrics['models'].keys())}",
        f"- Inference steps: {args.inference_steps}",
        f"- Loss video frames: {args.loss_video_frames}; action horizon: {args.action_horizon}",
        "",
        "## Overall Metrics",
        "",
        "| model | MSE | RMSE | MAE | L2/step | val loss | val action loss | val dynamics loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in metrics["models"].items():
        overall = item["overall"]
        val = item["validation_loss"]
        lines.append(
            f"| {label} | {overall['mse']:.6f} | {overall['rmse']:.6f} | "
            f"{overall['mae']:.6f} | {overall['l2_mean_per_timestep']:.6f} | "
            f"{val['loss']['mean']:.6f} | {val['action_loss']['mean']:.6f} | "
            f"{val['dynamics_loss']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Metrics use the first 16 real action dimensions: left_arm(7), right_arm(7), gripper(2).",
            "- Validation loss calls the model training forward path with GT future actions and reports weighted loss terms.",
            "- The evaluator patches the current bc-grpo VAE chunk-cache indexing to match the VAE implementation used by the SFT run.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--grpo-model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--loss-video-frames", type=int, default=33)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--max-plot-samples", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["zero_shot_same_config", "sft", "grpo"],
        default=["zero_shot_same_config", "sft", "grpo"],
        help="Which models to evaluate.",
    )
    parser.add_argument("--disable-torch-compile", action="store_true", default=True)
    args = parser.parse_args()

    setup_runtime(args.disable_torch_compile)
    patch_current_vae_to_sft_run_behavior()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = HaimiandianDataset(args.dataset, args.action_horizon, args.loss_video_frames)
    samples = dataset.sample_plan(args.num_samples, args.seed, args.sample_stride)
    (args.output_dir / "samples.json").write_text(json.dumps(samples, indent=2))

    loaders = {
        "zero_shot_same_config": load_zeroshot_model,
        "sft": load_sft_model,
        "grpo": load_grpo_model,
    }
    all_results = {
        label: evaluate_model(label, loaders[label], args, dataset, samples)
        for label in args.models
    }

    metrics = {
        "definition": {
            "zero_shot": "same Wan2.2 SFT config; pretrained Wan/T5/CLIP/VAE components; no SFT weights",
            "sft": str(args.sft_model),
            "grpo": str(args.grpo_model) if args.grpo_model is not None else None,
        },
        "models": {},
    }
    npz_payload = {}
    for label, result in all_results.items():
        result["metrics"] = compute_metrics(result["pred"], result["gt"], result["losses"])
        metrics["models"][label] = result["metrics"]
        metrics["models"][label]["samples"] = result["samples"]
        npz_payload[f"{label}_pred"] = result["pred"]
        npz_payload[f"{label}_gt"] = result["gt"]

    np.savez_compressed(args.output_dir / "predictions.npz", **npz_payload)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_trajectories(all_results, args.output_dir, args.max_plot_samples)
    write_summary(args.output_dir, args, metrics, samples)
    print(f"[done] wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
