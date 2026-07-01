#!/usr/bin/env python3
"""Extract Wan2.2 VAE latents for a LeRobot-style LingBot-VA dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--camera-keys", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mirror-episode-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also mirror each output to chunk-{episode_index:03d} for v3-style datasets.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def lerobot_chunk(episode_index: int, info: dict[str, Any]) -> int:
    chunks_size = int(info.get("chunks_size") or 1000)
    return episode_index // max(chunks_size, 1)


def find_video_path(dataset_path: Path, camera_key: str, episode_index: int, chunk: int) -> Path:
    candidates = [
        dataset_path / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{episode_index:03d}.mp4",
        dataset_path / "videos" / camera_key / f"chunk-{episode_index:03d}" / "file-000.mp4",
        dataset_path / "videos" / camera_key / "chunk-000" / f"file-{episode_index:03d}.mp4",
        dataset_path / "videos" / camera_key / f"chunk-{episode_index:03d}" / f"file-{episode_index:03d}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find video for camera={camera_key}, episode={episode_index}; tried: "
        + ", ".join(str(v) for v in candidates)
    )


def make_frame_ids(start: int, end: int, ori_fps: float, target_fps: int) -> list[int]:
    stride = max(1, round(float(ori_fps) / float(target_fps)))
    frame_ids = list(range(start, end, stride))
    if not frame_ids:
        frame_ids = [start]
    valid_len = ((len(frame_ids) - 1) // 4) * 4 + 1
    return frame_ids[:valid_len]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mirror_file(src: Path, dst: Path, force: bool) -> None:
    if src == dst:
        return
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def lazy_import_runtime():
    import imageio.v3 as iio
    import numpy as np
    import torch
    import torch.nn.functional as F

    from wan_va.modules import load_text_encoder, load_tokenizer, load_vae

    return iio, np, torch, F, load_text_encoder, load_tokenizer, load_vae


def torch_dtype(torch_module, name: str):
    if name == "bf16":
        return torch_module.bfloat16
    if name == "fp16":
        return torch_module.float16
    return torch_module.float32


def encode_text(text: str, tokenizer, text_encoder, torch, device: str, dtype, max_sequence_length: int):
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        embeds = text_encoder(input_ids, mask).last_hidden_state.to(dtype=dtype)
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in embeds],
        dim=0,
    )
    return embeds[0].detach().cpu()


def read_selected_frames(video_path: Path, frame_ids: list[int], iio, np):
    wanted = set(frame_ids)
    frames = []
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx in wanted:
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)
            frames.append(frame[..., :3])
        if idx > frame_ids[-1]:
            break
    if len(frames) != len(frame_ids):
        raise RuntimeError(f"{video_path} yielded {len(frames)} frames, expected {len(frame_ids)}")
    return np.stack(frames, axis=0)


def encode_video(frames, vae, torch, F, device: str, dtype, height: int, width: int):
    video = torch.from_numpy(frames).float().permute(3, 0, 1, 2)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False).unsqueeze(0)
    video = video / 255.0 * 2.0 - 1.0
    with torch.no_grad():
        posterior = vae.encode(video.to(device=device, dtype=dtype)).latent_dist
        mu = posterior.mean
        latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
        mu_norm = ((mu.float() - latents_mean) / latents_std).to(dtype)
    latent = mu_norm[0].permute(1, 2, 3, 0).reshape(-1, mu_norm.shape[1])
    return latent.detach().cpu(), tuple(mu_norm.shape[-3:])


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    model_path = args.model_path.resolve()
    info = load_json(dataset_path / "meta" / "info.json")
    episodes = load_episodes(dataset_path / "meta" / "episodes.jsonl")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    ori_fps = float(info.get("fps") or 30)
    plan_count = 0
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        chunk = lerobot_chunk(ep_idx, info)
        for segment in episode.get("action_config", []):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            for camera_key in args.camera_keys:
                find_video_path(dataset_path, camera_key, ep_idx, chunk)
                out = (
                    dataset_path
                    / "latents"
                    / f"chunk-{chunk:03d}"
                    / camera_key
                    / f"episode_{ep_idx:06d}_{start}_{end}.pth"
                )
                plan_count += 1
                if args.dry_run:
                    print(out)
    if args.dry_run:
        print(f"dry_run_ok files_planned={plan_count}")
        return 0

    iio, np, torch, F, load_text_encoder, load_tokenizer, load_vae = lazy_import_runtime()
    dtype = torch_dtype(torch, args.dtype)
    tokenizer = load_tokenizer(model_path / "tokenizer")
    text_encoder = load_text_encoder(model_path / "text_encoder", torch_dtype=dtype, torch_device=args.device).eval()
    vae = load_vae(model_path / "vae", torch_dtype=dtype, torch_device=args.device).eval()

    empty_emb = encode_text("", tokenizer, text_encoder, torch, args.device, dtype, args.max_sequence_length)
    torch.save(empty_emb.to(torch.bfloat16), dataset_path / "empty_emb.pt")

    done = 0
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        chunk = lerobot_chunk(ep_idx, info)
        for segment in episode.get("action_config", []):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            text = str(segment.get("action_text") or " ".join(episode.get("tasks") or [""]))
            text_emb = encode_text(text, tokenizer, text_encoder, torch, args.device, dtype, args.max_sequence_length)
            frame_ids = make_frame_ids(start, end, ori_fps, args.target_fps)
            for camera_key in args.camera_keys:
                out = (
                    dataset_path
                    / "latents"
                    / f"chunk-{chunk:03d}"
                    / camera_key
                    / f"episode_{ep_idx:06d}_{start}_{end}.pth"
                )
                if out.exists() and not args.force:
                    continue
                video_path = find_video_path(dataset_path, camera_key, ep_idx, chunk)
                frames = read_selected_frames(video_path, frame_ids, iio, np)
                camera_index = args.camera_keys.index(camera_key)
                height = args.height if camera_index == 0 else args.height // 2
                width = args.width if camera_index == 0 else args.width // 2
                latent, latent_shape = encode_video(frames, vae, torch, F, args.device, dtype, height, width)
                latent_frames, latent_height, latent_width = latent_shape
                payload = {
                    "latent": latent.to(torch.bfloat16),
                    "latent_num_frames": int(latent_frames),
                    "latent_height": int(latent_height),
                    "latent_width": int(latent_width),
                    "video_num_frames": int(len(frame_ids)),
                    "video_height": int(height),
                    "video_width": int(width),
                    "text_emb": text_emb.to(torch.bfloat16),
                    "text": text,
                    "frame_ids": frame_ids,
                    "start_frame": start,
                    "end_frame": end,
                    "fps": int(args.target_fps),
                    "ori_fps": int(round(ori_fps)),
                }
                ensure_parent(out)
                torch.save(payload, out)
                if args.mirror_episode_chunk:
                    mirror = (
                        dataset_path
                        / "latents"
                        / f"chunk-{ep_idx:03d}"
                        / camera_key
                        / f"episode_{ep_idx:06d}_{start}_{end}.pth"
                    )
                    mirror_file(out, mirror, args.force)
                done += 1
                print(f"saved {done}/{plan_count}: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
