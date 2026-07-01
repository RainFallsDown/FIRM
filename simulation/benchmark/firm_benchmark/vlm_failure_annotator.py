#!/usr/bin/env python3
"""
VLM semantic failure annotator for FIRM.

Input:
    raw_episode_metrics_mask.jsonl

Episode images:
    episodes/
    └── episode_000000/
        ├── final_frames/
        └── sampled_frames/

Output:
    raw_episode_metrics_vlm.jsonl

Environment:
    OPENAI_API_KEY required
    OPENAI_BASE_URL optional for OpenAI-compatible endpoints
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_idx}: {e}") from e
    return records


def write_jsonl(records: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    mime = "image/png"
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def uniform_select(files: List[Path], max_items: int) -> List[Path]:
    files = sorted(files)
    if len(files) <= max_items:
        return files
    if max_items <= 1:
        return [files[-1]]
    idxs = [round(i * (len(files) - 1) / (max_items - 1)) for i in range(max_items)]
    return [files[i] for i in idxs]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(folder.glob(ext))
        files.extend(folder.glob(f"**/{ext}"))
    return sorted(set(files))


def select_episode_images(episode_dir: Path, max_images: int) -> List[Path]:
    final_images = list_images(episode_dir / "final_frames")
    sampled_images = list_images(episode_dir / "sampled_frames")

    images: List[Path] = []
    images.extend(uniform_select(final_images, min(len(final_images), 4)))

    remaining = max_images - len(images)
    if remaining > 0:
        images.extend(uniform_select(sampled_images, remaining))

    return images[:max_images]


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON found in VLM response:\n{text}")

    return json.loads(match.group(0))


def build_prompt(task: str) -> str:
    task_notes = {
        "Manual": "Focus on incomplete insertion, severe folding, skewed placement, protrusion, pinching, or trapped corners.",
        "Cable": "Focus on residual tangling, cable outside target region, snagging, slipping, or failed routing.",
        "Sponge": "Focus on folded corners, trapped corners, residual compression, rebound displacement, shifted placement, or dropping.",
        "Tape": "Focus on rolling, wrong orientation, face-down state, on-lid placement, outside-target placement, or slipping.",
        "Box": "Focus on incomplete folding, hinge jamming, collision, severe deformation, or failure to close.",
    }

    return f"""
You are annotating an industrial flexible-object robot manipulation episode for the FIRM benchmark.

Task category: {task}
Task-specific focus: {task_notes.get(task, "Focus on deformation, contact instability, and placement failure.")}

Return ONLY valid JSON. Do not include markdown or extra explanation.

Required JSON schema:
{{
  "folded_corner": boolean,
  "trapped_corner": boolean,
  "rolling": boolean,
  "face_down": boolean,
  "outside_target": boolean,
  "on_lid": boolean,
  "slip": boolean,
  "jammed": boolean,
  "dropped": boolean,
  "severe_fold": boolean,
  "residual_tangling": number,
  "failure_mode_hint": string,
  "confidence": number,
  "explanation": string
}}

Rules:
- Do not penalize normal deformation needed for the task.
- Mark only out-of-tolerance deformation or unstable contact.
- residual_tangling must be in [0, 1].
- confidence must be in [0, 1].
- failure_mode_hint must be one of:
  grasp_failure,
  alignment_failure,
  incomplete_insertion_or_placement,
  out_of_tolerance_deformation,
  slip_or_rolling,
  contact_instability,
  collision_or_jamming,
  over_compression,
  timeout,
  other.
""".strip()


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def call_vlm(client: OpenAI, model: str, task: str, images: List[Path], temperature: float) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": build_prompt(task)}]

    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(img)},
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )

    text = response.choices[0].message.content
    if text is None:
        raise RuntimeError("Empty VLM response.")

    return extract_json(text)


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"true", "1", "yes", "y"}
    return bool(x)


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))


def normalize_flags(raw: Dict[str, Any]) -> Dict[str, Any]:
    bool_keys = [
        "folded_corner",
        "trapped_corner",
        "rolling",
        "face_down",
        "outside_target",
        "on_lid",
        "slip",
        "jammed",
        "dropped",
        "severe_fold",
    ]

    out: Dict[str, Any] = {}
    for k in bool_keys:
        out[k] = to_bool(raw.get(k, False))

    out["residual_tangling"] = clamp01(raw.get("residual_tangling", 0.0))

    valid_modes = {
        "grasp_failure",
        "alignment_failure",
        "incomplete_insertion_or_placement",
        "out_of_tolerance_deformation",
        "slip_or_rolling",
        "contact_instability",
        "collision_or_jamming",
        "over_compression",
        "timeout",
        "other",
    }

    mode = str(raw.get("failure_mode_hint", "other")).strip()
    if mode not in valid_modes:
        mode = "other"

    out["failure_mode_hint"] = mode
    out["vlm_confidence"] = clamp01(raw.get("confidence", 0.0))
    out["vlm_explanation"] = str(raw.get("explanation", ""))[:500]

    return out


def annotate(
    records: List[Dict[str, Any]],
    episodes_root: Path,
    model: str,
    max_images: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    client = make_client()
    outputs = []

    for record in records:
        episode_id = record["episode_id"]
        task = record["task"]
        episode_dir = episodes_root / episode_id

        if not episode_dir.exists():
            print(f"[WARN] Missing episode dir: {episode_dir}")
            outputs.append(record)
            continue

        images = select_episode_images(episode_dir, max_images=max_images)
        if not images:
            print(f"[WARN] No images found for {episode_id}")
            outputs.append(record)
            continue

        try:
            raw = call_vlm(
                client=client,
                model=model,
                task=task,
                images=images,
                temperature=temperature,
            )
            flags = normalize_flags(raw)

            record.setdefault("metrics", {})
            record["metrics"].update(flags)

            record.setdefault("metadata", {})
            record["metadata"]["vlm_model"] = model
            record["metadata"]["vlm_images"] = [str(p) for p in images]
            record["metadata"]["vlm_raw_response"] = raw

            print(f"[OK] {episode_id}: {flags['failure_mode_hint']} conf={flags['vlm_confidence']:.2f}")

        except Exception as exc:
            print(f"[WARN] VLM failed for {episode_id}: {exc}")

        outputs.append(record)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    outputs = annotate(
        records=records,
        episodes_root=args.episodes_root,
        model=args.model,
        max_images=args.max_images,
        temperature=args.temperature,
    )
    write_jsonl(outputs, args.output)

    print(f"[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()