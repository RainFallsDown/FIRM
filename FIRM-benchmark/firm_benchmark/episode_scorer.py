#!/usr/bin/env python3
"""
FIRM episode-level scorer.

This script converts raw task-level measurements into DAP episode annotations.

Input:
    raw_episode_metrics.jsonl

Output:
    scored_annotations.jsonl

Each output line contains:
    success
    completion_quality
    q_def.score
    failure_mode

The output can be passed to dap_eval.py for SR/CQ/DQ aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# -----------------------------
# Basic utilities
# -----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or abs(float(b)) < 1e-8:
        return default
    return float(a) / float(b)


def get_float(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = d.get(key, default)
    if value is None:
        return default
    return float(value)


def get_bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = d.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}
    return bool(value)


def require_task_config(config: Dict[str, Any], task: str) -> Dict[str, Any]:
    if "tasks" not in config:
        raise KeyError("Config must contain a top-level `tasks` field.")
    if task not in config["tasks"]:
        raise KeyError(f"Missing config for task `{task}`.")
    return config["tasks"][task]


# -----------------------------
# Failure mode inference
# -----------------------------

def infer_failure_mode(metrics: Dict[str, Any], completion: float, q_def: float) -> Optional[str]:
    """
    Failure modes follow the DAP taxonomy:
        grasp failure
        alignment failure
        incomplete insertion or placement
        out-of-tolerance deformation
        slip or rolling
        contact instability
        collision or jamming
        over-compression
        timeout
    """

    if get_bool(metrics, "timeout"):
        return "timeout"

    if get_bool(metrics, "grasp_failure") or get_bool(metrics, "dropped"):
        return "grasp_failure"

    if get_bool(metrics, "collision") or get_bool(metrics, "jammed") or get_bool(metrics, "hinge_jam"):
        return "collision_or_jamming"

    if get_bool(metrics, "slip") or get_bool(metrics, "rolling"):
        return "slip_or_rolling"

    if get_bool(metrics, "contact_instability"):
        return "contact_instability"

    if get_bool(metrics, "over_compression"):
        return "over_compression"

    if q_def < 0.5:
        return "out_of_tolerance_deformation"

    alignment_error = metrics.get("alignment_error", None)
    alignment_threshold = metrics.get("alignment_threshold", None)
    if alignment_error is not None and alignment_threshold is not None:
        if float(alignment_error) > float(alignment_threshold):
            return "alignment_failure"

    if completion < 1.0:
        return "incomplete_insertion_or_placement"

    return None


def make_output(success: bool, completion: float, q_def: float, failure_mode: Optional[str]) -> Dict[str, Any]:
    output = {
        "success": bool(success),
        "completion_quality": clamp01(completion),
        "q_def": {
            "score": clamp01(q_def)
        },
    }

    if not success:
        output["failure_mode"] = failure_mode or "other"

    return output


# -----------------------------
# Task-specific scorers
# -----------------------------

def score_manual(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Instruction manual insertion.

    Completion variables:
        object_target_overlap_ratio

    Deformation/contact variables:
        severe_fold
        jammed
        dropped
    """

    object_in_target = get_float(metrics, "object_target_overlap_ratio")
    alignment_error = get_float(metrics, "alignment_error", 1e9)
    max_alignment_error = get_float(cfg, "max_alignment_error", 0.05)
    align_score = clamp01(1.0 - safe_div(alignment_error, max_alignment_error))

    w_inside = get_float(cfg, "w_inside", 0.95)
    w_align = get_float(cfg, "w_align", 0.05)
    completion = clamp01(w_inside * object_in_target + w_align * align_score)

    severe_fold = get_bool(metrics, "severe_fold")
    jammed = get_bool(metrics, "jammed")
    dropped = get_bool(metrics, "dropped")
    timeout = get_bool(metrics, "timeout")
    safety_violation = get_bool(metrics, "safety_violation")
    final_compactness = get_float(metrics, "final_compactness", 1.0)
    dq_fold_warning_compactness = get_float(cfg, "dq_fold_warning_compactness", 0.08)
    dq_w_fold = get_float(cfg, "dq_w_fold", 0.7)
    dq_w_contact = get_float(cfg, "dq_w_contact", 0.3)

    fold_penalty = 0.0
    if severe_fold:
        fold_penalty = 1.0
    elif final_compactness < dq_fold_warning_compactness:
        fold_penalty = clamp01(
            safe_div(
                dq_fold_warning_compactness - final_compactness,
                dq_fold_warning_compactness,
            )
        )

    contact_penalty = float(jammed or dropped)
    q_def = 1.0 - clamp01(dq_w_fold * fold_penalty + dq_w_contact * contact_penalty)

    min_object_in_target_ratio = get_float(
        cfg,
        "min_object_in_target_ratio",
        get_float(cfg, "target_insertion_depth", 0.98),
    )

    success = (
        object_in_target >= min_object_in_target_ratio
        and not severe_fold
        and not jammed
        and not dropped
        and not timeout
        and not safety_violation
    )

    failure_mode = infer_failure_mode(metrics, completion, q_def)
    return make_output(success, completion, q_def, failure_mode)


def score_cable(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cable manipulation.

    Completion variables:
        contained_cable_length

    Deformation/contact variables:
        residual_tangling
        slip
        dropped
    """

    contained_length = get_float(metrics, "contained_cable_length")
    completion = clamp01(contained_length)

    tangling = get_float(metrics, "residual_tangling")
    boundary_contact = get_float(metrics, "boundary_contact")
    slip = get_bool(metrics, "slip")
    dropped = get_bool(metrics, "dropped")
    timeout = get_bool(metrics, "timeout")

    dq_tangling_warning = get_float(cfg, "dq_tangling_warning", 0.90)
    dq_w_tangling = get_float(cfg, "dq_w_tangling", 0.4)
    dq_w_contact = get_float(cfg, "dq_w_contact", 0.6)

    tangling_penalty = 0.0
    if tangling > dq_tangling_warning:
        tangling_penalty = clamp01(
            safe_div(tangling - dq_tangling_warning, 1.0 - dq_tangling_warning)
        )
    contact_penalty = clamp01(float(boundary_contact > 0.0) + float(slip or dropped))
    q_def = 1.0 - clamp01(dq_w_tangling * tangling_penalty + dq_w_contact * contact_penalty)

    min_contained_length = get_float(
        cfg,
        "min_contained_length",
        get_float(cfg, "success_completion", 0.98),
    )

    success = (
        contained_length >= min_contained_length
        and not slip
        and not dropped
        and not timeout
    )

    failure_mode = infer_failure_mode(metrics, completion, q_def)
    return make_output(success, completion, q_def, failure_mode)


def score_box(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Box folding.

    Completion variables:
        object_target_overlap_ratio
        fold_angle

    Deformation/contact variables:
        hinge_jam
        collision
        severe_deformation
    """

    object_in_target = get_float(metrics, "object_target_overlap_ratio")
    fold_angle = get_float(metrics, "fold_angle")
    target_angle = get_float(cfg, "target_fold_angle", 90.0)
    max_angle_error = get_float(cfg, "max_angle_error", 15.0)

    angle_error = abs(fold_angle - target_angle)
    angle_score = clamp01(1.0 - safe_div(angle_error, max_angle_error))
    w_inside = get_float(cfg, "w_inside", 0.95)
    w_angle = get_float(cfg, "w_angle", 0.05)
    completion = clamp01(w_inside * object_in_target + w_angle * angle_score)

    hinge_jam = get_bool(metrics, "hinge_jam")
    collision = get_bool(metrics, "collision")
    severe_deformation = get_bool(metrics, "severe_deformation")
    timeout = get_bool(metrics, "timeout")

    final_compactness = get_float(metrics, "final_compactness", 1.0)
    dq_deform_warning_compactness = get_float(cfg, "dq_deform_warning_compactness", 0.08)
    dq_w_deform = get_float(cfg, "dq_w_deform", 0.7)
    dq_w_contact = get_float(cfg, "dq_w_contact", 0.3)

    deform_penalty = 0.0
    if severe_deformation:
        deform_penalty = 1.0
    elif final_compactness < dq_deform_warning_compactness:
        deform_penalty = clamp01(
            safe_div(
                dq_deform_warning_compactness - final_compactness,
                dq_deform_warning_compactness,
            )
        )
    contact_penalty = float(hinge_jam or collision)
    q_def = 1.0 - clamp01(dq_w_deform * deform_penalty + dq_w_contact * contact_penalty)

    min_object_in_target_ratio = get_float(
        cfg,
        "min_object_in_target_ratio",
        get_float(cfg, "success_completion", 0.98),
    )

    success = (
        object_in_target >= min_object_in_target_ratio
        and not hinge_jam
        and not collision
        and not timeout
    )

    failure_mode = infer_failure_mode(metrics, completion, q_def)
    return make_output(success, completion, q_def, failure_mode)


def score_sponge(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sponge / flat-object placement.

    Completion variables:
        object_target_overlap_ratio
        pose_error

    Deformation/contact variables:
        folded_corner
        trapped_corner
        residual_compression
        rebound_shift
    """

    object_in_target = get_float(metrics, "object_target_overlap_ratio")
    pose_error = get_float(metrics, "pose_error", 1e9)
    max_pose_error = get_float(cfg, "max_pose_error", 0.05)

    pose_score = clamp01(1.0 - safe_div(pose_error, max_pose_error))

    w_inside = get_float(cfg, "w_inside", get_float(cfg, "w_coverage", 0.7))
    w_pose = get_float(cfg, "w_pose", 0.3)
    completion = clamp01(w_inside * object_in_target + w_pose * pose_score)

    folded_corner = get_bool(metrics, "folded_corner")
    trapped_corner = get_bool(metrics, "trapped_corner")
    residual_compression = get_float(metrics, "residual_compression")
    rebound_shift = get_float(metrics, "rebound_shift")
    dropped = get_bool(metrics, "dropped")
    jammed = get_bool(metrics, "jammed")
    timeout = get_bool(metrics, "timeout")

    max_residual_compression = get_float(cfg, "max_residual_compression", 0.15)
    max_rebound_shift = get_float(cfg, "max_rebound_shift", 0.05)

    e_tol = clamp01(
        (
            float(folded_corner)
            + float(trapped_corner)
            + safe_div(residual_compression, max_residual_compression)
            + safe_div(rebound_shift, max_rebound_shift)
        ) / 4.0
    )
    e_contact = float(dropped or jammed)

    q_def = 1.0 - clamp01(0.7 * e_tol + 0.3 * e_contact)

    min_object_in_target_ratio = get_float(
        cfg,
        "min_object_in_target_ratio",
        get_float(cfg, "success_completion", 0.80),
    )

    success = (
        object_in_target >= min_object_in_target_ratio
        and pose_error <= max_pose_error
        and not folded_corner
        and not trapped_corner
        and residual_compression <= max_residual_compression
        and rebound_shift <= max_rebound_shift
        and not dropped
        and not jammed
        and not timeout
    )

    failure_mode = infer_failure_mode(metrics, completion, q_def)
    return make_output(success, completion, q_def, failure_mode)


def score_tape(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tape placement.

    Completion variables:
        object_target_overlap_ratio

    Deformation/contact variables:
        rolling
        slip
    """

    object_in_target = get_float(metrics, "object_target_overlap_ratio")
    completion = clamp01(object_in_target)

    rolling = get_bool(metrics, "rolling")
    slip = get_bool(metrics, "slip")
    timeout = get_bool(metrics, "timeout")

    q_def = 1.0 - clamp01(float(rolling or slip))

    min_object_in_target_ratio = get_float(
        cfg,
        "min_object_in_target_ratio",
        get_float(cfg, "success_completion", 0.80),
    )

    success = (
        object_in_target >= min_object_in_target_ratio
        and not timeout
    )

    failure_mode = infer_failure_mode(metrics, completion, q_def)
    return make_output(success, completion, q_def, failure_mode)


SCORERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "Manual": score_manual,
    "Cable": score_cable,
    "Box": score_box,
    "Sponge": score_sponge,
    "Tape": score_tape,
}


# -----------------------------
# Record scoring
# -----------------------------

def score_record(record: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if "episode_id" not in record:
        raise KeyError("Each record must contain `episode_id`.")
    if "task" not in record:
        raise KeyError("Each record must contain `task`.")
    if "metrics" not in record:
        raise KeyError("Each record must contain `metrics`.")

    task = record["task"]
    if task not in SCORERS:
        raise ValueError(f"Unknown task `{task}`. Expected one of {list(SCORERS.keys())}.")

    task_cfg = require_task_config(config, task)
    scored = SCORERS[task](record["metrics"], task_cfg)

    output = {
        "episode_id": record["episode_id"],
        "method": record.get("method", "unknown_method"),
        "task": task,
        **scored,
    }

    if "perturbation" in record:
        output["perturbation"] = record["perturbation"]

    if "metadata" in record:
        output["metadata"] = record["metadata"]

    return output


# -----------------------------
# I/O
# -----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_id}: {e}") from e

    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_config(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid config JSON: {e}") from e


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw FIRM episode metrics into DAP episode annotations."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input raw episode metrics JSONL.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Task-specific DAP tolerance config JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output scored annotations JSONL.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_jsonl(args.input)

    scored = [score_record(record, config) for record in records]
    write_jsonl(scored, args.output)

    print(f"[OK] Scored {len(scored)} episodes.")
    print(f"[OK] Wrote: {args.output}")


if __name__ == "__main__":
    main()
