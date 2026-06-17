#!/usr/bin/env python3
"""
FIRM DAP benchmark evaluator.

This script implements the Deformation-aware Assessment Protocol (DAP)
defined in the FIRM paper.

Metrics:
    S_succ     : binary success
    S_comp     : completion quality in [0, 1]
    Q_def      : deformation-aware execution quality
    S_robust   : average completion quality under perturbations
    PR-AUC     : area under success-perturbation curve
    P(f_j)     : failure-mode distribution

The evaluator intentionally keeps task-specific thresholds outside the code.
Industrial acceptance conditions, ordinal labels, deformation labels, and
failure annotations should be provided by the annotation files or config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Default label mappings
# -----------------------------

COMPLETION_ORDINAL_MAP = {
    "none": 0.0,
    "no_meaningful_completion": 0.0,
    "failure": 0.0,
    "partial": 0.5,
    "suboptimal": 0.5,
    "partial_or_suboptimal": 0.5,
    "complete": 1.0,
    "success": 1.0,
}

QDEF_LABEL_MAP = {
    "acceptable": 1.0,
    "minor_defect": 2.0 / 3.0,
    "minor defect": 2.0 / 3.0,
    "major_defect": 1.0 / 3.0,
    "major defect": 1.0 / 3.0,
    "catastrophic_failure": 0.0,
    "catastrophic failure": 0.0,
}

FAILURE_MODES = [
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
]


# -----------------------------
# Utility functions
# -----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_ci(
    values: List[float],
    num_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """Empirical bootstrap CI for a mean."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None

    rng = random.Random(seed)
    means = []
    n = len(values)

    for _ in range(num_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)

    return (
        percentile(means, alpha / 2.0),
        percentile(means, 1.0 - alpha / 2.0),
    )


def get_nested(record: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = record
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_label(label: Any) -> str:
    return str(label).strip().lower().replace("-", "_")


# -----------------------------
# Episode-level DAP scoring
# -----------------------------

@dataclass
class EpisodeScore:
    episode_id: str
    method: str
    task: str
    success: float
    completion_quality: float
    deformation_quality: Optional[float]
    failure_mode: Optional[str]
    perturbation_type: Optional[str]
    perturbation_magnitude: Optional[float]


def compute_binary_success(record: Dict[str, Any]) -> float:
    """
    Implements S_succ = I(y = success).

    Preferred fields:
        success: bool
        binary_success: bool

    Fallback final_state format:
        final_state.target_reached: bool
        final_state.safety_violation: bool
        final_state.severe_misalignment: bool
        final_state.timeout: bool
        final_state.out_of_tolerance_damage: bool
    """
    if "success" in record:
        return 1.0 if bool(record["success"]) else 0.0

    if "binary_success" in record:
        return 1.0 if bool(record["binary_success"]) else 0.0

    final_state = record.get("final_state", {})
    if isinstance(final_state, dict) and "target_reached" in final_state:
        target_reached = bool(final_state.get("target_reached", False))
        invalid = any(
            bool(final_state.get(k, False))
            for k in [
                "safety_violation",
                "severe_misalignment",
                "timeout",
                "out_of_tolerance_damage",
                "out_of_tolerance_physical_damage",
            ]
        )
        return 1.0 if target_reached and not invalid else 0.0

    raise ValueError(
        "Cannot compute binary success. Provide `success`, "
        "`binary_success`, or a valid `final_state` field."
    )


def compute_completion_quality(
    record: Dict[str, Any],
    ordinal_map: Dict[str, float] = COMPLETION_ORDINAL_MAP,
) -> float:
    """
    Implements S_comp in [0, 1].

    Preferred continuous fields:
        completion_quality: float
        completion.value: float

    Ordinal fallback:
        completion.ordinal: "none" | "partial" | "complete"
        completion_label: string
    """
    if "completion_quality" in record:
        return clamp01(float(record["completion_quality"]))

    value = get_nested(record, ["completion", "value"])
    if value is not None:
        return clamp01(float(value))

    ordinal = get_nested(record, ["completion", "ordinal"])
    if ordinal is None:
        ordinal = record.get("completion_label")

    if ordinal is not None:
        key = normalize_label(ordinal)
        if key in ordinal_map:
            return clamp01(ordinal_map[key])
        raise ValueError(f"Unknown completion ordinal label: {ordinal}")

    # If no completion annotation is given, fall back to binary success.
    # This is conservative and should be avoided for final reporting.
    return compute_binary_success(record)


def compute_deformation_quality(
    record: Dict[str, Any],
    qdef_label_map: Dict[str, float] = QDEF_LABEL_MAP,
) -> Optional[float]:
    """
    Implements Q_def.

    Preferred fields:
        deformation_quality: float
        q_def.score: float

    Ordinal real-robot fallback:
        q_def.label: acceptable | minor_defect | major_defect | catastrophic_failure

    Proxy fallback:
        q_def.proxy.E_tol
        q_def.proxy.E_contact
        q_def.proxy.w_tol, q_def.proxy.w_contact optional

    Because the paper defines Q_def = phi(E_tol, E_contact) without fixing
    a universal phi, this implementation uses a simple configurable proxy:
        Q_def = 1 - clamp(w_tol * E_tol + w_contact * E_contact, 0, 1)
    """
    if "deformation_quality" in record:
        return clamp01(float(record["deformation_quality"]))

    score = get_nested(record, ["q_def", "score"])
    if score is not None:
        return clamp01(float(score))

    label = get_nested(record, ["q_def", "label"])
    if label is not None:
        key = normalize_label(label)
        if key in qdef_label_map:
            return clamp01(qdef_label_map[key])
        raise ValueError(f"Unknown Q_def ordinal label: {label}")

    proxy = get_nested(record, ["q_def", "proxy"])
    if isinstance(proxy, dict):
        e_tol = float(proxy.get("E_tol", 0.0))
        e_contact = float(proxy.get("E_contact", 0.0))
        w_tol = float(proxy.get("w_tol", 0.5))
        w_contact = float(proxy.get("w_contact", 0.5))
        violation = clamp01(w_tol * e_tol + w_contact * e_contact)
        return 1.0 - violation

    return None


def get_failure_mode(record: Dict[str, Any]) -> Optional[str]:
    mode = record.get("failure_mode")
    if mode is None:
        mode = get_nested(record, ["failure", "mode"])

    if mode is None:
        return None

    mode = normalize_label(mode)
    return mode if mode in FAILURE_MODES else "other"


def get_perturbation(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    ptype = get_nested(record, ["perturbation", "type"])
    if ptype is None:
        ptype = record.get("perturbation_type")

    mag = get_nested(record, ["perturbation", "magnitude"])
    if mag is None:
        mag = record.get("perturbation_magnitude")
    if mag is None:
        mag = record.get("delta")

    return (str(ptype) if ptype is not None else None, float(mag) if mag is not None else None)


def score_episode(record: Dict[str, Any]) -> EpisodeScore:
    episode_id = str(record.get("episode_id", record.get("id", "")))
    method = str(record.get("method", "unknown_method"))
    task = str(record.get("task", "unknown_task"))

    success = compute_binary_success(record)
    completion = compute_completion_quality(record)
    qdef = compute_deformation_quality(record)
    failure_mode = get_failure_mode(record)

    ptype, pmag = get_perturbation(record)

    return EpisodeScore(
        episode_id=episode_id,
        method=method,
        task=task,
        success=success,
        completion_quality=completion,
        deformation_quality=qdef,
        failure_mode=failure_mode,
        perturbation_type=ptype,
        perturbation_magnitude=pmag,
    )


# -----------------------------
# Dataset-level DAP aggregation
# -----------------------------

def group_scores(scores: List[EpisodeScore]) -> Dict[Tuple[str, str], List[EpisodeScore]]:
    groups: Dict[Tuple[str, str], List[EpisodeScore]] = defaultdict(list)
    for s in scores:
        groups[(s.method, s.task)].append(s)
    return groups


def failure_distribution(scores: List[EpisodeScore]) -> Dict[str, float]:
    """
    Computes P(f_j) over failed episodes.
    """
    selected = [
        s for s in scores
        if s.success < 1.0
    ]

    modes = []
    for s in selected:
        if s.failure_mode is not None:
            modes.append(s.failure_mode)
        else:
            modes.append("other")

    if not modes:
        return {}

    counts = Counter(modes)
    total = sum(counts.values())
    return {mode: counts.get(mode, 0) / total for mode in FAILURE_MODES}


def robustness_metrics(scores: List[EpisodeScore]) -> Dict[str, Any]:
    """
    Computes S_robust and PR-AUC when perturbation magnitudes are available.

    S_robust is the mean S_comp across perturbation levels.
    PR-AUC is normalized trapezoidal AUC of SR(delta).
    """
    perturbed = [s for s in scores if s.perturbation_magnitude is not None]
    if not perturbed:
        return {}

    by_delta: Dict[float, List[EpisodeScore]] = defaultdict(list)
    for s in perturbed:
        by_delta[float(s.perturbation_magnitude)].append(s)

    deltas = sorted(by_delta.keys())
    comp_by_delta = {
        d: safe_mean([s.completion_quality for s in by_delta[d]])
        for d in deltas
    }
    sr_by_delta = {
        d: safe_mean([s.success for s in by_delta[d]])
        for d in deltas
    }

    s_robust = safe_mean([comp_by_delta[d] for d in deltas])

    pr_auc = None
    if len(deltas) >= 2 and deltas[-1] > deltas[0]:
        auc = 0.0
        for d0, d1 in zip(deltas[:-1], deltas[1:]):
            y0 = sr_by_delta[d0]
            y1 = sr_by_delta[d1]
            if y0 is None or y1 is None:
                continue
            auc += 0.5 * (y0 + y1) * (d1 - d0)
        pr_auc = auc / (deltas[-1] - deltas[0])

    return {
        "S_robust": s_robust,
        "PR_AUC": pr_auc,
        "completion_by_delta": comp_by_delta,
        "success_by_delta": sr_by_delta,
    }


def summarize_group(
    scores: List[EpisodeScore],
    bootstrap: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    success_values = [s.success for s in scores]
    comp_values = [s.completion_quality for s in scores]
    qdef_values = [
        s.deformation_quality
        for s in scores
        if s.deformation_quality is not None
    ]

    summary: Dict[str, Any] = {
        "num_episodes": len(scores),
        "SR": safe_mean(success_values),
        "CQ": safe_mean(comp_values),
        "DQ": safe_mean(qdef_values),
        "failure_distribution": failure_distribution(scores),
        "robustness": robustness_metrics(scores),
    }

    if bootstrap > 0:
        summary["bootstrap_ci"] = {
            "SR": bootstrap_ci(success_values, bootstrap, seed),
            "CQ": bootstrap_ci(comp_values, bootstrap, seed + 1),
            "DQ": bootstrap_ci(qdef_values, bootstrap, seed + 2) if qdef_values else None,
        }

    return summary


def summarize_all(
    scores: List[EpisodeScore],
    bootstrap: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    groups = group_scores(scores)

    output: Dict[str, Any] = {
        "overall": summarize_group(scores, bootstrap=bootstrap, seed=seed),
        "by_method_task": {},
    }

    for (method, task), group in sorted(groups.items()):
        key = f"{method}::{task}"
        output["by_method_task"][key] = summarize_group(
            group,
            bootstrap=bootstrap,
            seed=seed,
        )

    return output


# -----------------------------
# I/O
# -----------------------------

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
                raise ValueError(f"Invalid JSON on line {line_idx}: {e}") from e
    return records


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_group_csv(summary: Dict[str, Any], path: Path) -> None:
    rows = []
    for key, item in summary["by_method_task"].items():
        method, task = key.split("::", 1)
        robust = item.get("robustness", {}) or {}
        rows.append({
            "method": method,
            "task": task,
            "num_episodes": item.get("num_episodes"),
            "SR": item.get("SR"),
            "CQ": item.get("CQ"),
            "DQ": item.get("DQ"),
            "S_robust": robust.get("S_robust"),
            "PR_AUC": robust.get("PR_AUC"),
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "task",
                "num_episodes",
                "SR",
                "CQ",
                "DQ",
                "S_robust",
                "PR_AUC",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def format_float(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, (float, int)):
        return f"{float(x):.4f}"
    return str(x)


def save_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# FIRM DAP Evaluation Report\n")
    lines.append("## Overall\n")

    overall = summary["overall"]
    lines.append(f"- Episodes: {overall['num_episodes']}")
    lines.append(f"- SR: {format_float(overall['SR'])}")
    lines.append(f"- CQ: {format_float(overall['CQ'])}")
    lines.append(f"- DQ: {format_float(overall['DQ'])}")

    robust = overall.get("robustness", {}) or {}
    if robust:
        lines.append(f"- S_robust: {format_float(robust.get('S_robust'))}")
        lines.append(f"- PR-AUC: {format_float(robust.get('PR_AUC'))}")

    lines.append("\n## Per Method and Task\n")
    lines.append("| Method | Task | N | SR | CQ | DQ | S_robust | PR-AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for key, item in summary["by_method_task"].items():
        method, task = key.split("::", 1)
        robust = item.get("robustness", {}) or {}
        lines.append(
            "| "
            + " | ".join([
                method,
                task,
                str(item.get("num_episodes")),
                format_float(item.get("SR")),
                format_float(item.get("CQ")),
                format_float(item.get("DQ")),
                format_float(robust.get("S_robust")),
                format_float(robust.get("PR_AUC")),
            ])
            + " |"
        )

    lines.append("\n## Failure-Mode Distributions\n")
    for key, item in summary["by_method_task"].items():
        method, task = key.split("::", 1)
        fd = item.get("failure_distribution", {}) or {}
        if not fd:
            continue
        lines.append(f"\n### {method} / {task}\n")
        for mode, prob in fd.items():
            if prob > 0:
                lines.append(f"- {mode}: {prob:.4f}")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Paired comparison
# -----------------------------

def paired_delta(
    scores: List[EpisodeScore],
    method_a: str,
    method_b: str,
    metric: str = "completion_quality",
) -> Dict[str, Any]:
    """
    Computes paired difference over matched episode IDs.

    metric options:
        success
        completion_quality
        deformation_quality
    """
    by_method_episode: Dict[Tuple[str, str], EpisodeScore] = {}
    for s in scores:
        by_method_episode[(s.method, s.episode_id)] = s

    deltas = []
    for (method, episode_id), score_a in by_method_episode.items():
        if method != method_a:
            continue
        score_b = by_method_episode.get((method_b, episode_id))
        if score_b is None:
            continue

        va = getattr(score_a, metric)
        vb = getattr(score_b, metric)
        if va is None or vb is None:
            continue
        deltas.append(float(vb) - float(va))

    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "num_pairs": len(deltas),
        "mean_delta_b_minus_a": safe_mean(deltas),
        "deltas": deltas,
    }


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FIRM episodes with DAP metrics."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL file with per-episode annotations.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("dap_summary.json"),
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dap_summary.csv"),
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("dap_report.md"),
        help="Output Markdown report.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--paired",
        nargs=3,
        metavar=("METHOD_A", "METHOD_B", "METRIC"),
        help="Optional paired comparison: METHOD_A METHOD_B metric.",
    )

    args = parser.parse_args()

    records = load_jsonl(args.input)
    scores = [score_episode(r) for r in records]
    summary = summarize_all(scores, bootstrap=args.bootstrap, seed=args.seed)

    if args.paired:
        method_a, method_b, metric = args.paired
        summary["paired_comparison"] = paired_delta(
            scores,
            method_a=method_a,
            method_b=method_b,
            metric=metric,
        )

    save_json(summary, args.output_json)
    save_group_csv(summary, args.output_csv)
    save_markdown_report(summary, args.output_md)

    print(f"[OK] Wrote {args.output_json}")
    print(f"[OK] Wrote {args.output_csv}")
    print(f"[OK] Wrote {args.output_md}")


if __name__ == "__main__":
    main()  
