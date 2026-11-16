"""Strict aggregation of complete paired rollout records."""

from collections import defaultdict
import numpy as np
from firm_sim.perturbations import FAMILIES, LEVELS


def summarize(conditions, results):
    expected = {c.id: c for c in conditions}
    if len(expected) != len(conditions):
        raise ValueError("Duplicate conditions")
    records = {}
    for r in results:
        if r["condition_id"] not in expected or r["condition_id"] in records:
            raise ValueError("Unexpected or duplicate result")
        if r.get("status") != "completed" or type(r.get("success")) is not bool:
            raise ValueError("Incomplete/error rollout; PR-AUC is undefined")
        records[r["condition_id"]] = r
    if set(records) != set(expected):
        raise ValueError("Missing rollouts; PR-AUC is undefined")
    groups = defaultdict(list)
    trials = defaultdict(set)
    for key, c in expected.items():
        if (
            c.family not in FAMILIES
            or c.delta not in LEVELS
            or c.trial not in range(20)
        ):
            raise ValueError("Invalid protocol condition")
        groups[c.task, c.family, c.delta].append(records[key]["success"])
        trials[c.task, c.family, c.delta].add(c.trial)
    output = {}
    for task in sorted({c.task for c in conditions}):
        families = {}
        for family in FAMILIES:
            rates = []
            for delta in LEVELS:
                values = groups[task, family, delta]
                if len(values) != 20 or trials[task, family, delta] != set(range(20)):
                    raise ValueError("Expected 20 rollouts per level")
                rates.append(float(np.mean(values)))
            auc = sum(
                (rates[i] + rates[i + 1]) * (LEVELS[i + 1] - LEVELS[i]) / 2
                for i in range(len(LEVELS) - 1)
            ) / (LEVELS[-1] - LEVELS[0])
            families[family] = {"success_rates": rates, "auc": auc}
        output[task] = {
            "families": families,
            "pr_auc": sum(v["auc"] for v in families.values()) / 3,
        }
    return output
