"""Export machine-readable metrics and success-rate curves."""

import csv
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from firm_sim.perturbations import FAMILIES, LEVELS


def write_report(directory, method, summary):
    (directory / "summary.json").write_text(
        json.dumps({"method": method, "tasks": summary}, indent=2) + "\n"
    )
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "task", "object_auc", "fixture_auc", "scene_auc", "pr_auc"]
        )
        for task, values in summary.items():
            writer.writerow(
                [
                    method,
                    task,
                    *(values["families"][f]["auc"] for f in FAMILIES),
                    values["pr_auc"],
                ]
            )
    with (directory / "success_rates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "task", "family", "delta", "success_rate"])
        for task, values in summary.items():
            figure, axis = plt.subplots(figsize=(5, 3.5), layout="constrained")
            for family in FAMILIES:
                rates = values["families"][family]["success_rates"]
                axis.plot(LEVELS, rates, marker="o", label=family)
                for delta, rate in zip(LEVELS, rates):
                    writer.writerow([method, task, family, delta, rate])
            axis.set(
                xlabel="Perturbation strength",
                ylabel="Success rate",
                xlim=(0, 1),
                ylim=(0, 1),
                title=f"{method}: {task}",
            )
            axis.legend()
            axis.grid(alpha=0.2)
            figure.savefig(directory / f"{task}_curves.png", dpi=180)
            plt.close(figure)
