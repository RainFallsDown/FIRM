"""Generate schedules, run reproducible policy evaluations, and summarize PR-AUC."""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from firm_sim.perturbations import Condition, generate_conditions
from firm_sim.evaluation.metrics import summarize


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schedule = commands.add_parser("schedule")
    schedule.add_argument("--tasks", nargs="+", required=True)
    schedule.add_argument("--seed", type=int, default=0)
    schedule.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--schedule", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=1800)
    report = commands.add_parser("summarize")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "schedule":
        from firm_sim.tasks import task_names

        if not set(args.tasks).issubset(task_names()):
            parser.error(f"Available tasks: {task_names()}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output,
            [c.to_dict() for c in generate_conditions(args.tasks, args.seed)],
        )
        return
    if args.command == "run":
        if args.timeout <= 0:
            parser.error("timeout must be positive")
        config = json.loads(args.config.read_text())
        for name in ("method", "policy", "judge", "camera", "max_steps"):
            if name not in config:
                parser.error(f"Missing configuration field: {name}")
        if type(config["max_steps"]) is not int or config["max_steps"] <= 0:
            parser.error("max_steps must be a positive integer")
        package = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in sorted(package.rglob("*.py")):
            digest.update(str(source.relative_to(package)).encode())
            digest.update(source.read_bytes())
        asset = package.parent / "tianqing_urdf.zip"
        asset_digest = hashlib.sha256()
        with asset.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                asset_digest.update(block)
        payload = {
            "robot_asset_sha256": asset_digest.hexdigest(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "genesis-world": version("genesis-world"),
                "numpy": version("numpy"),
                "scipy": version("scipy"),
            },
            "conditions": json.loads(args.schedule.read_text()),
            "config": config,
            "source_sha256": digest.hexdigest(),
        }

        conditions = [Condition(**c) for c in payload["conditions"]]
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        # Reject malformed schedules before starting any simulations.
        summarize(
            conditions,
            [
                {"condition_id": c.id, "status": "completed", "success": False}
                for c in conditions
            ],
        )
        if not conditions:
            parser.error("Schedule must contain at least one task")
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = args.output / "run.json"
        if (
            metadata.exists()
            and json.loads(metadata.read_text())["identity"] != identity
        ):
            parser.error("Output belongs to another schedule/configuration")
        if not metadata.exists():
            write_json(metadata, dict(payload, identity=identity))
        root = str(Path(__file__).resolve().parents[2])
        environment = dict(
            os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
        request = args.output / "request.json"
        for condition in conditions:
            result = args.output / f"{condition.id}.json"
            if (
                result.exists()
                and json.loads(result.read_text()).get("status") == "completed"
            ):
                continue
            write_json(request, {"condition": condition.to_dict(), "config": config})
            with (args.output / f"{condition.id}.log").open("w") as log:
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "firm_sim.evaluation.worker",
                            str(request.resolve()),
                            str(result.resolve()),
                        ],
                        env=environment,
                        stdout=log,
                        stderr=log,
                        timeout=args.timeout,
                    )
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"Rollout failed; inspect {result} and its log"
                        )
                except subprocess.TimeoutExpired as exc:
                    write_json(
                        result,
                        {
                            "condition_id": condition.id,
                            "status": "error",
                            "error": "timeout",
                        },
                    )
                    raise RuntimeError(f"Rollout timed out: {condition.id}") from exc
            print(
                f"{condition.task} {condition.family} delta={condition.delta} trial={condition.trial}",
                flush=True,
            )
        request.unlink(missing_ok=True)
    metadata = json.loads((args.output / "run.json").read_text())
    conditions = [Condition(**c) for c in metadata["conditions"]]
    results = [
        json.loads((args.output / f"{c.id}.json").read_text())
        for c in conditions
        if (args.output / f"{c.id}.json").exists()
    ]
    summary = summarize(conditions, results)
    from firm_sim.evaluation.report import write_report

    write_report(args.output, metadata["config"]["method"], summary)


if __name__ == "__main__":
    main()
