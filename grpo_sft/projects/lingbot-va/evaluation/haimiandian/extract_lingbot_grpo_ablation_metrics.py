import json
import os
import re
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "LINGBOT_GRPO_ABLATION_ROOT",
        "${LINGBOT_ROOT:-/path/to/lingbot-va}/outputs/haimiandian_lingbot_grpo_ablation_20260503",
    )
)

METRIC_RE = re.compile(
    r"(latent_loss|action_loss|step|grpo_reward|grpo_advantage|grpo_weight)=([-+0-9.eE]+)"
)


def _read_launch_info(path):
    info = {}
    if not path.exists():
        return info
    for line in path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def _last_metric(log_text):
    last = {}
    for line in log_text.splitlines():
        if "latent_loss=" not in line or "action_loss=" not in line:
            continue
        pairs = dict(METRIC_RE.findall(line))
        if "latent_loss" in pairs and "action_loss" in pairs:
            last = pairs
    return last


def collect_rows(root):
    rows = []
    for run_dir in sorted(root.glob("lb_hmd_*")):
        log_path = run_dir / "logs" / "nohup_grpo.log"
        launch_info = _read_launch_info(run_dir / "logs" / "launch_info.txt")
        row = {
            "name": run_dir.name,
            "use_grpo": launch_info.get("use_grpo"),
            "reward_scale": launch_info.get("reward_scale"),
            "alpha": launch_info.get("alpha"),
            "buffer_size": launch_info.get("buffer_size"),
            "num_steps": launch_info.get("num_steps"),
            "save_interval": launch_info.get("save_interval"),
            "resume_from": launch_info.get("resume_from"),
            "vae": launch_info.get("vae"),
            "log": str(log_path),
        }
        if not log_path.exists():
            row["status"] = "missing_log"
            rows.append(row)
            continue

        text = log_path.read_text(errors="ignore")
        row.update(_last_metric(text))
        if "Training completed!" in text and "[ablation] exit 0" in text:
            row["status"] = "ok"
        elif "Traceback" in text:
            row["status"] = "traceback"
        elif "CUDA out of memory" in text or "OutOfMemoryError" in text:
            row["status"] = "oom"
        else:
            row["status"] = "check_log"
        rows.append(row)
    return rows


def main():
    rows = collect_rows(ROOT)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
