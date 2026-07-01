#!/usr/bin/env bash
set -eo pipefail

LINGBOT_ROOT="${LINGBOT_ROOT:-/path/to/lingbot-va}"
REPO=${REPO:-$LINGBOT_ROOT/lingbot-va}
BASE_MODEL=${BASE_MODEL:-$LINGBOT_ROOT/models/lingbot-va-base}
LINGBOT_DATA_ROOT="${LINGBOT_DATA_ROOT:-/path/to/lingbot-data}"
DATASET_PATH=${DATASET_PATH:-$LINGBOT_DATA_ROOT/haimiandian_50}
RUN_ROOT=${RUN_ROOT:-$LINGBOT_ROOT/outputs/haimiandian_full_$(date +%Y%m%d_%H%M%S)}
POLL_INTERVAL=${POLL_INTERVAL:-60}
NUM_EVAL_SAMPLES=${NUM_EVAL_SAMPLES:-100}
SEED=${SEED:-20260501}
MASTER_PORT=${MASTER_PORT:-29521}
CONDA_ENV=${CONDA_ENV:-lingbot-va-torch29}

mkdir -p "$RUN_ROOT"/{logs,backups}
exec > >(tee -a "$RUN_ROOT/pipeline.log") 2>&1

echo "[pipeline] start $(date)"
echo "[pipeline] repo=$REPO"
echo "[pipeline] run_root=$RUN_ROOT"
echo "[pipeline] conda_env=$CONDA_ENV"

source ${CONDA_ROOT:-/path/to/miniconda3}/bin/activate "$CONDA_ENV"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
python - <<'PY'
import torch
print(f"[pipeline] python_torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
PY

run_and_monitor() {
  local name="$1"
  shift
  local log="$RUN_ROOT/logs/${name}.log"
  echo "[pipeline] launch $name at $(date)"
  echo "[pipeline] command: $*"
  ( "$@" ) > "$log" 2>&1 &
  local pid=$!
  echo "[pipeline] $name pid=$pid log=$log"
  while kill -0 "$pid" 2>/dev/null; do
    echo "[pipeline] poll $name $(date)"
    tail -n 30 "$log" || true
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    sleep "$POLL_INTERVAL"
  done
  set +e
  wait "$pid"
  local rc=$?
  set -e
  echo "[pipeline] $name exit_code=$rc at $(date)"
  tail -n 80 "$log" || true
  if [ "$rc" -ne 0 ]; then
    echo "[pipeline] ERROR: $name failed"
    exit "$rc"
  fi
}

set_attn_mode() {
  local config_file="$1"
  local mode="$2"
  python - "$config_file" "$mode" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
mode = sys.argv[2]
d = json.loads(p.read_text())
d["attn_mode"] = mode
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"{p}: attn_mode={mode}")
PY
}

echo "[pipeline] preflight $(date)"
date
hostname
nvidia-smi
git status --short || true

cp -a wan_va/configs/va_haimiandian_train_cfg.py "$RUN_ROOT/backups/va_haimiandian_train_cfg.py.before"
cp -a "$BASE_MODEL/transformer/config.json" "$RUN_ROOT/backups/base_transformer_config.json.before"

python - <<'PY'
from pathlib import Path
import re
p = Path("wan_va/configs/va_haimiandian_train_cfg.py")
s = p.read_text()
s = re.sub(r"va_haimiandian_train_cfg\.save_interval\s*=\s*\d+", "va_haimiandian_train_cfg.save_interval = 500", s)
s = re.sub(r"va_haimiandian_train_cfg\.num_steps\s*=\s*\d+", "va_haimiandian_train_cfg.num_steps = 5000", s)
p.write_text(s)
print("updated train config: save_interval=500 num_steps=5000")
PY

grep -n "save_interval\|num_steps\|batch_size\|gradient_accumulation_steps" wan_va/configs/va_haimiandian_train_cfg.py
test -f "$BASE_MODEL/vae/config.json"
test -f "$BASE_MODEL/vae/diffusion_pytorch_model.safetensors"
stat -c "%s %n" "$BASE_MODEL/vae/diffusion_pytorch_model.safetensors"

if [ ! -f "$DATASET_PATH/empty_emb.pt" ] || [ "$(find "$DATASET_PATH/latents" -type f -name '*.pth' 2>/dev/null | wc -l)" -lt 150 ]; then
  run_and_monitor latent_extract env PYTHONPATH="$PYTHONPATH" MODEL_PATH="$BASE_MODEL" DATASET_PATH="$DATASET_PATH" bash script/run_extract_haimiandian_latents.sh --force
else
  echo "[pipeline] latent extraction already complete enough; skipping"
fi

test -f "$DATASET_PATH/empty_emb.pt"
latent_count=$(find "$DATASET_PATH/latents" -type f -name '*.pth' | wc -l)
echo "[pipeline] latent_count=$latent_count"
if [ "$latent_count" -lt 150 ]; then
  echo "[pipeline] ERROR: expected at least 150 latent files"
  exit 2
fi

set_attn_mode "$BASE_MODEL/transformer/config.json" torch
ZS_OUT="$RUN_ROOT/zeroshot_eval"
mkdir -p "$ZS_OUT"
run_and_monitor zeroshot_eval env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" python evaluation/haimiandian/evaluate_lingbot_zeroshot_sft.py \
  --model-root "$BASE_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --mode zeroshot \
  --num-samples "$NUM_EVAL_SAMPLES" \
  --output-dir "$ZS_OUT" \
  --device cuda:0 \
  --seed "$SEED"

set_attn_mode "$BASE_MODEL/transformer/config.json" flex
SFT_ROOT="$RUN_ROOT/sft"
mkdir -p "$SFT_ROOT"
run_and_monitor sft_train env NGPU=4 MASTER_PORT="$MASTER_PORT" CONFIG_NAME=haimiandian_train SAVE_ROOT="$SFT_ROOT" PYTHONPATH="$PYTHONPATH" bash script/run_haimiandian_posttrain.sh

CKPT="$SFT_ROOT/checkpoints/checkpoint_step_5000"
test -d "$CKPT/transformer"

EVAL_MODEL="$RUN_ROOT/eval_model_checkpoint_step_5000"
mkdir -p "$EVAL_MODEL"
rm -rf "$EVAL_MODEL/transformer"
cp -a "$CKPT/transformer" "$EVAL_MODEL/transformer"
ln -sfn "$BASE_MODEL/vae" "$EVAL_MODEL/vae"
ln -sfn "$BASE_MODEL/text_encoder" "$EVAL_MODEL/text_encoder"
ln -sfn "$BASE_MODEL/tokenizer" "$EVAL_MODEL/tokenizer"
set_attn_mode "$EVAL_MODEL/transformer/config.json" torch

SFT_EVAL_OUT="$RUN_ROOT/sft_eval"
mkdir -p "$SFT_EVAL_OUT"
run_and_monitor sft_eval env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH" python evaluation/haimiandian/evaluate_lingbot_zeroshot_sft.py \
  --model-root "$EVAL_MODEL" \
  --dataset-path "$DATASET_PATH" \
  --mode sft \
  --checkpoint "$CKPT" \
  --num-samples "$NUM_EVAL_SAMPLES" \
  --output-dir "$SFT_EVAL_OUT" \
  --device cuda:0 \
  --seed "$SEED"

python - "$RUN_ROOT" "$ZS_OUT/metrics.json" "$SFT_EVAL_OUT/metrics.json" <<'PY'
import json
import sys
from pathlib import Path
run_root = Path(sys.argv[1])
zs = json.loads(Path(sys.argv[2]).read_text())
sft = json.loads(Path(sys.argv[3]).read_text())
keys = sorted(zs["metrics"])
lines = [
    "# LingBot-VA Haimiandian Full Pipeline Results",
    "",
    f"- Run root: `{run_root}`",
    f"- Zero-shot output: `{Path(sys.argv[2]).parent}`",
    f"- SFT eval output: `{Path(sys.argv[3]).parent}`",
    "",
    "| metric | zero-shot | SFT | abs delta | rel change |",
    "|---|---:|---:|---:|---:|",
]
for key in keys:
    a = float(zs["metrics"][key])
    b = float(sft["metrics"][key])
    delta = b - a
    rel = delta / a if abs(a) > 1e-12 else 0.0
    lines.append(f"| {key} | {a:.8f} | {b:.8f} | {delta:.8f} | {rel:.2%} |")
lines.append("")
lines.append("Lower is better for all listed loss/error metrics.")
(run_root / "final_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[pipeline] completed $(date)"
