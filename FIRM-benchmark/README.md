# FIRM Benchmark

Utilities for evaluating robotic manipulation episodes with a DAP-style benchmark pipeline.

The current codebase supports mask-based evaluation for industrial flexible-object manipulation, including object mask generation, raw metric extraction, task-specific scoring, and dataset-level reporting.

## Overview

FIRM evaluates robotic manipulation episodes beyond binary success. The benchmark focuses on industrial flexible-object and mixed-stiffness manipulation tasks, where policies may partially complete a task, deform the object, jam against fixtures, or fail due to unstable contact.

The evaluation pipeline supports:

* Object mask generation with official SAM3 text prompting.
* Mask-based geometric metric extraction.
* Task-specific scoring for completion quality and deformation-aware quality.
* Dataset-level aggregation of success rate, completion quality, deformation quality, and failure distributions.

## Pipeline

The full benchmark pipeline is:

1. Prepare an annotation workspace with episode folders, final frames, sampled frames, `target_mask.png`, and metadata.
2. Generate object masks with official SAM3.
3. Extract raw mask metrics.
4. Score episodes with task-specific rules.
5. Aggregate SR/CQ/DQ and failure distributions.

Core scripts:

```text
firm_benchmark/generate_sam3_masks_real.py
firm_benchmark/mask_metric_extractor.py
firm_benchmark/episode_scorer.py
firm_benchmark/dap_eval.py
```

## Repository Layout

```text
FIRM-benchmark/
├── README.md
├── BENCHMARK_RUN_SUMMARY.md
├── requirements.txt
├── configs/
│   └── scoring_config.json
└── firm_benchmark/
    ├── generate_sam3_masks_real.py
    ├── mask_metric_extractor.py
    ├── episode_scorer.py
    └── dap_eval.py
```

Large generated folders are intentionally not committed to this repository.

## Installation

Create and activate a Python environment:

```bash
conda create -n firm_benchmark python=3.10 -y
conda activate firm_benchmark
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If you use CUDA-enabled PyTorch, install the PyTorch version that matches your local CUDA driver before running the benchmark scripts.

## Official SAM3 Setup

This benchmark uses the official Meta Segment Anything Model 3 implementation for text-prompted mask generation.

### 1. Clone official SAM3

Clone the official SAM3 repository:

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
cd third_party/sam3
pip install -e .
cd ../..
```

If additional dependencies are required, follow the installation instructions in the official SAM3 repository:

```text
https://github.com/facebookresearch/sam3
```

### 2. Download SAM3 checkpoint

The official SAM3 checkpoint can be downloaded from the Meta SAM3 Hugging Face model page:

```text
https://huggingface.co/facebook/sam3
```

If the model page requires access approval, log in to Hugging Face and accept the model terms first.

Install the Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Log in:

```bash
huggingface-cli login
```

Download the SAM3 checkpoint:

```bash
mkdir -p checkpoints/sam3

huggingface-cli download facebook/sam3 sam3.pt \
  --local-dir checkpoints/sam3 \
  --local-dir-use-symlinks False
```

The expected checkpoint path is:

```text
checkpoints/sam3/sam3.pt
```

Large checkpoint files should not be committed to this repository.

### 3. Configure SAM3 paths

Before running the mask generation script, make the official SAM3 repository importable:

```bash
export PYTHONPATH=$(pwd)/third_party/sam3:$PYTHONPATH
```

Optionally, define environment variables for local scripts:

```bash
export SAM3_REPO=$(pwd)/third_party/sam3
export SAM3_CHECKPOINT=$(pwd)/checkpoints/sam3/sam3.pt
```

If your local script expects a fixed checkpoint path such as `~/sam3/sam3.pt`, create a symlink:

```bash
mkdir -p ~/sam3
ln -sf $(pwd)/checkpoints/sam3/sam3.pt ~/sam3/sam3.pt
```

## Annotation Workspace

The benchmark expects each task to be organized as an annotation workspace.

Example structure:

```text
annotation_workspace_tape/
└── episodes/
    ├── episode_000000/
    │   ├── target_mask.png
    │   ├── meta.json
    │   ├── final_frames/
    │   │   └── observation.images.head.color.png
    │   └── sampled_frames/
    │       └── observation.images.head.color/
    │           ├── frame_000000.png
    │           ├── frame_000001.png
    │           └── ...
    ├── episode_000001/
    └── ...
```

Required files:

* `target_mask.png`: manually or semi-automatically annotated target region.
* `final_frames/`: final observation frames for each camera.
* `sampled_frames/`: sampled episode frames for visualization and optional VLM checks.
* `meta.json`: optional episode metadata.

## Generate Object Masks with SAM3

Example command:

```bash
PYTHONPATH=$(pwd)/third_party/sam3:$PYTHONPATH \
python firm_benchmark/generate_sam3_masks_real.py \
  --episodes-root annotation_workspace_tape/episodes \
  --camera observation.images.head.color \
  --text-prompt "tape roll in paper box" \
  --device cuda \
  --use-target-mask-selection \
  --apply-ring-filter
```

For different tasks, change the text prompt accordingly.

Example prompts:

```text
Tape: tape roll in paper box
Sponge: white sponge pad
Manual: paper manual
Cable: cable
Box: cardboard box
```

If the script exposes a checkpoint argument in your local version, pass:

```bash
--checkpoint checkpoints/sam3/sam3.pt
```

Otherwise, make sure the checkpoint path expected by the script points to the official SAM3 checkpoint.

## Extract Raw Mask Metrics

After object masks are generated, extract raw geometric and mask-based metrics:

```bash
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root annotation_workspace_tape/episodes \
  --config configs/scoring_config.json \
  --output raw_episode_metrics_tape.jsonl
```

The extracted metrics may include:

* target-region coverage
* object-target overlap
* pose or centroid error
* mask area
* compactness
* task-specific geometry proxies

The exact metric set depends on the task category and available masks.

## Score Episodes

Score each episode with task-specific rules:

```bash
python firm_benchmark/episode_scorer.py \
  --input raw_episode_metrics_tape.jsonl \
  --config configs/scoring_config.json \
  --output scored_annotations_tape.jsonl
```

The scorer converts raw metrics into:

* binary success
* completion quality
* deformation-aware quality
* dominant failure mode

## Aggregate Benchmark Results

Aggregate episode-level scores into dataset-level results:

```bash
python firm_benchmark/dap_eval.py \
  --input scored_annotations_tape.jsonl \
  --output-json dap_summary_tape.json \
  --output-csv dap_summary_tape.csv \
  --output-md dap_report_tape.md \
  --bootstrap 1000
```

The aggregated report includes:

* SR: success rate
* CQ: completion quality
* DQ: deformation-aware execution quality
* failure-mode distribution
* optional bootstrap confidence intervals

## Scoring Policy

The current scoring config emphasizes task completion over rigid geometric matching.

Task-level scoring summary:

* `Manual`: success is mainly whether the manual or paper is inside the accepted target region; DQ focuses on severe folding, jamming, or dropping.
* `Cable`: success is mainly whether the cable is contained; DQ focuses on severe tangling and contact issues rather than the natural low compactness of cables.
* `Box`: success is mainly whether the formed box lies in the target region; DQ focuses on severe collapse, deformation, or contact instability.
* `Tape`: success is whether at least one tape roll is in the paper box.
* `Sponge`: success uses object-in-target ratio with a lightweight pose component.

See:

```text
configs/scoring_config.json
```

for thresholds, weights, and task-specific rules.

## Generated Files

Large generated folders are intentionally ignored by git:

```text
annotation_workspace*/
results*/
*.mp4
*.parquet
*.pt
*.pth
*.ckpt
checkpoints/
```

Do not commit:

* raw videos
* LeRobot parquet files
* model checkpoints
* annotation workspaces
* large result images
* intermediate masks

Use:

```text
BENCHMARK_RUN_SUMMARY.md
```

for the latest compact result table instead of committing large result folders.

## Tested Tasks

The latest local run covered five task lines:

* Sponge
* Tape
* Manual1
* Cable&Mouse
* Manufacturing_Lines_Box

See:

```text
BENCHMARK_RUN_SUMMARY.md
```

for metrics and artifact locations.

## Notes on SAM3 Access

SAM3 weights may require Hugging Face login and access approval. If checkpoint download fails, check:

1. Whether you are logged in with `huggingface-cli login`.
2. Whether your Hugging Face account has access to `facebook/sam3`.
3. Whether the checkpoint file exists at `checkpoints/sam3/sam3.pt`.
4. Whether the script can import the official SAM3 repository through `PYTHONPATH`.

## License

This repository contains benchmark utilities and configuration files. Dataset files, model checkpoints, and third-party assets are not included.

Please follow the license terms of:

* FIRM benchmark code and data release.
* Official SAM3 repository.
* SAM3 model weights.
* Any third-party datasets used in evaluation.
