# FIRM Benchmark

Utilities for evaluating robotic manipulation episodes with a DAP-style
benchmark pipeline. The current codebase supports mask-based evaluation with
local SAM3 text prompting, task-specific scoring, and dataset-level reporting.

## Pipeline

1. Prepare an annotation workspace with episode folders, final frames, sampled
   frames, `target_mask.png`, and metadata.
2. Generate object masks with local SAM3.
3. Extract raw mask metrics.
4. Score episodes with task-specific rules.
5. Aggregate SR/CQ/DQ and failure distributions.

Core scripts:

- `firm_benchmark/generate_sam3_masks_real.py`
- `firm_benchmark/mask_metric_extractor.py`
- `firm_benchmark/episode_scorer.py`
- `firm_benchmark/dap_eval.py`

## Local SAM3 Setup

This project expects the local SAM3 setup used in CaP-X:

- SAM3 repo: `~/cap-x/capx/third_party/sam3`
- SAM3 checkpoint: `~/sam3/sam3.pt`

Example invocation:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv --directory ~/cap-x run --no-sync --active \
python firm_benchmark/generate_sam3_masks_real.py \
  --episodes-root annotation_workspace_tape/episodes \
  --camera observation.images.head.color \
  --text-prompt "tape roll in paper box" \
  --device cuda \
  --use-target-mask-selection \
  --apply-ring-filter
```

## Scoring Policy

The current scoring config emphasizes task completion over rigid geometric
matching:

- `Manual`: success is mainly whether the manual/paper is inside the accepted
  target region; DQ focuses on severe folding, jamming, or dropping.
- `Cable`: success is mainly whether the cable is contained; DQ focuses on
  severe tangling/contact issues rather than the natural low compactness of
  cables.
- `Box`: success is mainly whether the formed box lies in the target region;
  DQ focuses on severe collapse/deformation/contact issues.
- `Tape`: success is whether at least one tape roll is in the paper box.
- `Sponge`: success uses object-in-target ratio with a light pose component.

See `configs/scoring_config.json` for thresholds and weights.

## Generated Files

Large generated folders are intentionally ignored by git:

- `annotation_workspace*/`
- `results*/`
- raw dataset/video/checkpoint files

Use `BENCHMARK_RUN_SUMMARY.md` for the latest compact result table instead of
committing large result images or intermediate masks.

## Tested Tasks

The latest local run covered five task lines:

- Sponge
- Tape
- Manual1
- Cable&Mouse
- Manufacturing_Lines_Box

See `BENCHMARK_RUN_SUMMARY.md` for metrics and artifact locations.

