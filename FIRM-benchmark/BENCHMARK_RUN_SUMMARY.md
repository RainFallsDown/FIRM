# Benchmark Run Summary

Latest local benchmark runs using local SAM3 and task-completion-oriented
scoring.

## Results

| Dataset / task line | Task | Episodes | Prompt | SR | CQ | DQ | Result directory |
|---|---:|---:|---|---:|---:|---:|---|
| `haimiandian` / Sponge | Sponge | 10 | `white paper` | 1.0000 | 0.9172 | 0.9518 | `results_sponge_sam3_white_paper_object_inside/` |
| `/media/kemove/tianqing_data/FIRM/Tape` | Tape | 10 | `tape roll in paper box` | 1.0000 | 0.9938 | 1.0000 | `results_tape_sam3_tape_roll_in_paper_box_targetsel_ring_first10/` |
| `/media/kemove/tianqing_data/FIRM/Manual1` | Manual | 10 | `white paper` | 1.0000 | 0.9586 | 0.9673 | `results_manual1_sam3_white_paper_target_inbox_first10/` |
| `/media/kemove/tianqing_data/FIRM/Cable&Mouse` | Cable | 10 | `power cable` | 1.0000 | 0.9995 | 1.0000 | `results_cable_mouse_sam3_power_cable_first10/` |
| `/media/kemove/tianqing_data/FIRM/Manufacturing_Lines_Box/task_id_15_stage_1_2_overall` | Box | 10 | `box` | 1.0000 | 0.9722 | 0.8440 | `results_ml_box_sam3_box_first10/` |

## Notes

- All runs use final frames, not first frames, for final-state evaluation.
- SAM3 masks are selected with optional target-mask-biased candidate selection.
- Tape additionally uses a ring-like component filter to avoid selecting large
  paper-box background regions.
- The scoring rules in `configs/scoring_config.json` prioritize whether the
  intended object reaches the task-defined accepted region. DQ is reserved for
  deformation/contact quality rather than strict pose error.

## Important Generated Artifacts

The following local artifacts are useful for inspection but are ignored by git:

- `results_tape_sam3_tape_roll_in_paper_box_targetsel_ring_first10/all_10_tape_rgb_mask_overlay_ring.png`
- `results_manual1_sam3_white_paper_target_inbox_first10/all_10_manual1_rgb_mask_overlay.png`
- `results_cable_mouse_sam3_power_cable_first10/all_10_cable_mouse_rgb_mask_overlay.png`
- `results_ml_box_sam3_box_first10/all_10_ml_box_rgb_mask_overlay.png`
- `results_sponge_sam3_white_paper_object_inside/all_10_target_vs_sam3_masks.png`

