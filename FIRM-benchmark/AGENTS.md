# Project Instructions for Codex

## Permission Settings

**User Preference**: Automatically approve all operations except file modifications.

- ✅ **Auto-approve**: Bash commands, reading files, searching, installing packages, running scripts
- ❌ **Require approval**: Edit, Write, NotebookEdit (any file modification operations)

## Workflow

When working on tasks:
1. Read and analyze code freely
2. Run commands and scripts without asking
3. Install dependencies as needed
4. Only ask for permission when modifying existing files or creating new files

## Project Context

This is the FIRM benchmark project for evaluating robotic manipulation with deformable objects using the Deformation-aware Assessment Protocol (DAP).

### Key Components
- `firm_benchmark/` - Core evaluation scripts
- `configs/` - Configuration files for metric extraction and scoring
- `haimiandian/` - Example dataset (Sponge task, 50 episodes)

### Evaluation Pipeline
1. `prepare_lerobot_annotations.py` - Prepare annotation workspace
2. `mask_metric_extractor.py` - Extract metrics from SAM masks
3. `episode_scorer.py` - Calculate DAP scores
4. `dap_eval.py` - Aggregate dataset-level metrics

### Configuration Files
- `metric_extraction_config.json` - For mask metric extraction
- `scoring_config.json` - For episode scoring
