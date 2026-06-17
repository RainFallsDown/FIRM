# FIRM Benchmark 评估系统 - 工作进度同步（2026-05-14）

## 项目目标
- 在服务器 `rsy-prod` 上完整运行 FIRM benchmark 评估流程。
- 使用真实 SAM3 模型生成 object masks，并输出 Sponge 任务 DAP 指标。

## 关键路径
- 项目路径: `/home/kemove/FIRM-benchmark`
- 本地 SAM3 模型目录: `/home/kemove/sam3`
- SAM3权重: `/home/kemove/sam3/sam3.pt`
- SAM3代码路径: `/home/kemove/cap-x/capx/third_party/sam3`

## 当前状态
- 已完成：
  - 标注工作区构建（50 episodes）
  - AV1 视频帧提取（`extract_frames_pyav.py`）
  - 全流程联调（使用 dummy masks）
  - 指标提取、打分、报告导出流程验证
- 已完成（更新）：
  - `firm_benchmark/generate_sam3_masks_real.py` 已可用，不再是“待完成”状态。
  - 默认即对接本机真实 SAM3 路径：
    - `--checkpoint` 默认 `/home/kemove/sam3/sam3.pt`
    - `--sam3-repo` 默认 `/home/kemove/cap-x/capx/third_party/sam3`
    - `--text-prompt` 默认 `white paper`
  - 脚本不会创建或覆盖 `target_mask.png`，严格要求已有真实 target mask。

## 使用真实 SAM3 的执行命令

### 1) 生成真实 masks
```bash
cd /home/kemove/FIRM-benchmark
UV_CACHE_DIR=/tmp/uv-cache uv --directory /home/kemove/cap-x run --no-sync --active \
  python /home/kemove/FIRM-benchmark/firm_benchmark/generate_sam3_masks_real.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --checkpoint /home/kemove/sam3/sam3.pt \
  --sam3-repo /home/kemove/cap-x/capx/third_party/sam3 \
  --text-prompt "white paper" \
  --camera observation.images.head.color \
  --device cuda
```

### 2) 重跑指标提取
```bash
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --config configs/metric_extraction_config.json \
  --output annotation_workspace_sponge/raw_episode_metrics_mask.jsonl
```

### 3) 重跑打分
```bash
python firm_benchmark/episode_scorer.py \
  --input annotation_workspace_sponge/raw_episode_metrics_mask.jsonl \
  --config configs/scoring_config.json \
  --output annotation_workspace_sponge/scored_annotations.jsonl
```

### 4) 生成最终报告
```bash
mkdir -p results_sponge
python firm_benchmark/dap_eval.py \
  --input annotation_workspace_sponge/scored_annotations.jsonl \
  --output-json results_sponge/summary.json \
  --output-csv results_sponge/metrics_table.csv \
  --output-md results_sponge/report.md
```

## 备注
- 旧进度文档中“`generate_sam3_masks_real.py` 待完成”已过期，现已同步为“已完成并可直接运行”。
- 第一阶段使用 CaP-X 的 `uv` 环境运行本地 SAM3；若已有激活好的等价环境，也可以直接运行脚本。
