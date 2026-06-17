# FIRM Benchmark 评分说明（当前代码实现）

本文档对应当前仓库实现，说明从 `mask` 到 `SR/CQ/DQ` 的完整评分链路。  
涉及脚本：
- `firm_benchmark/mask_metric_extractor.py`
- `firm_benchmark/episode_scorer.py`
- `firm_benchmark/dap_eval.py`
- `configs/metric_extraction_config.json`
- `configs/scoring_config.json`

## 1) 总体流程

1. 从每个 episode 读取：
   - `masks/object/*.png`（物体 mask 序列）
   - `target_mask.png`（目标区域 mask）
   - `actions.npy`（可选，用于 timeout/jammed 推断）
   - `meta.json`（任务类型、方法名等）
2. `mask_metric_extractor.py` 产出每集原始指标（`raw_episode_metrics_*.jsonl`）。
3. `episode_scorer.py` 将原始指标映射为每集：
- `success`（0/1）
   - `completion_quality`（CQ，0~1）
   - `q_def.score`（DQ，0~1）
   - `failure_mode`
4. `dap_eval.py` 按数据集聚合：
   - `SR` = 平均 `success`
   - `CQ` = 平均 `completion_quality`
   - `DQ` = 平均 `q_def.score`
   - 失败模式分布、可选鲁棒性指标

## 2) 原始指标提取（mask_metric_extractor）

### 2.1 通用几何量

以 `first_mask`（第一帧）、`final_mask`（最后一帧）、`target_mask` 为核心：
- `object_area`：`final_mask` 前景像素数
- `target_area`：`target_mask` 前景像素数
- `overlap_area`：`final_mask ∩ target_mask` 前景像素数
- `object_target_overlap_ratio = overlap_area / object_area`
- `target_region_coverage = overlap_area / target_area`
- `pose_error`：`final_mask` 与 `target_mask` 质心距离 / 图像对角线长度
- `initial_compactness` / `final_compactness`：`4πA / P^2`

### 2.2 Sponge 任务特有指标

在通用量基础上，额外计算：
- `residual_compression = clamp(1 - final_area / first_area, 0, 1)`
- `rebound_shift = pose_error`
- `folded_corner = (final_compactness < sponge_folded_compactness)`  
  其中阈值来自 `configs/metric_extraction_config.json`，当前为 `0.20`
- `trapped_corner`：当前实现固定为 `False`
- `dropped = (object_area < min_visible_area)`，当前阈值 `20`

### 2.3 timeout / jammed 推断

- `timeout`：若存在 `actions.npy` 且步数 `>= max_steps`（当前 `900`）则为 `True`
- `jammed`：若动作平均范数 `>= min_action_norm`（当前 `1e-3`），但首末质心位移 `< min_object_motion_px`（当前 `5.0`）则为 `True`

## 3) 单集评分（episode_scorer）

## 3.1 Sponge 的 completion（CQ）

配置来自 `configs/scoring_config.json`：
- `w_inside = 0.7`
- `w_pose = 0.3`
- `max_pose_error = 0.05`

计算：
- `pose_score = clamp(1 - pose_error / max_pose_error, 0, 1)`
- `completion = clamp(w_inside * object_target_overlap_ratio + w_pose * pose_score, 0, 1)`

这里 `object_target_overlap_ratio = overlap_area / object_area`，表示物体有多少比例在目标区域内。  
这比 `target_region_coverage = overlap_area / target_area` 更适合“把白纸/海绵放进盒子或目标区域”的任务，因为目标区域通常比物体大，不应该要求物体填满整个目标区域。

## 3.2 Sponge 的 deformation quality（DQ）

配置阈值：
- `max_residual_compression = 0.15`
- `max_rebound_shift = 0.05`

中间量：
- `e_tol = clamp((folded_corner + trapped_corner + residual_compression/max_residual_compression + rebound_shift/max_rebound_shift)/4, 0, 1)`
- `e_contact = 1` 当 `dropped or jammed`，否则 `0`

最终：
- `q_def = 1 - clamp(0.7 * e_tol + 0.3 * e_contact, 0, 1)`

## 3.3 Sponge 的 success（SR 用到）

当前实现中，Sponge 单集 `success=True` 需同时满足：
- `object_target_overlap_ratio >= min_object_in_target_ratio`（当前 `0.80`）
- `pose_error <= max_pose_error`（`0.05`）
- `folded_corner == False`
- `trapped_corner == False`
- `residual_compression <= max_residual_compression`（`0.15`）
- `rebound_shift <= max_rebound_shift`（`0.05`）
- `dropped == False`
- `jammed == False`
- `timeout == False`

只要任一条件不满足，该集 `success=False`。

## 4) 数据集聚合（dap_eval）

对所有 episode：
- `SR = mean(success)`
- `CQ = mean(completion_quality)`
- `DQ = mean(q_def.score)`（若该字段存在）

失败模式分布统计对象：
- 所有 `success=0` 或 `completion_quality<1` 的 episode

## 5) 你当前最需要注意的两点

1. `target_mask.png` 决定评测是否可信  
   评分直接用 `target_mask` 计算覆盖率和位姿误差。  
   如果 `target_mask` 是模板中心框，分数没有可信物理意义。

2. `jammed` 对 SR 影响很大  
   即使 `CQ` 很高，如果触发 `jammed=True`，`success` 仍会变成 `False`。

## 6) 推荐命令（严格评测）

```bash
# 1) 审计 target 是否仍是模板框
python firm_benchmark/validate_target_masks.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --output-json annotation_workspace_sponge/target_mask_audit.json

# 2) 严格版本地 SAM3 生成（要求 target_mask 真实存在）
UV_CACHE_DIR=/tmp/uv-cache uv --directory /home/kemove/cap-x run --no-sync --active \
  python /home/kemove/FIRM-benchmark/firm_benchmark/generate_sam3_masks_real.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --checkpoint /home/kemove/sam3/sam3.pt \
  --sam3-repo /home/kemove/cap-x/capx/third_party/sam3 \
  --text-prompt "white paper" \
  --camera observation.images.head.color \
  --device cuda

# 3) raw metrics
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --config configs/metric_extraction_config.json \
  --output annotation_workspace_sponge/raw_episode_metrics_mask.jsonl

# 4) episode scoring
python firm_benchmark/episode_scorer.py \
  --input annotation_workspace_sponge/raw_episode_metrics_mask.jsonl \
  --config configs/scoring_config.json \
  --output annotation_workspace_sponge/scored_annotations.jsonl

# 5) dataset summary
python firm_benchmark/dap_eval.py \
  --input annotation_workspace_sponge/scored_annotations.jsonl \
  --output-json results_sponge/summary.json \
  --output-csv results_sponge/metrics_table.csv \
  --output-md results_sponge/report.md
```
