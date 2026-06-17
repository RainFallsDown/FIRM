# FIRM Benchmark 配置文件说明

## 配置文件概览

FIRM benchmark 使用两个独立的配置文件，分别用于评估流程的不同阶段：

### 1. `metric_extraction_config.json`
**用途：** `mask_metric_extractor.py` - 从 SAM mask 提取原始度量指标

**包含参数：**
- 视觉特征阈值（compactness, visible area, overlap ratio）
- 动作检测参数（max_steps, min_action_norm, min_object_motion_px）
- 任务特定的视觉判断标准

### 2. `scoring_config.json`
**用途：** `episode_scorer.py` - 将原始度量转换为 DAP 分数

**包含参数：**
- 任务完成度计算的目标值和权重
- 成功判定阈值
- 变形容忍度上限

---

## 评估流程

```
LeRobot数据集
    ↓
prepare_lerobot_annotations.py
    ↓ 生成 episode 包（meta.json, masks/, target_mask.png）
    ↓
mask_metric_extractor.py --config metric_extraction_config.json
    ↓ 输出 raw_episode_metrics_mask.jsonl
    ↓
episode_scorer.py --config scoring_config.json
    ↓ 输出 scored_annotations.jsonl
    ↓
dap_eval.py
    ↓ 输出 DAP 指标（SR, CQ, DQ, PR-AUC）
```

---

## 参数详解

### Manual 任务（说明书插入）

#### metric_extraction_config.json
```json
{
  "target_orientation_deg": 0.0,        // 目标朝向角度
  "manual_min_compactness": 0.05,       // 严重折叠判定阈值
  "max_steps": 900,                     // 超时步数
  "min_action_norm": 0.001,             // 最小动作幅度
  "min_object_motion_px": 5.0           // 最小物体移动像素
}
```

#### scoring_config.json
```json
{
  "target_insertion_depth": 1.0,        // 目标插入深度（归一化）
  "max_alignment_error": 0.05,          // 最大对齐误差
  "w_depth": 0.7,                       // 插入深度权重
  "w_align": 0.3,                       // 对齐精度权重
  "success_completion": 0.95            // 成功判定的完成度阈值
}
```

### Cable 任务（线缆操作）

#### metric_extraction_config.json
```json
{
  "min_contained_ratio": 0.8,           // 最小包含比例
  "min_visible_area": 20.0,             // 最小可见面积（像素）
  "max_steps": 900,
  "min_action_norm": 0.001,
  "min_object_motion_px": 5.0
}
```

#### scoring_config.json
```json
{
  "target_cable_length": 1.0,           // 目标线缆长度（归一化）
  "max_tangling": 0.3,                  // 最大缠绕度
  "max_boundary_contact": 0.3,          // 最大边界接触
  "success_completion": 0.90
}
```

### Box 任务（纸盒折叠）

#### metric_extraction_config.json
```json
{
  "box_min_compactness": 0.05,          // 严重变形判定阈值
  "max_steps": 900,
  "min_action_norm": 0.001,
  "min_object_motion_px": 5.0
}
```

#### scoring_config.json
```json
{
  "target_fold_angle": 90.0,            // 目标折叠角度（度）
  "max_angle_error": 15.0,              // 最大角度误差（度）
  "success_completion": 0.90
}
```

### Sponge 任务（海绵放置）

#### metric_extraction_config.json
```json
{
  "sponge_folded_compactness": 0.20,    // 折角判定阈值
  "min_visible_area": 20.0,
  "max_steps": 900,
  "min_action_norm": 0.001,
  "min_object_motion_px": 5.0
}
```

#### scoring_config.json
```json
{
  "max_pose_error": 0.05,               // 最大位姿误差
  "max_residual_compression": 0.15,     // 最大残余压缩
  "max_rebound_shift": 0.05,            // 最大回弹位移
  "w_inside": 0.7,                      // 物体在目标区域内比例权重
  "w_pose": 0.3,                        // 位姿精度权重
  "min_object_in_target_ratio": 0.80
}
```

### Tape 任务（胶带操作）

#### metric_extraction_config.json
```json
{
  "target_orientation_deg": 0.0,
  "min_overlap_ratio": 0.3,             // 最小重叠比例
  "max_steps": 900,
  "min_action_norm": 0.001,
  "min_object_motion_px": 5.0
}
```

#### scoring_config.json
```json
{
  "max_position_error": 0.05,           // 最大位置误差
  "max_orientation_error": 20.0,        // 最大朝向误差（度）
  "w_position": 0.6,                    // 位置权重
  "w_orientation": 0.4,                 // 朝向权重
  "success_completion": 0.90
}
```

---

## 使用示例

### 完整评估流程

```bash
# 1. 准备 episode 包
python firm_benchmark/prepare_lerobot_annotations.py \
  --dataset-root /path/to/lerobot_dataset \
  --output-root ./annotation_workspace

# 2. 从 mask 提取原始指标
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root ./annotation_workspace/episodes \
  --config configs/metric_extraction_config.json \
  --output ./annotation_workspace/raw_episode_metrics_mask.jsonl

# 3. 转换为 DAP 分数
python firm_benchmark/episode_scorer.py \
  --input ./annotation_workspace/raw_episode_metrics_mask.jsonl \
  --config configs/scoring_config.json \
  --output ./annotation_workspace/scored_annotations.jsonl

# 4. 聚合为数据集级别指标
python firm_benchmark/dap_eval.py \
  --input ./annotation_workspace/scored_annotations.jsonl \
  --output-dir ./results
```

---

## 参数调优建议

### 调整成功率阈值
如果任务难度较高，可以降低 `success_completion`：
```json
"success_completion": 0.85  // 从 0.90 降低到 0.85
```

### 调整变形容忍度
如果物体更容易变形，可以放宽阈值：
```json
"max_tangling": 0.4,           // 从 0.3 提高到 0.4
"max_residual_compression": 0.20  // 从 0.15 提高到 0.20
```

### 调整权重分配
根据任务重要性调整权重：
```json
// 更重视物体在目标区域内比例而非位姿精度
"w_inside": 0.8,
"w_pose": 0.2
```

---

## 注意事项

1. **归一化值范围：** 大部分误差和比例参数都是归一化的 [0, 1]
2. **角度单位：** 所有角度参数使用**度**（degrees），不是弧度
3. **像素单位：** 视觉相关的阈值（如 `min_visible_area`）使用像素
4. **权重和为1：** 所有权重参数（如 `w_depth + w_align`）应该加起来等于 1.0
5. **默认值：** 如果配置文件中缺少某个参数，代码会使用硬编码的默认值

---

## 故障排查

### 问题：所有 episode 都失败
- 检查 `success_completion` 是否设置过高
- 检查 `max_*_error` 阈值是否过严

### 问题：DQ 分数异常低
- 检查变形容忍度参数（`max_tangling`, `max_residual_compression` 等）
- 验证 mask 质量是否良好

### 问题：配置文件不生效
- 确认使用了正确的配置文件路径
- 检查 JSON 格式是否正确（使用 `jq` 或 JSON 验证器）
- 确认任务名称大小写匹配（"Manual" 不是 "manual"）
