# FIRM Benchmark 评估流程指南

## 当前进度

✅ **步骤1完成**: 标注工作空间已准备完成
- 50个episodes已处理
- 视频帧已提取
- 动作序列已保存
- 任务类型已设置为 Sponge

## 下一步：生成SAM Masks

### 方法1：使用SAM模型自动生成（推荐）

#### 1.1 安装依赖
```bash
pip install segment-anything opencv-python torch torchvision
```

#### 1.2 下载SAM模型
从 [SAM官方仓库](https://github.com/facebookresearch/segment-anything) 下载checkpoint：
```bash
# 下载 ViT-H 模型 (最大最准确，2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 或者 ViT-L 模型 (较小，1.2GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

#### 1.3 生成object masks
```bash
python firm_benchmark/generate_sam_masks.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --checkpoint sam_vit_h_4b8939.pth \
  --camera observation.images.hand.right.color \
  --sample-interval 10 \
  --create-dummy-target
```

**参数说明：**
- `--checkpoint`: SAM模型文件路径
- `--camera`: 使用哪个相机视角（推荐右手相机）
- `--sample-interval`: 每隔N帧采样一次（减少计算量）
- `--create-dummy-target`: 自动创建虚拟的target_mask（中心区域）
- `--device`: cuda 或 cpu（默认cuda）

**输出：**
```
episodes/episode_XXXXXX/
├── masks/
│   └── object/
│       ├── 000000.png  (第一帧的mask)
│       ├── 000001.png  (采样帧的mask)
│       └── ...
└── target_mask.png     (目标区域mask)
```

---

### 方法2：手动标注（如果没有GPU或SAM模型）

如果无法运行SAM，可以：

1. **使用其他分割工具**：
   - LabelMe
   - CVAT
   - Roboflow

2. **跳过mask提取，直接手动填写指标**：
   编辑 `annotation_workspace_sponge/raw_episode_metrics_template.jsonl`，手动填写每个episode的指标：

```json
{
  "episode_id": "episode_000000",
  "method": "unknown_method",
  "task": "Sponge",
  "metrics": {
    "target_region_coverage": 0.85,
    "pose_error": 0.03,
    "residual_compression": 0.10,
    "rebound_shift": 0.02,
    "folded_corner": false,
    "trapped_corner": false,
    "dropped": false,
    "jammed": false,
    "timeout": false
  }
}
```

然后直接跳到步骤4（episode_scorer.py）。

---

### 方法3：使用VLM进行语义标注

如果有OpenAI API key，可以使用视觉语言模型：

```bash
export OPENAI_API_KEY="your-api-key"

python firm_benchmark/vlm_failure_annotator.py \
  --input annotation_workspace_sponge/raw_episode_metrics_template.jsonl \
  --episodes-root annotation_workspace_sponge/episodes \
  --output annotation_workspace_sponge/raw_episode_metrics_vlm.jsonl \
  --max-images 6
```

---

## 完整评估流程

### 步骤1: 准备工作空间 ✅ 已完成
```bash
python firm_benchmark/prepare_lerobot_annotations.py \
  --dataset-root haimiandian \
  --output-root annotation_workspace_sponge
```

### 步骤2: 生成SAM masks ⏳ 待执行
```bash
python firm_benchmark/generate_sam_masks.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --checkpoint sam_vit_h_4b8939.pth \
  --create-dummy-target
```

### 步骤3: 提取mask指标 ⏳ 待执行
```bash
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --config configs/metric_extraction_config.json \
  --output annotation_workspace_sponge/raw_episode_metrics_mask.jsonl
```

### 步骤4: 计算DAP分数 ⏳ 待执行
```bash
python firm_benchmark/episode_scorer.py \
  --input annotation_workspace_sponge/raw_episode_metrics_mask.jsonl \
  --config configs/scoring_config.json \
  --output annotation_workspace_sponge/scored_annotations.jsonl
```

### 步骤5: 聚合评估指标 ⏳ 待执行
```bash
python firm_benchmark/dap_eval.py \
  --input annotation_workspace_sponge/scored_annotations.jsonl \
  --output-dir results_sponge
```

---

## 快速测试（无需SAM）

如果只是想测试流程，可以创建简单的测试数据：

```bash
# 创建虚拟masks和target_mask
python -c "
import cv2
import numpy as np
from pathlib import Path

episodes_dir = Path('annotation_workspace_sponge/episodes')
for ep_dir in sorted(episodes_dir.iterdir())[:5]:  # 只处理前5个
    if ep_dir.is_dir():
        # 创建object masks目录
        mask_dir = ep_dir / 'masks' / 'object'
        mask_dir.mkdir(parents=True, exist_ok=True)

        # 创建3个虚拟mask（第一帧、中间帧、最后一帧）
        for i in range(3):
            mask = np.zeros((480, 640), dtype=np.uint8)
            # 创建一个圆形mask
            cv2.circle(mask, (320, 240), 100 + i*10, 255, -1)
            cv2.imwrite(str(mask_dir / f'{i:06d}.png'), mask)

        # 创建target_mask
        target = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(target, (220, 140), (420, 340), 255, -1)
        cv2.imwrite(str(ep_dir / 'target_mask.png'), target)

print('[OK] Created dummy masks for 5 episodes')
"

# 然后运行步骤3-5
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --config configs/metric_extraction_config.json \
  --output annotation_workspace_sponge/raw_episode_metrics_mask.jsonl

python firm_benchmark/episode_scorer.py \
  --input annotation_workspace_sponge/raw_episode_metrics_mask.jsonl \
  --config configs/scoring_config.json \
  --output annotation_workspace_sponge/scored_annotations.jsonl

python firm_benchmark/dap_eval.py \
  --input annotation_workspace_sponge/scored_annotations.jsonl \
  --output-dir results_sponge
```

---

## 故障排查

### 问题1: CUDA out of memory
```bash
# 使用CPU模式
python firm_benchmark/generate_sam_masks.py --device cpu ...

# 或增加采样间隔
python firm_benchmark/generate_sam_masks.py --sample-interval 20 ...
```

### 问题2: SAM分割不准确
- 尝试不同的相机视角（`--camera`）
- 调整SAM参数（修改脚本中的 `SamAutomaticMaskGenerator` 参数）
- 手动标注关键帧

### 问题3: 缺少target_mask
- 使用 `--create-dummy-target` 创建虚拟target
- 或手动标注每个episode的目标区域

---

## 输出结果

最终会生成：

```
results_sponge/
├── summary.json          # DAP指标汇总
├── metrics_table.csv     # 详细指标表格
└── report.md            # Markdown报告
```

**关键指标：**
- **SR (Success Rate)**: 成功率
- **CQ (Completion Quality)**: 完成质量 [0,1]
- **DQ (Deformation Quality)**: 变形质量 [0,1]
- **PR-AUC**: 鲁棒性曲线下面积
- **Failure Modes**: 失败模式分布
