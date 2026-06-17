# FIRM Benchmark - 快速参考指南

## 🔧 关键命令速查

### 环境激活
```bash
# lerobot环境（推荐用于数据处理和评估）
source /share/project/rsy/miniconda3/bin/activate lerobot

# sam3环境（用于SAM3 mask生成）
source /share/project/rsy/miniconda3/bin/activate sam3
```

### 代理设置（如需安装包）
```bash
export https_proxy=http://10.8.36.21:80
export http_proxy=http://10.8.36.21:80
```

### 完整评估流程（一键运行）
```bash
ssh rsy-prod << 'EOF'
source /share/project/rsy/miniconda3/bin/activate lerobot
cd /share/project/rsy/firm

# 1. 准备工作空间
python firm_benchmark/prepare_lerobot_annotations.py \
  --dataset-root /share/project/rsy/tianqing/tianqing_data/test_data_rain/test_data_haimiandian/A2p_dataset_haimiandian_resize_1776393366 \
  --output-root annotation_workspace_sponge

# 2. 提取视频帧
python firm_benchmark/extract_frames_pyav.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --sample-interval 10

# 3. 生成masks（当前使用虚拟masks）
python firm_benchmark/create_dummy_masks_server.py \
  --episodes-root annotation_workspace_sponge/episodes

# 4. 提取指标
python firm_benchmark/mask_metric_extractor.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --config configs/metric_extraction_config.json \
  --output annotation_workspace_sponge/raw_episode_metrics_mask.jsonl

# 5. 计算分数
python firm_benchmark/episode_scorer.py \
  --input annotation_workspace_sponge/raw_episode_metrics_mask.jsonl \
  --config configs/scoring_config.json \
  --output annotation_workspace_sponge/scored_annotations.jsonl

# 6. 生成报告
mkdir -p results_sponge
python firm_benchmark/dap_eval.py \
  --input annotation_workspace_sponge/scored_annotations.jsonl \
  --output-json results_sponge/summary.json \
  --output-csv results_sponge/metrics_table.csv \
  --output-md results_sponge/report.md

# 查看报告
cat results_sponge/report.md
EOF
```

---

## 📁 关键文件位置

### 数据集
- **原始数据**: `/share/project/rsy/tianqing/tianqing_data/test_data_rain/test_data_haimiandian/A2p_dataset_haimiandian_resize_1776393366`
  - `data/chunk-000/`: parquet文件（50个episodes）
  - `videos/chunk-000/`: AV1编码视频（6个相机视角）
  - `meta/`: 元数据

### 工作空间
- **主目录**: `/share/project/rsy/firm/annotation_workspace_sponge/`
  - `episodes/`: 50个episode目录
    - `episode_000000/meta.json`: 元数据
    - `episode_000000/actions.npy`: 动作数据
    - `episode_000000/final_frames/`: 最后一帧（6个相机）
    - `episode_000000/sampled_frames/`: 采样帧（6个相机）
    - `episode_000000/masks/object/`: 分割masks
    - `episode_000000/target_mask.png`: 目标区域mask
  - `raw_episode_metrics_mask.jsonl`: 原始指标
  - `scored_annotations.jsonl`: DAP分数

### 结果
- **报告目录**: `/share/project/rsy/firm/results_sponge/`
  - `summary.json`: JSON格式的汇总结果
  - `metrics_table.csv`: CSV格式的指标表
  - `report.md`: Markdown格式的报告

### 脚本
- **已创建脚本**:
  - `extract_frames_pyav.py`: PyAV视频帧提取（已测试✅）
  - `create_dummy_masks_server.py`: 虚拟mask生成（已测试✅）
  - `generate_sam3_masks_real.py`: 本地 SAM3 真实 mask 生成（已可用✅）

### 配置文件
- `configs/metric_extraction_config.json`: 指标提取配置
- `configs/scoring_config.json`: DAP分数计算配置

---

## 🐛 常见问题和解决方案

### 问题1: AV1视频无法解码
**症状**: `ImportError: libGL.so.1: cannot open shared object file`
**原因**: OpenCV需要系统OpenGL库，且AV1编码需要特殊解码器
**解决方案**: 使用PyAV库（已实现）
```bash
# PyAV已在lerobot环境中安装
source /share/project/rsy/miniconda3/bin/activate lerobot
python -c "import av; print(av.__version__)"  # 应输出 15.1.0
```

### 问题2: 任务类型识别失败
**症状**: `[WARN] Unknown task 'Unknown'`
**原因**: meta.json中的task字段为"Unknown"
**解决方案**: 批量更新任务类型
```bash
ssh rsy-prod "cd /share/project/rsy/firm && \
  find annotation_workspace_sponge/episodes -name 'meta.json' \
  -exec sed -i 's/\"task\": \"Unknown\"/\"task\": \"Sponge\"/g' {} \;"
```

### 问题3: numpy版本冲突
**症状**: `sam3 0.1.0 requires numpy<2,>=1.26, but you have numpy 2.4.4`
**原因**: SAM3需要numpy<2，但opencv-python需要numpy>=2
**解决方案**: 在sam3环境中使用opencv-python-headless
```bash
source /share/project/rsy/miniconda3/bin/activate sam3
pip install opencv-python-headless 'numpy<2,>=1.26'
```

### 问题4: 网络连接超时
**症状**: `Connection timed out` 或 `Temporary failure in name resolution`
**原因**: 服务器网络隔离，需要使用代理
**解决方案**: 设置代理环境变量
```bash
export https_proxy=http://10.8.36.21:80
export http_proxy=http://10.8.36.21:80
```

---

## 📊 指标说明

### DAP指标（Data-Agnostic Performance）

| 指标 | 含义 | 范围 | 说明 |
|------|------|------|------|
| SR | Success Rate | [0,1] | 任务成功率 |
| CQ | Completion Quality | [0,1] | 完成质量（目标覆盖率） |
| DQ | Deformation Quality | [0,1] | 变形质量（物体完整性） |
| S_robust | Robustness Score | [0,1] | 鲁棒性分数 |
| PR-AUC | Precision-Recall AUC | [0,1] | 精确率-召回率曲线下面积 |

### Sponge任务特定指标

| 指标 | 含义 |
|------|------|
| target_region_coverage | 目标区域覆盖率 |
| pose_error | 位姿误差（中心点距离） |
| residual_compression | 残余压缩（面积变化） |
| rebound_shift | 回弹位移 |
| folded_corner | 折叠角落 |
| dropped | 物体掉落 |

---

## 🎯 SAM3 Mask生成详细步骤

### 1. 理解SAM3 API
```python
import sam3
from sam3.model.sam3_image_processor import Sam3Processor

# 加载模型
model = sam3.build_sam3_image_model(
    enable_inst_interactivity=True,
    checkpoint_path="/home/kemove/sam3/sam3.pt",
    load_from_HF=False,
)

# 创建处理器
processor = Sam3Processor(model, confidence_threshold=0.3)

# 处理单张图像
image = Image.open("image.png").convert("RGB")
inference_state = processor.set_image(image)

# 添加文本提示，分割白纸
result = processor.set_text_prompt(
    state=inference_state,
    prompt="white paper",
)

# 获取masks
masks = result["masks"]  # shape: (num_masks, H, W)
```

### 2. 实现完整脚本
- 遍历所有episodes
- 对每个episode的采样帧和最终帧生成masks
- 保存为PNG格式（0-255）
- 不创建或覆盖 `target_mask.png`，运行前必须已有真实 target mask

### 3. 运行脚本
```bash
cd /home/kemove/FIRM-benchmark
UV_CACHE_DIR=/tmp/uv-cache uv --directory /home/kemove/cap-x run --no-sync --active \
  python /home/kemove/FIRM-benchmark/firm_benchmark/generate_sam3_masks_real.py \
  --episodes-root annotation_workspace_sponge/episodes \
  --checkpoint /home/kemove/sam3/sam3.pt \
  --sam3-repo /home/kemove/cap-x/capx/third_party/sam3 \
  --text-prompt "white paper" \
  --camera observation.images.head.color \
  --device cuda \
  --max-episodes 5  # 先测试5个episodes
```

### 4. 验证masks
```bash
ls -la /home/kemove/FIRM-benchmark/annotation_workspace_sponge/episodes/episode_000000/masks/object/ | head -10
```

---

## 📝 配置文件模板

### metric_extraction_config.json
```json
{
  "tasks": {
    "Sponge": {
      "min_visible_area": 20.0,
      "sponge_folded_compactness": 0.20,
      "max_steps": 500,
      "min_action_norm": 0.001,
      "min_object_motion_px": 5.0
    }
  }
}
```

### scoring_config.json
```json
{
  "tasks": {
    "Sponge": {
      "w_inside": 0.7,
      "w_pose": 0.3,
      "min_object_in_target_ratio": 0.80,
      "max_pose_error": 0.05
    }
  }
}
```

---

## 🔍 调试技巧

### 查看单个episode的结构
```bash
ssh rsy-prod "tree /share/project/rsy/firm/annotation_workspace_sponge/episodes/episode_000000 -L 2"
```

### 查看meta.json内容
```bash
ssh rsy-prod "cat /share/project/rsy/firm/annotation_workspace_sponge/episodes/episode_000000/meta.json | python -m json.tool"
```

### 查看评估结果
```bash
ssh rsy-prod "cat /share/project/rsy/firm/results_sponge/report.md"
ssh rsy-prod "cat /share/project/rsy/firm/results_sponge/summary.json | python -m json.tool"
```

### 检查mask文件
```bash
ssh rsy-prod "ls -lh /share/project/rsy/firm/annotation_workspace_sponge/episodes/episode_000000/masks/object/ | head -5"
ssh rsy-prod "file /share/project/rsy/firm/annotation_workspace_sponge/episodes/episode_000000/masks/object/000000.png"
```

---

## 📞 关键联系信息

- **服务器**: rsy-prod (SSH)
- **项目负责人**: 需要补充
- **SAM3代码**: `/home/kemove/cap-x/capx/third_party/sam3`
- **SAM3模型**: `/home/kemove/sam3`
- **FIRM代码**: `/share/project/rsy/firm/firm_benchmark/`
