# FIRM-Sim Scene Build Plan

这份文档记录当前仓库现状，以及我们把 `firm-sim` 从“骨架”推进到“5 个场景都能搭起来并可视化运行”的建议路径。

## 1. 当前状态

- 已有通用环境封装：`firm_sim/env.py`
- 已有任务注册表：`firm_sim/registry.py`
- 已有运行入口：`scripts/run_task.py`
- 已有 5 个任务 family 的配置项：`configs/tasks/*.yaml`
- `instruction_manual_insertion` 已经有一个最小场景原型
- 其余 4 个任务当前还是 placeholder
- `assets/` 目录目前基本为空，说明当前方案主要依赖基础几何体搭场景

这意味着我们现在最合适的策略不是先追求高保真，而是先把每个场景都做成：

1. 能加载
2. 能在 GUI 里看见
3. 核心工位关系正确
4. 主要操作对象有一个可替代 proxy

## 2. 目标边界

当前阶段不接入 VLA，不追求策略成功率，只追求：

- 5 个任务都能 `scene-only` 方式运行
- 每个任务都有最小可辨识的工位布局
- 每个任务的对象类别和交互关系与论文设定一致
- 后续如果要接控制或评估，不需要推倒重来

## 3. 推荐搭建顺序

推荐按下面顺序逐个实现：

1. `instruction_manual_insertion`
2. `sponge_pad_placement`
3. `tape_manipulation`
4. `cable_manipulation`
5. `box_folding`

原因：

- `instruction_manual_insertion` 已经有原型，适合先收敛公共搭建模式
- `sponge_pad_placement` 和 `tape_manipulation` 可以先用简单 proxy 建出稳定场景
- `cable_manipulation` 需要处理细长体布局，但仍可先做近似版本
- `box_folding` 最晚做，因为它最可能需要 articulated 建模

## 4. 每个场景的最小可用定义

### `instruction_manual_insertion`

- 环境：
  - table
  - outer package / receptacle
  - target marker
- 操作对象：
  - manual proxy

当前已经具备雏形，接下来主要是整理参数、统一命名和补配置化。

### `sponge_pad_placement`

- 环境：
  - table
  - open box / tray
  - target marker
- 操作对象：
  - sponge proxy，先用 soft-looking box 代理

第一版不需要真实可压缩，只要外观和摆位合理即可。

### `tape_manipulation`

- 环境：
  - table
  - box / tray / fixture
  - target marker
- 操作对象：
  - tape proxy，第一版可用薄环或多个小刚体近似

第一版优先把闭环对象“看起来像胶带卷”。

### `cable_manipulation`

- 环境：
  - table
  - tray / boundary / fixture
  - target marker
- 操作对象：
  - cable proxy，第一版可用一串 capsule / cylinder 近似

先实现静态场景可视化，不急着做高质量柔性物理。

### `box_folding`

- 环境：
  - table
  - optional support fixture
  - target marker
- 操作对象：
  - unfolded cardboard proxy

第一版可以先做成“展开纸盒板件 + 目标区域”，第二版再补关节或折页关系。

## 5. 统一实现模式

建议所有任务都沿用同一套结构：

- `reset()`：
  - 清空 `asset_ids`
  - 重置相机
  - 搭桌面
  - 搭容器 / 边界 / 治具
  - 搭 target marker
  - 搭 manipulated object proxy

- `reward()`：
  - 当前统一返回场景状态字段

- `done()`：
  - 当前统一用 `max_steps`

也就是说，这个阶段每个任务重点只在 `reset()`。

## 6. 第一轮该做的公共抽象

在继续堆任务前，建议先补一个简单的公共 scene helper，减少重复代码：

- 创建桌面
- 创建目标 marker
- 创建盒状 receptacle / tray
- 设置 GUI 相机
- 为常见几何体封装颜色和尺寸接口

这样后面 5 个任务会快很多，也更一致。

## 7. 眼下最值得先做的事

最自然的下一步是：

1. 把 `instruction_manual_insertion` 整理成“公共 helper + 配置化参数”的标准模板
2. 立刻照这个模板实现 `sponge_pad_placement`

这样我们很快就能拥有 2 个真正可视化的场景，后面再复制模式到另外 3 个任务。
