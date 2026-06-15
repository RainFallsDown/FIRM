# FIRM PyBullet 项目骨架

这个项目是一个轻量级的 PyBullet 骨架，整体组织方式参考了 `deformable-ravens`，但面向论文 `588_FIRM_A_Benchmark_for_Indus.pdf` 中提出的 5 个 FIRM 任务族进行了通用化设计。

当前阶段的目标是先把架构搭好，而不是立即复现完整任务。现在这 5 个任务族都已经完成注册，并能通过统一的环境、rollout 和评估流程以占位任务的形式运行。

## 任务族

- `instruction_manual_insertion`
- `cable_manipulation`
- `box_folding`
- `sponge_pad_placement`
- `tape_manipulation`

## 项目结构

- `firm_sim/`：核心 Python 包
- `assets/`：后续放置 URDF、mesh、纹理和场景资源
- `configs/`：环境、评估和任务元数据配置
- `scripts/`：命令行入口脚本
- `tests/`：用于验证注册表和流程的冒烟测试
- `docs/`：架构说明文档

## 快速开始

查看已注册任务：

```bash
python scripts/list_tasks.py
```

以无界面模式运行一个占位任务：

```bash
python scripts/run_task.py --task cable_manipulation --headless
```

运行通用评估骨架：

```bash
python scripts/evaluate.py --task tape_manipulation --policy no_op
```

运行冒烟测试：

```bash
python -m unittest discover -s tests
```

## 当前状态

- 通用 PyBullet 环境封装：已完成
- 通用 rollout 流程：已完成
- DAP 风格指标插槽：已搭好骨架
- 5 个 FIRM 任务族：已注册为占位任务
- 具体场景、物体、奖励和工业容差规则：暂未实现

## 当前设计共识

这一节记录的是当前阶段基于论文阅读和场景抽象已经达成的共识，用来帮助后续实现者快速理解我们准备如何搭建 FIRM 的仿真 `setup`。这里总结的是建模口径，不是最终场景规范。

### 场景分层

当前我们把每个任务场景先拆成两侧：

- `fixed / rigid environment`
- `manipulated object`

其中第一阶段会优先抽象 `setup` 所需的 3 类环境资产 class，再补具体操作对象 class。

这 3 类环境资产是：

- `table / work surface`
- `target region marker`
- `rigid boundary / fixture / receptacle`

### 已确定的建模口径

- 论文中的 5 个核心任务是：
  - `instruction manual insertion`
  - `cable manipulation`
  - `box folding`
  - `sponge pad placement`
  - `tape manipulation`
- `rigid boundary / fixture / receptacle` 在第一版统一按“固定在世界坐标系中的 rigid environment object”处理。
- 除 `Box Folding` 外，其余 4 个任务中的 `box / tray / receptacle / outer package` 都归入 rigid environment。
- `Box Folding` 中的 `box` 不属于 rigid environment，而属于被操作对象，当前建模口径应为 `articulated / semi-rigid manipulated object`。
- 其余主要操作对象先统一归到 `manipulated non-rigid object` 一侧，包括：
  - `manual`
  - `cable`
  - `sponge`
  - `tape`

### 任务与对象划分

| 任务 | environment assets | manipulated object assets |
| --- | --- | --- |
| `instruction manual insertion` | `table / work surface`、`target region marker`、`slot / receptacle / outer package` | `manual` |
| `cable manipulation` | `table / work surface`、`target region marker`、`boundary / tray / receptacle / fixture` | `cable` |
| `box folding` | `table / work surface`、`target region marker`、可选的 `support fixture` | `articulated / semi-rigid box` |
| `sponge pad placement` | `table / work surface`、`target region marker`、`tray / receptacle / outer package` | `sponge` |
| `tape manipulation` | `table / work surface`、`target region marker`、`tray / receptacle / fixture / outer package` | `tape` |

### 当前未定项

以下内容目前还没有定下来，后续会在实现任务时逐步补齐：

- 具体 CAD / mesh 来源
- 精确尺寸
- 物理参数
- 任务奖励与阈值
- `deformable` / `articulated` 的具体建模方式
