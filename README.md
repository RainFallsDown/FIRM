# FIRM

FIRM 是一个围绕工业柔性物体操作整理的机器人代码仓库，包含真机训练/部署、仿真与 benchmark、GRPO/SFT 实验代码三条主线。

## 目录入口

```text
real_robot/       真机训练、Tianqing A2 部署、RTC 推理入口
simulation/       Genesis FIRM-Sim 场景、FIRM benchmark、视频 benchmark
grpo_sft/         ACT/Pi0.5、DreamZero、LingBot-VA 的 SFT/GRPO 实验代码
archive/          原始压缩包和历史归档
src/lerobot/      可安装 Python 包，保留 LeRobot fork 的实际源码
examples/         LeRobot 示例与 RTC 示例
tests/            单元测试和集成测试
docs/             仓库结构补充说明
```

## 快速开始

建议使用 Python 3.10。

```bash
pip install -e .
pip install -e ".[async]"
pip install -e ".[tianqing_a2]"
```

真机部署从 [real_robot/README.md](real_robot/README.md) 开始。

仿真和评测从 [simulation/README.md](simulation/README.md) 开始。

GRPO/SFT 实验从 [grpo_sft/README.md](grpo_sft/README.md) 开始。

## 设计说明

`src/lerobot/` 仍然保留为 Python 包源码位置。`tianqing_a2` 等机器人实现没有强行搬出 `src/`，因为 CLI、配置注册、测试和外部导入都依赖这个包路径。顶层功能目录提供清晰入口、运行说明和实验组织，不破坏可安装性。

## Git Hygiene

不要提交本地数据、输出、模型权重、日志、缓存、解压临时目录或机器私有配置。原始 GRPO/SFT 压缩包保存在 `archive/`，日常使用请看 `grpo_sft/` 中整理后的源码。
