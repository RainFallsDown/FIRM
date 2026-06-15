# 架构说明

## 总览

这个项目把“可复用的 benchmark 基础设施”和“任务本身的实现细节”分开管理。

```text
scripts/*
  -> firm_sim.registry.make_task()
  -> firm_sim.env.BenchmarkEnv
  -> firm_sim.policies.make_policy()
  -> firm_sim.rollout.rollout()
  -> firm_sim.evaluation.Evaluator
```

## 核心模块

- `firm_sim/env.py`
  - 管理 PyBullet 的连接和关闭生命周期
  - 负责 `reset`、`step`、`render` 和 episode 计数
  - 暴露稳定的 `BenchmarkEnv.reset(task)` 接口

- `firm_sim/tasks/base.py`
  - 定义统一的任务抽象接口
  - 保存任务元数据、占位状态和共享默认行为

- `firm_sim/tasks/*.py`
  - 每个 FIRM 任务族一个模块
  - 先注册元数据和未来的扩展入口

- `firm_sim/registry.py`
  - 把任务名映射到任务构造器
  - 提供 `make_task(name: str) -> Task`

- `firm_sim/policies/`
  - 提供最小策略接口
  - 当前包含一个 `no_op` 占位策略，用来验证流程

- `firm_sim/rollout.py`
  - 以任务无关的方式运行单个 episode
  - 输出结构化的 episode 结果

- `firm_sim/evaluation/`
  - 封装重复 rollout 的评估逻辑
  - 预留 DAP 风格的指标插槽：
    - `binary_success`
    - `completion_quality`
    - `deformation_quality`
    - `robustness`

## 占位任务行为

当前阶段，每个任务都可以完成元数据初始化并接入环境，但真正的场景构建还没有实现。因此在调用 `task.reset()` 时，会主动抛出清晰的 `NotImplementedError`。命令行脚本会捕获这个错误，并输出可读的占位提示，而不是直接给出一大段难读的崩溃栈信息。

## 后续扩展方向

- 在 `assets/` 下加入任务相关的 URDF、mesh 和贴图资源
- 将当前仅包含元数据的任务配置替换为可执行的场景配置
- 增加 scripted policy 或 oracle policy，方便调试
- 为不同任务补充 DAP 的具体计算逻辑和扰动测试流程
