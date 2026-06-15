# Genesis Advanced IK 笔记

这份笔记整理自 Genesis 文档页面 `Advanced and Parallel IK`，重点记录多末端逆运动学、姿态掩码控制，以及并行环境中的 IK 输入方式，方便后续在本项目里做机器人控制原型时快速查阅。

## 1. 文档主题

原文页面：

- [Advanced and Parallel IK](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/advanced_ik.html)

核心内容分成两部分：

- 多个末端执行器同时求解 IK
- 在并行仿真环境中批量求解 IK

## 2. 多末端 IK 的关键点

Genesis 提供 `robot.inverse_kinematics_multilink()`，用于同时约束多个 link 的目标位姿。

典型输入包括：

- `links`：目标 link 对象列表
- `poss`：每个 link 的目标位置列表
- `quats`：每个 link 的目标四元数列表
- `rot_mask`：控制姿态约束哪些轴生效
- `pos_mask`：控制位置约束哪些坐标轴生效

文档里的示例把 Panda 夹爪的 `left_finger` 和 `right_finger` 当成两个独立目标 link，同时跟踪它们的位置。

## 3. `rot_mask` 和 `pos_mask` 的含义

这页文档最有价值的点，是说明 IK 目标不一定必须是完整的 6-DoF 位姿，可以只约束我们真正关心的部分。

- `rot_mask = [False, False, True]`
  - 表示只约束局部 z 轴方向
  - 不强制 x/y 轴朝向
  - 适合“末端朝下即可，但平面内转角无所谓”的抓取场景

- `pos_mask`
  - 用于选择只约束位置的某些轴
  - 例如只关心平面内位置、不关心高度时会很有用

实践意义：

- 目标定义更柔性，IK 更容易收敛
- 可以减少“不必要但严格”的姿态要求
- 很适合工业操作中“只要方向对、位置对即可”的末端控制

## 4. 调试可视化

文档示例使用了两个调试接口：

- `scene.draw_debug_frame()`
- `scene.update_debug_objects()`

它们的作用是给 IK 目标位姿画可视化坐标系，便于观察求解结果。

需要注意：

- 这些 debug objects 只属于可视化层
- 它们不参与物理仿真
- 连续动画里应优先更新现有对象，而不是每帧删除重建

这对我们后续做轨迹调试、检查末端目标定义是否合理会很有帮助。

## 5. 纯可视化更新 vs 真实控制循环

文档特别区分了两种使用方式：

- 纯演示 / 纯可视化：
  - 直接调用 `robot.set_dofs_position(q)`
  - 然后调用 `scene.visualizer.update()`
  - 不需要 `scene.step()`

- 真实控制 / 物理仿真：
  - 应使用 `robot.control_dofs_position()`
  - 配合 `scene.step()` 推进仿真

这说明：

- `set_dofs_position` 更像“直接改状态”，适合展示和调试
- `control_dofs_position + scene.step()` 才更接近真实控制闭环

如果后面我们把 Genesis 当成控制验证环境，应该优先按第二种方式组织控制循环。

## 6. 并行环境中的 IK

Genesis 支持在 batched / parallel envs 里直接做 IK。

文档示例中：

- `scene.build(n_envs=16, env_spacing=(1.0, 1.0))`
- 一次生成 `16` 个并行环境
- 每个环境里的末端目标位置都不同

关键规则只有一个：

- 所有目标位姿变量都要多一个 batch 维度

例如：

- 单环境位置：`(3,)`
- 并行环境位置：`(n_envs, 3)`

- 单环境四元数：`(4,)`
- 并行环境四元数：`(n_envs, 4)`

文档中的 `robot.inverse_kinematics()` 在并行情况下仍然可用，只要输入已经批量化即可。

## 7. 并行 IK 的直接启发

这对后续大规模 rollout 或策略评估很关键，因为它意味着：

- 不需要手写 Python 循环逐环境求 IK
- 可以让目标位姿直接按 batch 组织
- IK 接口本身就能匹配并行仿真的数据布局

如果我们以后在一个策略里同时控制多份场景，IK 的目标张量设计最好一开始就按 batch-first 方式规划。

## 8. 和当前项目的关联

虽然当前仓库主线还是 PyBullet 骨架，但这页文档提供了几个值得保留的控制设计经验：

- 末端目标不必总是完整 6-DoF，可按任务需要降低约束
- 多指 / 多接触点任务可以把多个 link 一起送进 IK
- 调试阶段应把“目标可视化”作为标准工具
- 如果未来转向并行仿真，IK 输入张量最好天然支持 batch 维

这些思路尤其适合：

- 双指夹持
- 插入类任务
- 需要控制末端方向但不关心偏航角的工业操作

## 9. 可直接记住的结论

- 多末端 IK：用 `robot.inverse_kinematics_multilink()`
- 只约束部分姿态：用 `rot_mask`
- 只约束部分位置：用 `pos_mask`
- 可视化目标位姿：用 `scene.draw_debug_frame()` 和 `scene.update_debug_objects()`
- 演示时可直接设关节位置并刷新可视化
- 真正仿真控制时应走 `control_dofs_position()` 加 `scene.step()`
- 并行 IK 的本质要求是：目标输入带 batch 维度

## 10. 后续可落地方向

如果我们后面要把这套知识落到代码里，比较自然的切入点是：

- 写一个独立的 Genesis IK playground 脚本
- 先验证单末端 / 多末端目标定义
- 再验证 `rot_mask` 对抓取朝向约束的效果
- 最后补一个 batched IK demo，和并行 rollout 设计保持一致
