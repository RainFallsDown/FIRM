# Genesis FIRM-Sim 搭建计划

这份文档记录根目录主线接下来如何用 `Genesis World` 重建 `firm-sim`。

## 1. 目标

当前阶段只做一件事：

- 把 FIRM 的每一个场景在 Genesis 里搭出来，并且能运行、能显示、能复用

当前明确不做：

- 不接入 VLA
- 不优先追求任务成功策略
- 不优先追求高保真可变形物理

## 2. 仓库约定

- 根目录是 `Genesis` 主线
- `PYBULLET/` 只作为旧方案归档
- 新代码不要继续依赖 `PYBULLET/` 的运行入口

## 3. 第一阶段交付标准

每个任务先满足下面 4 点：

- 能启动 viewer
- 能看到工位和目标区域
- 能看到操作对象的 proxy
- 代码结构支持后续继续细化

## 4. 推荐顺序

1. 建立 Genesis 版本公共场景骨架
2. 先实现 `instruction_manual_insertion`
3. 再实现 `sponge_pad_placement`
4. 然后实现 `tape_manipulation`
5. 然后实现 `cable_manipulation`
6. 最后实现 `box_folding`

## 5. 设计原则

- 第一版先用 proxy 几何体搭关系
- 优先把尺寸、摆位、视角和 target region 定准
- 公共桌面、容器、marker、相机逻辑应尽早抽成 helper
- 每个任务都应有最小可运行 demo

## 6. 当前已完成

- 共享桌子 + 盒子底座
- `instruction_manual` 独立场景
- `sponge_pad` 独立场景
- `tape_manipulation` 独立场景
- `cable_manipulation` 独立场景
- `box_folding` 独立场景
- 统一任务场景注册表

当前架构原则已经明确：

- 每个任务是独立 scene
- 任务之间可以复用桌子和盒子
- 不把说明书、海绵垫等不同任务对象混放到同一个 scene
