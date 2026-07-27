# Genesis FIRM-Sim 中文说明

本目录提供 FIRM 五个工业柔性物体操作任务的 Genesis World 场景实现，重点是可复现的场景搭建、物理交互和扰动配置。这里不包含策略训练、VLA 接入或 FIRM benchmark 的评估流水线。

## 已实现的场景

| 场景名称 | 操作对象 | 盒盖初始状态 |
| --- | --- | --- |
| `instruction_manual` | 五层纸张组成、中央带折痕合页的说明书 | 向外打开 |
| `sponge_pad` | 薄型 PBD 可变形泡棉垫 | 向外打开 |
| `tape_manipulation` | 质量匹配的空心刚性胶带卷 | 向外打开 |
| `cable_manipulation` | 与刚性鼠标物理连接的柔性成束线缆 | 向外打开 |
| `box_folding` | 带可抓取加长边和顶部合页的纸盒 | 关闭 |

五个场景共用桌子、纸盒、目标区域和天擎机器人位置。待操作物体默认放在桌面上、纸盒正前方的中心线上。

## 安装

```bash
git clone https://github.com/RainFallsDown/FIRM.git
cd FIRM/simulation/genesis_firm_sim
git lfs pull --include="simulation/genesis_firm_sim/tianqing_urdf.zip"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

检查 Genesis 版本并列出可用场景：

```bash
python -c "import genesis as gs; print(gs.__version__)"
python scripts/list_task_scenes.py
```

## 交互运行

直接启动说明书场景：

```bash
python scripts/launch_interactive_scene.py --scene instruction_manual
```

将 `instruction_manual` 替换为 `sponge_pad`、`tape_manipulation`、`cable_manipulation` 或 `box_folding` 即可查看其他场景。

所有场景直接使用当前正式物理参数，包括尺寸、质量、摩擦、柔顺性、阻尼、碰撞和求解器设置；发布包不再包含旧版本切换入口。

## 扰动设置

扰动支持固定随机种子，因此同一组参数可以重复生成完全相同的初始条件：

```bash
python scripts/launch_interactive_scene.py \
  --scene sponge_pad \
  --perturbation-level high \
  --perturbation-axis object_translation \
  --seed 22
```

扰动等级包括 `nominal`、`low`、`medium` 和 `high`。扰动轴包括 `none`、`object_translation`、`fixture_translation`、`object_yaw`、`pose_noise`、`rgb_noise`、`depth_noise` 和 `combined`。

## 离屏渲染

```bash
python scripts/render_scene_snapshot.py \
  --scene instruction_manual \
  --camera-preset overhead \
  --output outputs/instruction_manual.png
```

程序会同时输出 PNG 图像和 JSON 元数据。JSON 中记录场景名称、物理参数集、随机种子和具体扰动值。

## 测试

```bash
pytest -q
```

测试覆盖场景注册、扰动可复现性、求解器配置、质量目标、空心胶带几何、五层说明书、盒盖初始状态、桌面接触和物体摆放位置。

## 目录结构

```text
firm_sim/
  perturbations.py          扰动采样与感知噪声
  physical_parameters.py   当前正式物理参数
  runtime.py                Genesis 初始化
  scenes/
    workspace.py            共用工作台与对象构建逻辑
  tasks/                    任务注册与统一构建接口
scripts/
  launch_interactive_scene.py
  list_task_scenes.py
  render_scene_snapshot.py
tests/
docs/
  physics_parameters.md     物理参数、求解器和模型边界
  scene_parameters.md       实物尺寸与摆位记录
tianqing_urdf.zip           天擎机器人模型（Git LFS）
```

机器人压缩包会在首次运行时解压到 `assets/tianqing_urdf/`。MJCF 和 OBJ 碰撞代理会根据代码中的参数在运行时生成，不需要手工准备。

## 当前模型边界

线缆和薄泡棉目前使用 Genesis PBD 代理。线缆与刚性鼠标保持物理连接；薄泡棉能够表现平面弯曲、自由落体和桌面接触，但暂不表示三维压缩与力级别的弹性恢复。胶带滚动阻力目标值和线缆弯曲刚度目标值均被显式记录，但 Genesis 当前接口中的等效实现仍应结合 [`docs/physics_parameters.md`](docs/physics_parameters.md) 理解。
