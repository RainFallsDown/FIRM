# ACT and Pi05 Real-Robot Training and Deployment

本仓库整理了基于 `LeRobot` 的真机训练与部署流程，重点覆盖：

- `ACT` 策略训练与推理
- `Pi05` 策略训练与推理
- `tianqing_a2` 机器人接入
- 异步推理与 RTC 场景下的真机部署

仓库中的 `tianqing_a2` 名称仅作为设备接口标识使用，不包含作者、单位或组织归属信息。

## 目录概览

核心实现位于以下路径：

```text
src/lerobot/robots/tianqing_a2
src/lerobot/policies/act
src/lerobot/policies/pi0
src/lerobot/policies/rtc
```

## 环境安装

建议使用 Python 3.10。

```bash
pip install -e .
pip install -e ".[async]"
pip install -e ".[tianqing_a2]"
```

如果需要运行视觉相关流程，还需要按设备情况准备相机驱动、ROS2 环境和消息定义。

## 服务端启动

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=10 \
  --inference_latency=0.12 \
  --obs_queue_timeout=1.0
```

## 机器人侧桥接

```bash
python3 src/lerobot/robots/tianqing_a2/tianqing_a2_ros2.py \
  --ros_joint_state_topic=/joint_states \
  --ros_hand_state_topic=/hand_states \
  --ros_joint_cmd_topic=/joint_commands \
  --ros_hand_cmd_topic=/hand_commands \
  --ros_joint_cmd_model_topic=/joint_commands_model \
  --ros_model_action_topic=/model_action \
  --state_port=5555 \
  --action_port=6666 \
  --state_topic=tianqing_state \
  --action_topic=tianqing_action
```

请根据你的 ROS2 topic、消息类型和部署网络修改参数。

## 客户端示例

```bash
python examples/rtc/eval_with_real_robot.py \
  --policy.path=/path/to/pi05_or_act_checkpoint \
  --policy.device=cuda \
  --rtc.enabled=true \
  --rtc.execution_horizon=20 \
  --robot.type=tianqing_a2 \
  --robot.robot_ip=127.0.0.1 \
  --robot.state_port=5555 \
  --robot.action_port=6666 \
  --robot.id=tianqing_a2 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="Pick and place the target object" \
  --duration=120
```
## FIRM Benchmark 与仿真资源

除 ACT / Pi05 真机训练与部署流程外，本仓库还包含与 FIRM 相关的 benchmark 评测工具和仿真视频资源。这些模块与真机部署代码相互独立，可用于工业柔性物体操作任务的评测、可视化和仿真分析。

### FIRM Benchmark 评测工具

`FIRM-benchmark/` 目录包含 FIRM benchmark 的评测工具。FIRM 面向工业场景中的柔性物体与刚柔性交互操作任务，例如说明书、电缆、海绵垫、胶带卷和纸箱等对象在产线约束下的抓取、插入、放置、折叠和整理任务。

该模块支持 DAP 风格的评测流程，主要包括：

- 目标物体 mask 生成与 mask-based 指标提取；
- 面向不同任务的 episode scoring；
- 成功率、完成质量和形变感知质量统计；
- 柔性物体操作失败模式诊断；
- 紧凑 benchmark 结果报告生成。

该模块的核心目标是让机器人操作评测不只停留在二值成功率，而是进一步分析任务部分完成情况、形变质量、扰动鲁棒性以及物理失败模式，例如折叠、滑移、滚动、卡滞、残余压缩和回弹位移等问题。

主目录为：

```text
FIRM-benchmark/
## 说明

- 文档中的路径、IP、端口和 topic 仅为示例。
- 仓库已移除旧 Git 历史中可能暴露身份的信息，适合以新的干净仓库形式发布。
- 若需要英文说明，请参考 `README_EN.md`。
