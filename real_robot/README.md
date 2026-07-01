# Real-Robot Training and Deployment

This folder is the entry point for real-robot work with the Tianqing A2 interface.

The runnable LeRobot package code stays under `src/lerobot/` so that installed CLI commands, imports, and tests keep working. Use this folder as the deployment guide and launcher collection.

## Key Code

```text
../src/lerobot/robots/tianqing_a2/      Tianqing A2 robot interface and ROS2/ZMQ bridge
../src/lerobot/async_inference/         Policy server and async robot client infrastructure
../examples/rtc/eval_with_real_robot.py RTC real-robot inference example
../src/lerobot/scripts/lerobot_train.py Offline training entry point
```

## Install

```bash
pip install -e .
pip install -e ".[async]"
pip install -e ".[tianqing_a2]"
```

Prepare ROS2, camera drivers, robot-side message definitions, and network access for your hardware.

## Train

```bash
lerobot-train \
  --dataset.repo_id=<dataset_repo_id> \
  --dataset.root=/path/to/lerobot_dataset \
  --policy.type=act \
  --output_dir=outputs/train/<run_name> \
  --job_name=<run_name>
```

## Start Policy Server

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=10 \
  --inference_latency=0.12 \
  --obs_queue_timeout=1.0
```

## Start Tianqing A2 Bridge

```bash
python src/lerobot/robots/tianqing_a2/tianqing_a2_ros2.py \
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

## Run RTC Inference

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
