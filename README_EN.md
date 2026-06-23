# ACT and Pi05 Real-Robot Training and Deployment

This repository packages a real-robot training and deployment workflow built on top of `LeRobot`, with a focus on:

- `ACT` policy training and inference
- `Pi05` policy training and inference
- `tianqing_a2` robot integration
- asynchronous inference and RTC-based real-robot deployment

The `tianqing_a2` name is kept only as a device interface identifier. It does not encode author, lab, company, or institutional attribution.

## Repository Layout

Key components are located in:

```text
src/lerobot/robots/tianqing_a2
src/lerobot/policies/act
src/lerobot/policies/pi0
src/lerobot/policies/rtc
```

## Installation

Python 3.10 is recommended.

```bash
pip install -e .
pip install -e ".[async]"
pip install -e ".[tianqing_a2]"
```

For vision and hardware execution, prepare the required camera drivers, ROS2 environment, and message definitions in your local setup.

## Start the Policy Server

```bash
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=10 \
  --inference_latency=0.12 \
  --obs_queue_timeout=1.0
```

## Start the Robot Bridge

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

Adjust ROS2 topics, message types, network settings, and timing parameters for your own hardware setup.

## Example Client Command

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
## FIRM Benchmark and Simulation Assets

In addition to the ACT / Pi05 real-robot training and deployment pipeline, this repository also includes FIRM-related benchmark and simulation utilities. These components are provided as independent modules and can be used for evaluation, visualization, and simulation-based analysis of industrial flexible-object robot manipulation.

### FIRM Benchmark Evaluation Utilities

The `FIRM-benchmark/` directory contains the evaluation utilities for the FIRM benchmark. FIRM focuses on industrial flexible-object and mixed-stiffness manipulation tasks, such as manipulating manuals, cables, sponge pads, tape rolls, and cardboard boxes under production-line constraints.

The benchmark utilities support a DAP-style evaluation pipeline, including:

- object mask generation and mask-based metric extraction;
- task-specific episode scoring;
- success rate, completion quality, and deformation-aware quality aggregation;
- failure-mode diagnosis for flexible-object manipulation;
- compact benchmark result reporting.

The core purpose of this module is to evaluate robot policies beyond binary success. Instead of only checking whether a task succeeds or fails, the benchmark also analyzes partial completion, deformation quality, perturbation robustness, and physical failure modes such as folding, slipping, rolling, jamming, residual compression, and rebound-induced displacement.

The main directory is:

```text
FIRM-benchmark/
## Notes

- Paths, IPs, ports, and ROS2 topics in this document are examples only.
- The repository is being prepared as a clean standalone release without prior Git history.
- A Chinese version of this document is available in `README.md`.
