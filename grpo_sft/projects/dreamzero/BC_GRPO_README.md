# BC + GRPO Training for Haimiandian Dataset

## 概述

本目录包含集成了GRPO (Group Relative Policy Optimization) 的BC (Behavior Cloning) 训练代码。

## 主要修改

### 1. GRPO模块 ()
- : GRPO损失函数
- : 奖励塑形器，基于轨迹平滑度计算奖励

### 2. 训练基类修改 ()
- 在中初始化GRPO组件
- 在中添加GRPO损失计算
- 支持通过配置参数启用/禁用GRPO

### 3. 训练脚本 ()
- 完整的训练启动脚本
- 使用Wan2.2-TI2V-5B模型（5B参数）
- 4个GPU，DeepSpeed ZeRO-2优化

## 配置参数

### GRPO相关参数
- : 是否启用GRPO（默认true）
- : GRPO损失权重（默认0.1）
- : GRPO损失函数中的奖励缩放（默认1.0）
- : 奖励塑形器中的缩放因子（默认0.01）

### 训练参数
- : 5000
- : 1
- : 4
- : 1e-5
- : true
- : zero2.json

## 使用方法

### 1. 冒泡测试（已完成）


### 2. 启动训练


### 3. 监控训练
训练日志会输出以下指标：
- : 总损失（BC + GRPO）
- : BC损失
- : GRPO损失
- : 平均奖励
- : 奖励标准差

## 预期效果

根据GRPO_INTEGRATION_GUIDE.md：
- BC baseline: 70-75% 成功率
- BC + GRPO: 82-88% 成功率
- 预期提升: 10-15%

## 文件结构



## 备份

原始代码已备份到：
- 代码: 
- 数据: 

## 注意事项

1. **模型要求**: 需要Wan2.2-TI2V-5B模型（约20GB）
2. **GPU要求**: 4个GPU，每个至少80GB显存
3. **数据要求**: haimiandian_50数据集需要包含state信息
4. **训练时间**: 预计5000步需要数小时

## 故障排除

### 如果GRPO计算失败
训练会自动回退到纯BC模式，并在日志中输出警告信息。

### 如果内存不足
- 减少
- 增加
- 减少视频帧数或分辨率

### 如果奖励计算异常
检查数据集中是否包含正确的state信息（16维：left_arm(7) + right_arm(7) + gripper(2)）
