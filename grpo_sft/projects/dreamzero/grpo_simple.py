"""
Simplified GRPO Implementation for DreamZero Integration
可以直接在现有训练流程中使用的简化版GRPO
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SimpleGRPOLoss(nn.Module):
    """
    简化的GRPO损失函数
    可以直接添加到现有的训练循环中
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        advantage_normalize: bool = True
    ):
        super().__init__()
        self.reward_scale = reward_scale
        self.advantage_normalize = advantage_normalize

    def compute_trajectory_reward(
        self,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        effector_positions: torch.Tensor,
        target_object_center: torch.Tensor,
        target_placement_position: torch.Tensor
    ) -> torch.Tensor:
        """
        计算轨迹奖励

        Args:
            predicted_actions: 预测的动作 [B, T, A]
            target_actions: 目标动作 [B, T, A]
            effector_positions: 末端执行器位置 [B, T, 3]
            target_object_center: 目标物体中心 [B, 3]
            target_placement_position: 目标放置位置 [B, 3]

        Returns:
            rewards: 每个样本的奖励 [B]
        """
        batch_size = effector_positions.shape[0]
        seq_len = effector_positions.shape[1]

        # 计算到目标物体的距离
        # [B, T, 3] - [B, 1, 3] -> [B, T, 3]
        dist_to_object = torch.norm(
            effector_positions - target_object_center.unsqueeze(1),
            dim=-1
        )  # [B, T]

        # 计算到目标放置位置的距离
        dist_to_placement = torch.norm(
            effector_positions - target_placement_position.unsqueeze(1),
            dim=-1
        )  # [B, T]

        # 总距离
        total_distance = (dist_to_object + dist_to_placement).sum(dim=1)  # [B]

        # 奖励 = -距离（距离越小，奖励越高）
        rewards = -total_distance * self.reward_scale

        return rewards

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算相对优势

        Args:
            rewards: 奖励值 [B]

        Returns:
            advantages: 归一化的优势 [B]
        """
        if self.advantage_normalize:
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            advantages = (rewards - mean_reward) / std_reward
        else:
            advantages = rewards

        return advantages

    def forward(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算GRPO损失

        Args:
            log_probs: 动作的对数概率 [B]
            rewards: 轨迹奖励 [B]

        Returns:
            loss: GRPO损失
        """
        # 计算优势
        advantages = self.compute_advantages(rewards)

        # GRPO损失：最大化加权对数概率
        loss = -(log_probs * advantages).mean()

        return loss


class GRPORewardShaper:
    """
    GRPO奖励塑形器
    用于在训练过程中计算基于轨迹质量的奖励
    """

    def __init__(self, reward_scale: float = 0.01):
        self.reward_scale = reward_scale

    def extract_effector_positions(
        self,
        state: torch.Tensor,
        state_dim_config: Dict
    ) -> torch.Tensor:
        """
        从状态中提取末端执行器位置

        Args:
            state: 状态张量 [B, T, S]
            state_dim_config: 状态维度配置

        Returns:
            effector_positions: 末端执行器位置 [B, T, 3]
        """
        # 根据Tianqing数据集的state配置
        # state包含：joint positions (14维) + effector positions (6维，左右各3维)
        # 这里假设使用右手末端执行器位置（最后3维）

        effector_pos = state[..., -3:]  # [B, T, 3]
        return effector_pos

    def compute_batch_rewards(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算批次中每个样本的奖励

        Args:
            batch: 批次数据，包含state、action、task_info等

        Returns:
            rewards: 每个样本的奖励 [B]
        """
        state = batch['observation.state']  # [B, T, S]
        effector_positions = self.extract_effector_positions(state, {})

        # 从任务信息中提取目标位置
        # 这里需要根据实际数据格式调整
        if 'target_object_center' in batch:
            target_object = batch['target_object_center']  # [B, 3]
            target_placement = batch['target_placement_position']  # [B, 3]
        else:
            # 如果没有明确的目标位置，使用轨迹终点作为目标
            target_object = effector_positions[:, -1, :]  # [B, 3]
            target_placement = target_object

        # 计算距离
        dist_to_object = torch.norm(
            effector_positions - target_object.unsqueeze(1),
            dim=-1
        ).sum(dim=1)  # [B]

        dist_to_placement = torch.norm(
            effector_positions - target_placement.unsqueeze(1),
            dim=-1
        ).sum(dim=1)  # [B]

        # 奖励
        rewards = -(dist_to_object + dist_to_placement) * self.reward_scale

        return rewards


def add_grpo_to_training_loop(
    model,
    batch,
    bc_loss,
    grpo_weight: float = 0.1
):
    """
    在现有训练循环中添加GRPO损失

    使用示例：
    ```python
    # 在训练循环中
    bc_loss = model.compute_bc_loss(batch)

    # 添加GRPO
    total_loss = add_grpo_to_training_loop(
        model, batch, bc_loss, grpo_weight=0.1
    )

    total_loss.backward()
    optimizer.step()
    ```

    Args:
        model: 模型
        batch: 批次数据
        bc_loss: BC损失
        grpo_weight: GRPO损失权重

    Returns:
        total_loss: 总损失（BC + GRPO）
    """
    # 创建奖励塑形器
    reward_shaper = GRPORewardShaper(reward_scale=0.01)

    # 计算奖励
    rewards = reward_shaper.compute_batch_rewards(batch)

    # 计算动作的对数概率
    # 这里需要根据实际模型接口调整
    with torch.no_grad():
        log_probs = model.compute_action_log_prob(batch)

    # 计算GRPO损失
    grpo_loss_fn = SimpleGRPOLoss()
    grpo_loss = grpo_loss_fn(log_probs, rewards)

    # 总损失
    total_loss = bc_loss + grpo_weight * grpo_loss

    return total_loss


# 使用示例
"""
在DreamZero训练脚本中集成GRPO：

1. 导入模块：
from grpo_simple import SimpleGRPOLoss, GRPORewardShaper

2. 在训练循环中添加：

# 原始BC训练
bc_loss = model(batch)

# 添加GRPO奖励塑形
reward_shaper = GRPORewardShaper(reward_scale=0.01)
rewards = reward_shaper.compute_batch_rewards(batch)

# 计算GRPO损失
grpo_loss_fn = SimpleGRPOLoss()
log_probs = model.compute_action_log_prob(batch)
grpo_loss = grpo_loss_fn(log_probs, rewards)

# 总损失
total_loss = bc_loss + 0.1 * grpo_loss

total_loss.backward()
optimizer.step()

3. 监控指标：
logger.info(f"BC Loss: {bc_loss.item():.4f}, GRPO Loss: {grpo_loss.item():.4f}")
logger.info(f"Mean Reward: {rewards.mean().item():.4f}")
"""
