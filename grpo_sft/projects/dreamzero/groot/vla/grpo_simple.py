"""LingBot-style GRPO utilities for DreamZero.

This module intentionally mirrors LingBot-VA's action-only GRPO logic so the
two projects can be compared under the same reward and weight-buffer rules.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(value, mask=None):
    if mask is None:
        return value.mean()
    mask = mask.to(dtype=value.dtype, device=value.device)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _as_lingbot_action_shape(action, action_mask=None):
    """Convert DreamZero [B, T, D] action tensors to LingBot [B, C, T, ...]."""
    if action.dim() == 3:
        action = action.unsqueeze(1)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(1)
    return action, action_mask


def compute_action_grpo_reward(action, action_mask=None, reward_scale=1.0):
    """Intrinsic action reward, kept in parity with LingBot-VA."""
    action, action_mask = _as_lingbot_action_shape(action, action_mask)
    action = action.detach().float()
    if action_mask is not None:
        action_mask = action_mask.detach().bool()

    reward = action.new_tensor(0.0)

    if action.shape[2] > 1:
        diff = action[:, :, 1:] - action[:, :, :-1]
        diff_mask = None
        if action_mask is not None:
            diff_mask = action_mask[:, :, 1:] & action_mask[:, :, :-1]
        reward = reward - 0.3 * _masked_mean(diff.square(), diff_mask)

    if action.shape[2] > 2:
        jerk = action[:, :, 2:] - 2.0 * action[:, :, 1:-1] + action[:, :, :-2]
        jerk_mask = None
        if action_mask is not None:
            jerk_mask = (
                action_mask[:, :, 2:]
                & action_mask[:, :, 1:-1]
                & action_mask[:, :, :-2]
            )
        reward = reward - 0.2 * _masked_mean(jerk.square(), jerk_mask)

    magnitude = torch.relu(action.abs() - 1.0)
    reward = reward - 0.1 * _masked_mean(magnitude.square(), action_mask)

    if action.shape[2] > 1:
        midpoint = max(action.shape[2] // 2, 1)
        first = action[:, :, :midpoint].flatten(1)
        second = action[:, :, midpoint:].flatten(1)
        min_width = min(first.shape[1], second.shape[1])
        if min_width > 0:
            consistency = F.cosine_similarity(
                first[:, :min_width],
                second[:, :min_width],
                dim=1,
                eps=1e-6,
            ).mean()
            reward = reward + 0.2 * consistency

    return reward * reward_scale


class GRPOWeightBuffer:
    def __init__(
        self,
        buffer_size=32,
        warmup_steps=4,
        alpha=1.0,
        clamp_min=0.5,
        clamp_max=2.0,
    ):
        self.buffer_size = int(buffer_size)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self._values = []

    def update(self, reward):
        reward = reward.detach().float().mean()
        device = reward.device

        if len(self._values) < self.warmup_steps:
            advantage = torch.zeros((), device=device, dtype=reward.dtype)
            weight = torch.ones((), device=device, dtype=reward.dtype)
        else:
            history = torch.tensor(self._values, device=device, dtype=reward.dtype)
            advantage = (reward - history.mean()) / (history.std(unbiased=False) + 1e-8)
            weight = torch.exp(self.alpha * advantage).clamp(
                self.clamp_min,
                self.clamp_max,
            )

        self._values.append(float(reward.detach().cpu()))
        if len(self._values) > self.buffer_size:
            self._values.pop(0)

        return {
            "reward": reward,
            "advantage": advantage.detach(),
            "weight": weight.detach(),
        }


class GRPORewardShaper:
    """Compatibility wrapper around LingBot-style action reward."""

    def __init__(self, reward_scale: float = 1.0):
        self.reward_scale = reward_scale

    def compute_batch_rewards(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        action = batch.get("action", None)
        if action is None:
            device = None
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
            return torch.zeros((), device=device)

        action_mask = batch.get("action_mask", batch.get("actions_mask", None))
        return compute_action_grpo_reward(
            action,
            action_mask,
            reward_scale=self.reward_scale,
        )


class SimpleGRPOLoss(nn.Module):
    """Backward-compatible wrapper; DreamZero training now uses GRPOWeightBuffer."""

    def __init__(self, reward_scale: float = 1.0, advantage_normalize: bool = True, **_):
        super().__init__()
        self.reward_scale = reward_scale
        self.advantage_normalize = advantage_normalize

    def forward(self, log_probs, rewards):
        rewards = rewards.detach().float()
        if self.advantage_normalize and rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        return -(log_probs * rewards.to(device=log_probs.device, dtype=log_probs.dtype)).mean()
