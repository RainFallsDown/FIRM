from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION


@dataclass(frozen=True)
class ACTGRPOWeightConfig:
    beta: float = 1.0
    min_weight: float = 0.5
    max_weight: float = 2.0
    eps: float = 1e-6
    bc_reward_weight: float = 0.45
    smooth_reward_weight: float = 0.25
    accel_reward_weight: float = 0.20
    gripper_reward_weight: float = 0.10


def _as_float_tensor(value: Tensor) -> Tensor:
    if value.is_floating_point():
        return value
    return value.float()


def _zscore(values: Tensor, eps: float) -> Tensor:
    values = values.detach()
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or std < eps:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + eps)


def _masked_mean(values: Tensor, mask: Tensor, eps: float) -> Tensor:
    values = _as_float_tensor(values)
    mask = mask.to(device=values.device, dtype=values.dtype)
    numerator = (values * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(eps)
    return numerator / denominator


def _normalize_with_bounds(weights: Tensor, min_weight: float, max_weight: float, eps: float) -> Tensor:
    target_sum = float(weights.numel())
    if weights.numel() == 0:
        return weights

    bounded = weights.detach().clone().float().clamp(min=min_weight, max=max_weight)
    free = torch.ones_like(bounded, dtype=torch.bool)
    remaining_sum = target_sum

    for _ in range(int(weights.numel()) + 1):
        if not free.any():
            break
        free_sum = bounded[free].sum().clamp_min(eps)
        bounded[free] = bounded[free] * (remaining_sum / free_sum)
        low = (bounded < min_weight) & free
        high = (bounded > max_weight) & free
        fixed = low | high
        if not fixed.any():
            break
        bounded[low] = min_weight
        bounded[high] = max_weight
        remaining_sum = target_sum - float(bounded[~(free & ~fixed)].sum().item())
        free[fixed] = False

    bounded = bounded.clamp(min=min_weight, max=max_weight)
    if free.any():
        error = target_sum - float(bounded.sum().item())
        bounded[free] += error / free.sum().to(dtype=bounded.dtype)
    return bounded.clamp(min=min_weight, max=max_weight).to(device=weights.device, dtype=weights.dtype)


class ACTGRPOWeights:
    """Batch intrinsic reward weights for ACT.

    This ACT-GRPO stage treats each batch as a group and does not run online rollouts.
    Rewards and weights are detached; gradients only flow through the ACT loss.
    """

    def __init__(self, config: ACTGRPOWeightConfig | None = None) -> None:
        self.config = config or ACTGRPOWeightConfig()

    def compute_batch_weights(
        self,
        batch: dict[str, Tensor],
        per_sample_loss: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.config
        per_sample_loss = per_sample_loss.detach()
        batch_size = int(per_sample_loss.shape[0])

        if batch_size <= 1:
            weights = torch.ones_like(per_sample_loss)
            stats = {
                "grpo_reward_mean": 0.0,
                "grpo_reward_std": 0.0,
                "grpo_weight_min": 1.0,
                "grpo_weight_max": 1.0,
                "grpo_weight_mean": 1.0,
            }
            return weights.detach(), stats

        rewards, reward_stats = self.compute_batch_rewards(batch, per_sample_loss)
        advantage = _zscore(rewards, cfg.eps).clamp(min=-2.0, max=2.0)

        weights = torch.exp(cfg.beta * advantage)
        if not torch.isfinite(weights).all():
            weights = torch.ones_like(per_sample_loss)
        weights = _normalize_with_bounds(weights, cfg.min_weight, cfg.max_weight, cfg.eps)

        stats = {
            "grpo_reward_mean": float(rewards.mean().item()),
            "grpo_reward_std": float(rewards.std(unbiased=False).item()),
            "grpo_weight_min": float(weights.min().item()),
            "grpo_weight_max": float(weights.max().item()),
            "grpo_weight_mean": float(weights.mean().item()),
        }
        stats.update(reward_stats)
        return weights.detach(), stats

    def compute_batch_rewards(
        self,
        batch: dict[str, Tensor],
        per_sample_loss: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg = self.config
        actions = batch[ACTION].detach()
        action_is_pad = batch.get("action_is_pad")

        if action_is_pad is None:
            valid_steps = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        else:
            valid_steps = ~action_is_pad.to(device=actions.device).bool()

        bc_quality = _zscore(-per_sample_loss, cfg.eps)
        smoothness = self._action_smoothness(actions, valid_steps)
        acceleration = self._action_acceleration(actions, valid_steps)
        gripper_consistency = self._gripper_consistency(actions, valid_steps)

        reward = (
            cfg.bc_reward_weight * bc_quality
            + cfg.smooth_reward_weight * _zscore(smoothness, cfg.eps)
            + cfg.accel_reward_weight * _zscore(acceleration, cfg.eps)
            + cfg.gripper_reward_weight * _zscore(gripper_consistency, cfg.eps)
        )

        if not torch.isfinite(reward).all():
            reward = torch.zeros_like(per_sample_loss)

        stats = {
            "grpo_reward_bc_quality_mean": float(bc_quality.mean().item()),
            "grpo_reward_smoothness_mean": float(smoothness.mean().item()),
            "grpo_reward_acceleration_mean": float(acceleration.mean().item()),
            "grpo_reward_gripper_mean": float(gripper_consistency.mean().item()),
        }
        return reward.detach(), stats

    def _action_smoothness(self, actions: Tensor, valid_steps: Tensor) -> Tensor:
        cfg = self.config
        if actions.shape[1] < 2:
            return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        deltas = actions[:, 1:] - actions[:, :-1]
        valid_pairs = valid_steps[:, 1:] & valid_steps[:, :-1]
        per_step = -(deltas.square().mean(dim=-1))
        return _masked_mean(per_step, valid_pairs, cfg.eps)

    def _action_acceleration(self, actions: Tensor, valid_steps: Tensor) -> Tensor:
        cfg = self.config
        if actions.shape[1] < 3:
            return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
        accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
        valid_triples = valid_steps[:, 2:] & valid_steps[:, 1:-1] & valid_steps[:, :-2]
        per_step = -(accel.square().mean(dim=-1))
        return _masked_mean(per_step, valid_triples, cfg.eps)

    def _gripper_consistency(self, actions: Tensor, valid_steps: Tensor) -> Tensor:
        cfg = self.config
        if actions.shape[-1] < 2:
            return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

        gripper = actions[..., -2:]
        valid = valid_steps.to(device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        denominator = valid.sum(dim=1).clamp_min(cfg.eps)
        gripper_mean = (gripper * valid).sum(dim=1) / denominator
        centered = gripper - gripper_mean.unsqueeze(1)
        per_step = -(centered.square().mean(dim=-1))
        return _masked_mean(per_step, valid_steps, cfg.eps)
