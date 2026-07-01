import torch

from wan_va.grpo import GRPOWeightBuffer, compute_action_grpo_reward


def test_action_reward_penalizes_jerky_actions():
    smooth = torch.zeros(1, 2, 6, 1, 1)
    jerky = smooth.clone()
    jerky[:, :, 1::2] = 1.0
    mask = torch.ones_like(smooth, dtype=torch.bool)

    smooth_reward = compute_action_grpo_reward(smooth, mask)
    jerky_reward = compute_action_grpo_reward(jerky, mask)

    assert smooth_reward.item() > jerky_reward.item()


def test_weight_buffer_warms_up_then_scales_from_advantage():
    buffer = GRPOWeightBuffer(
        buffer_size=4,
        warmup_steps=2,
        alpha=1.0,
        clamp_min=0.5,
        clamp_max=2.0,
    )

    first = buffer.update(torch.tensor(0.0))
    second = buffer.update(torch.tensor(0.0))
    third = buffer.update(torch.tensor(1.0))

    assert first["weight"].item() == 1.0
    assert second["weight"].item() == 1.0
    assert third["advantage"].item() > 0.0
    assert third["weight"].item() > 1.0


if __name__ == "__main__":
    test_action_reward_penalizes_jerky_actions()
    test_weight_buffer_warms_up_then_scales_from_advantage()
    print("GRPO helper tests passed")
