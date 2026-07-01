import math
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The pynvml package is deprecated.*",
)
import torch

from groot.vla.grpo_simple import GRPOWeightBuffer, compute_action_grpo_reward


def masked_mean(value, mask=None):
    if mask is None:
        return value.mean()
    mask = mask.to(dtype=value.dtype, device=value.device)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def reference_lingbot_action_reward(action, action_mask=None, reward_scale=1.0):
    action = action.detach().float()
    if action_mask is not None:
        action_mask = action_mask.detach().bool()

    reward = action.new_tensor(0.0)

    if action.shape[2] > 1:
        diff = action[:, :, 1:] - action[:, :, :-1]
        diff_mask = None
        if action_mask is not None:
            diff_mask = action_mask[:, :, 1:] & action_mask[:, :, :-1]
        reward = reward - 0.3 * masked_mean(diff.square(), diff_mask)

    if action.shape[2] > 2:
        jerk = action[:, :, 2:] - 2.0 * action[:, :, 1:-1] + action[:, :, :-2]
        jerk_mask = None
        if action_mask is not None:
            jerk_mask = (
                action_mask[:, :, 2:]
                & action_mask[:, :, 1:-1]
                & action_mask[:, :, :-2]
            )
        reward = reward - 0.2 * masked_mean(jerk.square(), jerk_mask)

    magnitude = torch.relu(action.abs() - 1.0)
    reward = reward - 0.1 * masked_mean(magnitude.square(), action_mask)

    if action.shape[2] > 1:
        midpoint = max(action.shape[2] // 2, 1)
        first = action[:, :, :midpoint].flatten(1)
        second = action[:, :, midpoint:].flatten(1)
        min_width = min(first.shape[1], second.shape[1])
        if min_width > 0:
            consistency = torch.nn.functional.cosine_similarity(
                first[:, :min_width],
                second[:, :min_width],
                dim=1,
                eps=1e-6,
            ).mean()
            reward = reward + 0.2 * consistency

    return reward * reward_scale


def test_action_reward_matches_lingbot_formula_with_mask():
    action = torch.tensor(
        [
            [[[0.0, 0.5, 1.2, 1.4], [0.0, -0.5, -1.2, -1.4]]],
            [[[0.1, 0.2, 0.4, 0.9], [0.0, 0.3, 0.6, 0.9]]],
        ]
    )
    mask = torch.tensor(
        [
            [[[True, True, True, False], [True, True, False, False]]],
            [[[True, True, True, True], [True, False, True, True]]],
        ]
    )

    expected = reference_lingbot_action_reward(action, mask, reward_scale=1.7)
    actual = compute_action_grpo_reward(action, mask, reward_scale=1.7)

    assert torch.allclose(actual, expected)


def test_weight_buffer_matches_lingbot_warmup_and_history_weighting():
    buffer = GRPOWeightBuffer(
        buffer_size=3,
        warmup_steps=2,
        alpha=0.5,
        clamp_min=0.5,
        clamp_max=2.0,
    )

    first = buffer.update(torch.tensor(1.0))
    second = buffer.update(torch.tensor(2.0))
    third = buffer.update(torch.tensor(4.0))

    assert first["advantage"].item() == 0.0
    assert first["weight"].item() == 1.0
    assert second["advantage"].item() == 0.0
    assert second["weight"].item() == 1.0

    history = torch.tensor([1.0, 2.0])
    expected_advantage = (torch.tensor(4.0) - history.mean()) / (
        history.std(unbiased=False) + 1e-8
    )
    expected_weight = torch.exp(0.5 * expected_advantage).clamp(0.5, 2.0)

    assert math.isclose(third["advantage"].item(), expected_advantage.item())
    assert math.isclose(third["weight"].item(), expected_weight.item())


if __name__ == "__main__":
    test_action_reward_matches_lingbot_formula_with_mask()
    test_weight_buffer_matches_lingbot_warmup_and_history_weighting()
