import math

import pytest
import torch

from snakeppo.ppo.functional import (
    compute_gae,
    entropy_coef_at,
    explained_variance,
    lr_at,
    ppo_loss,
)


def _rollout(terminated, truncated):
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    next_values = torch.tensor([[0.5], [10.0], [0.5]])
    terminated = torch.tensor(terminated).view(3, 1)
    done = terminated | torch.tensor(truncated).view(3, 1)
    return rewards, values, next_values, terminated, done


def test_gae_without_episode_end_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    args = _rollout([False, False, False], [False, False, False])
    advantages, returns = compute_gae(*args, gamma, lam)
    d2 = 3.0 + gamma * 0.5 - 0.5
    d1 = 2.0 + gamma * 10.0 - 0.5
    d0 = 1.0 + gamma * 0.5 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    assert torch.allclose(advantages.squeeze(1), torch.tensor([a0, a1, a2]))
    assert torch.allclose(returns, advantages + args[1])


def test_termination_drops_bootstrap_and_cuts_trace():
    gamma, lam = 0.9, 0.8
    advantages, _ = compute_gae(*_rollout([False, True, False], [False, False, False]), gamma, lam)
    # step 1 ends the episode: no bootstrap and step 2's advantage does not leak back
    assert advantages[1, 0].item() == pytest.approx(2.0 - 0.5)
    d0 = 1.0 + gamma * 0.5 - 0.5
    assert advantages[0, 0].item() == pytest.approx(d0 + gamma * lam * (2.0 - 0.5))


def test_truncation_bootstraps_from_final_value_but_cuts_trace():
    gamma, lam = 0.9, 0.8
    advantages, _ = compute_gae(*_rollout([False, False, False], [False, True, False]), gamma, lam)
    a1 = 2.0 + gamma * 10.0 - 0.5
    assert advantages[1, 0].item() == pytest.approx(a1)
    d0 = 1.0 + gamma * 0.5 - 0.5
    assert advantages[0, 0].item() == pytest.approx(d0 + gamma * lam * a1)


def _loss(new_values, clip_value_loss):
    ones = torch.ones(4)
    return ppo_loss(
        new_log_probs=torch.log(torch.tensor([0.5, 0.5, 0.9, 0.1])),
        old_log_probs=torch.log(torch.full((4,), 0.5)),
        advantages=torch.tensor([1.0, -1.0, 1.0, -1.0]),
        new_values=new_values,
        old_values=ones,
        returns=ones * 2,
        entropy=ones,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        clip_value_loss=clip_value_loss,
    )


def test_ppo_loss_clips_ratio_and_reports_clip_fraction():
    out = _loss(torch.full((4,), 2.0), clip_value_loss=False)
    # ratios: 1, 1, 1.8, 0.2 -> clipped surrogate: 1, -1, 1.2, -0.8 (min with unclipped)
    assert out.policy.item() == pytest.approx(-(1.0 - 1.0 + 1.2 - 0.8) / 4)
    assert out.clip_fraction.item() == pytest.approx(0.5)
    assert out.value.item() == pytest.approx(0.0)
    assert out.total.item() == pytest.approx(out.policy.item() - 0.01)
    assert out.approx_kl.item() > 0


def test_value_clipping_uses_the_worse_of_clipped_and_unclipped_error():
    new_values = torch.full((4,), 3.0)
    assert _loss(new_values, clip_value_loss=False).value.item() == pytest.approx(1.0)
    # clipped prediction: 1 + 0.2 = 1.2 -> error (1.2 - 2)^2 = 0.64; max(1.0, 0.64) = 1.0
    assert _loss(new_values, clip_value_loss=True).value.item() == pytest.approx(1.0)
    near = torch.full((4,), 2.5)
    # unclipped 0.25, clipped (1.2 - 2)^2 = 0.64 -> 0.64
    assert _loss(near, clip_value_loss=True).value.item() == pytest.approx(0.64)


def test_schedules():
    assert lr_at("constant", 1e-3, 0.7) == 1e-3
    assert lr_at("linear", 1e-3, 0.25) == pytest.approx(7.5e-4)
    assert entropy_coef_at("constant", 0.01, 0.001, 0.5) == 0.01
    assert entropy_coef_at("linear", 0.01, 0.002, 0.5) == pytest.approx(0.006)
    assert entropy_coef_at("exponential", 0.01, 0.001, 0.5) == pytest.approx(math.sqrt(1e-5))
    with pytest.raises(ValueError):
        lr_at("cosine", 1e-3, 0.5)


def test_explained_variance():
    target = torch.tensor([1.0, 2.0, 3.0])
    assert explained_variance(target, target) == pytest.approx(1.0)
    assert explained_variance(torch.zeros(3), torch.ones(3)) == 0.0
