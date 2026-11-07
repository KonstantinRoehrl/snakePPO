"""Pure PPO math: GAE, the clipped surrogate loss and coefficient schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over a (T, N) rollout.

    ``next_values[t]`` is V(s_{t+1}); for a truncated step it must be the value of the
    pre-reset final observation. ``terminated`` stops bootstrapping (death has no future);
    ``done`` (terminated or truncated) stops the advantage trace at the episode boundary.
    Returns ``(advantages, returns)``, both (T, N).
    """
    not_terminated = 1.0 - terminated.float()
    not_done = 1.0 - done.float()
    deltas = rewards + gamma * next_values * not_terminated - values
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        running = deltas[t] + gamma * gae_lambda * not_done[t] * running
        advantages[t] = running
    return advantages, advantages + values


@dataclass
class LossOutput:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    clip_value_loss: bool,
) -> LossOutput:
    """Clipped-surrogate PPO loss; every input is a flat minibatch tensor of shape (B,)."""
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()
    surrogate = torch.min(
        ratio * advantages, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    )
    policy = -surrogate.mean()
    value_error = (new_values - returns).pow(2)
    if clip_value_loss:
        clipped = old_values + (new_values - old_values).clamp(-clip_eps, clip_eps)
        value_error = torch.max(value_error, (clipped - returns).pow(2))
    value = value_error.mean()
    entropy_mean = entropy.mean()
    total = policy + value_coef * value - entropy_coef * entropy_mean
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return LossOutput(total, policy, value, entropy_mean, approx_kl, clip_fraction)


def lr_at(schedule: str, base_lr: float, progress: float) -> float:
    """Learning rate at ``progress`` in [0, 1] of total_env_steps."""
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return base_lr * (1.0 - progress)
    raise ValueError(f"unknown lr schedule '{schedule}'")


def entropy_coef_at(schedule: str, start: float, final: float, progress: float) -> float:
    """Entropy coefficient at ``progress`` in [0, 1] of total_env_steps."""
    if schedule == "constant":
        return start
    if schedule == "linear":
        return start + (final - start) * progress
    if schedule == "exponential":
        return start * math.pow(final / start, progress)
    raise ValueError(f"unknown entropy schedule '{schedule}'")


def explained_variance(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(target - predicted) / Var(target); NaN-free (0.0 when the target is constant)."""
    target_var = target.var()
    if target_var.item() == 0.0:
        return 0.0
    return float(1.0 - (target - predicted).var() / target_var)
