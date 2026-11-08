"""Reward strategies: every per-step reward rule is built by name from here.

The env builds one :class:`StepContext` per ``step`` and calls the strategy with it. Each
strategy is a frozen dataclass; :func:`build_reward` makes one from a ``RewardConfig``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from snakeppo.config import RewardConfig


@dataclass(frozen=True)
class StepContext:
    """One batched env step, as reward strategies see it.

    ``old_head`` and ``old_food`` are from before the move, ``new_head`` is the cell the snake
    tried to enter (possibly a wall), and ``head``, ``food`` and ``body`` are after the move and
    the food respawn, before any auto-reset. ``playable`` is the ``(S, S)`` arena interior.
    """

    old_head: torch.Tensor  # (N, 2) long
    new_head: torch.Tensor  # (N, 2) long
    head: torch.Tensor  # (N, 2) long
    old_food: torch.Tensor  # (N, 2) long
    food: torch.Tensor  # (N, 2) long
    ate: torch.Tensor  # (N,) bool
    terminated: torch.Tensor  # (N,) bool: hit a wall or the body
    starved: torch.Tensor  # (N,) bool: starvation truncation on this step
    steps_since_food: torch.Tensor  # (N,) long, after this step
    starvation_steps: int
    body: torch.Tensor  # (N, S, S) bool
    playable: torch.Tensor  # (S, S) bool


class RewardStrategy(Protocol):
    def __call__(self, step: StepContext) -> torch.Tensor:
        """The (N,) float32 reward of this step."""
        ...


def manhattan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-env Manhattan distance between two (N, 2) cell tensors, as float32."""
    return (a - b).abs().sum(dim=1).float()


def _dilate(mask: torch.Tensor) -> torch.Tensor:
    grown = mask.clone()
    grown[:, 1:, :] |= mask[:, :-1, :]
    grown[:, :-1, :] |= mask[:, 1:, :]
    grown[:, :, 1:] |= mask[:, :, :-1]
    grown[:, :, :-1] |= mask[:, :, 1:]
    return grown


def reachable_fraction(
    body: torch.Tensor, head: torch.Tensor, playable: torch.Tensor
) -> torch.Tensor:
    """Share of free playable cells the head can reach through free cells; 1 with none free.

    An exact batched flood fill: dilate from the head until no env grows.
    """
    free = playable.unsqueeze(0) & ~body
    env_ids = torch.arange(body.shape[0], device=body.device)
    seed = torch.zeros_like(body)
    seed[env_ids, head[:, 0], head[:, 1]] = True
    reached = seed
    while True:
        grown = seed | (_dilate(reached) & free)
        if torch.equal(grown, reached):
            break
        reached = grown
    free_count = free.sum(dim=(1, 2))
    reached_count = (reached & free).sum(dim=(1, 2))
    fraction = reached_count.float() / free_count.clamp(min=1).float()
    return torch.where(free_count > 0, fraction, torch.ones_like(fraction))


def _terminal_rewards(
    step: StepContext, reward: torch.Tensor, apple: float, death: float
) -> torch.Tensor:
    reward = torch.where(step.ate, torch.full_like(reward, apple), reward)
    return torch.where(step.terminated, torch.full_like(reward, death), reward)


def _shaped(step: StepContext) -> torch.Tensor:
    """Steps that neither eat nor terminate: the ones shaping terms apply to."""
    return ~(step.ate | step.terminated)


@dataclass(frozen=True)
class ClassicReward:
    """The baseline: shaping on the change in squared distance, replaced on eating and dying."""

    apple: float = 1.0
    death: float = -3.0
    approach: float = 0.02
    retreat: float = 0.03

    def __call__(self, step: StepContext) -> torch.Tensor:
        old_dist = (step.old_head - step.old_food).float().pow(2).sum(dim=1)
        new_dist = (step.new_head - step.old_food).float().pow(2).sum(dim=1)
        delta = old_dist - new_dist
        reward = delta.clamp(min=0) * self.approach + delta.clamp(max=0) * self.retreat
        return _terminal_rewards(step, reward, self.apple, self.death)


@dataclass(frozen=True)
class SparseReward:
    """Apple and death rewards only."""

    apple: float = 1.0
    death: float = -3.0

    def __call__(self, step: StepContext) -> torch.Tensor:
        reward = torch.zeros(step.ate.shape, device=step.ate.device)
        return _terminal_rewards(step, reward, self.apple, self.death)


@dataclass(frozen=True)
class DistanceReward:
    """Sparse plus ``scale`` per cell moved toward the food, minus ``scale`` per cell away."""

    apple: float
    death: float
    scale: float

    def __call__(self, step: StepContext) -> torch.Tensor:
        closer = manhattan(step.old_head, step.old_food) - manhattan(step.new_head, step.old_food)
        return _terminal_rewards(step, closer * self.scale, self.apple, self.death)


@dataclass(frozen=True)
class PotentialReward:
    """Sparse plus ``F = discount * phi(s') - phi(s)`` on every step, ``phi = -scale * d``.

    ``phi`` of a terminal state is 0, so the shaping telescopes to ``-phi(s0)`` over an episode
    and leaves the optimal policy unchanged.
    """

    apple: float
    death: float
    scale: float
    discount: float

    def __call__(self, step: StepContext) -> torch.Tensor:
        potential = -self.scale * manhattan(step.old_head, step.old_food)
        following = -self.scale * manhattan(step.head, step.food)
        following = torch.where(step.terminated, torch.zeros_like(following), following)
        sparse = SparseReward(self.apple, self.death)(step)
        return sparse + (self.discount * following - potential)


@dataclass(frozen=True)
class HungerReward:
    """Classic plus a per-step hunger ramp and a one-off penalty on a starvation truncation."""

    classic: ClassicReward
    timeout_penalty: float
    hunger_penalty: float

    def __call__(self, step: StepContext) -> torch.Tensor:
        reward = self.classic(step)
        hunger = step.steps_since_food.float() / step.starvation_steps
        extra = -self.hunger_penalty * hunger - self.timeout_penalty * step.starved.float()
        return torch.where(_shaped(step), reward + extra, reward)


@dataclass(frozen=True)
class FreeSpaceReward:
    """Classic plus ``-penalty`` times the share of free cells the head can no longer reach."""

    classic: ClassicReward
    penalty: float

    def __call__(self, step: StepContext) -> torch.Tensor:
        reward = self.classic(step)
        trapped = 1 - reachable_fraction(step.body, step.head, step.playable)
        return torch.where(_shaped(step), reward - self.penalty * trapped, reward)


@dataclass(frozen=True)
class RewardSpec:
    name: str
    description: str
    build: Callable[[RewardConfig, float], RewardStrategy]


REWARDS: dict[str, RewardSpec] = {}


def register(spec: RewardSpec) -> None:
    if spec.name in REWARDS:
        raise ValueError(f"reward strategy '{spec.name}' is already registered")
    REWARDS[spec.name] = spec


def _classic(config: RewardConfig) -> ClassicReward:
    return ClassicReward(config.apple, config.death, config.approach, config.retreat)


register(
    RewardSpec(
        "classic",
        "apple and death rewards plus shaping on the change in squared distance (the baseline)",
        lambda config, discount: _classic(config),
    )
)
register(
    RewardSpec(
        "sparse",
        "apple and death rewards only",
        lambda config, discount: SparseReward(config.apple, config.death),
    )
)
register(
    RewardSpec(
        "distance",
        "sparse plus distance_scale per cell moved toward the food, minus it per cell away",
        lambda config, discount: DistanceReward(config.apple, config.death, config.distance_scale),
    )
)
register(
    RewardSpec(
        "potential",
        "sparse plus potential-based shaping on the distance to the food, discounted by ppo.gamma",
        lambda config, discount: PotentialReward(
            config.apple, config.death, config.potential_scale, discount
        ),
    )
)
register(
    RewardSpec(
        "hunger",
        "classic plus a per-step hunger ramp and a timeout penalty when the snake starves",
        lambda config, discount: HungerReward(
            _classic(config), config.timeout_penalty, config.hunger_penalty
        ),
    )
)
register(
    RewardSpec(
        "free_space",
        "classic plus a penalty for the share of free cells the head can no longer reach",
        lambda config, discount: FreeSpaceReward(_classic(config), config.free_space_penalty),
    )
)


def available_rewards() -> list[str]:
    return sorted(REWARDS)


def build_reward(config: RewardConfig, discount: float) -> RewardStrategy:
    """The strategy ``config.strategy`` names; raises ValueError for an unknown name."""
    spec = REWARDS.get(config.strategy)
    if spec is None:
        raise ValueError(
            f"unknown reward strategy '{config.strategy}'; available: "
            f"{', '.join(available_rewards())}"
        )
    return spec.build(config, discount)
