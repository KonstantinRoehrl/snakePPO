import pytest
import torch
from torch import nn

from snakeppo.config import EnvConfig
from snakeppo.env import STRAIGHT, TURN_RIGHT
from snakeppo.evaluation import evaluate

CPU = torch.device("cpu")


class _Scripted(nn.Module):
    """A policy that ignores the board: one fixed action, or uniformly random ones."""

    def __init__(self, action=None, seed=0):
        super().__init__()
        self.action = action
        self.generator = torch.Generator().manual_seed(seed)

    def act(self, obs, greedy=False):
        n = obs.shape[0]
        if self.action is None:
            actions = torch.randint(0, 3, (n,), generator=self.generator)
        else:
            actions = torch.full((n,), self.action, dtype=torch.long)
        return actions, torch.zeros(n), torch.zeros(n)


def test_a_snake_that_only_goes_straight_always_dies_on_the_wall():
    stats = evaluate(_Scripted(STRAIGHT), EnvConfig(board_size=8), episodes=5, device=CPU, seed=1)
    assert (stats.end_wall, stats.end_body, stats.end_starved) == (1.0, 0.0, 0.0)
    assert stats.score_mean == 0.0 and stats.steps_per_apple is None
    metrics = stats.as_metrics()
    assert metrics["eval/end_wall"] == 1.0 and "eval/steps_per_apple" not in metrics


def test_a_snake_circling_in_place_starves():
    env = EnvConfig(board_size=16, starvation_steps=8)
    stats = evaluate(_Scripted(TURN_RIGHT), env, episodes=4, device=CPU, seed=0)
    assert stats.end_starved == 1.0
    assert stats.length_mean == 8


def test_end_causes_sum_to_one_and_steps_per_apple_is_steps_over_apples():
    env = EnvConfig(board_size=6, starvation_steps=64)
    stats = evaluate(_Scripted(seed=3), env, episodes=32, device=CPU, seed=2)
    assert stats.end_wall + stats.end_body + stats.end_starved == pytest.approx(1.0)
    assert stats.score_mean > 0
    assert stats.steps_per_apple == pytest.approx(stats.length_mean / stats.score_mean)
    assert stats.as_metrics()["eval/steps_per_apple"] == stats.steps_per_apple
