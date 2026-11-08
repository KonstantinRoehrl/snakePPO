"""Score a policy over a fixed number of complete episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snakeppo.agents.networks import PolicyNetwork
from snakeppo.builders import make_play_env
from snakeppo.config import EnvConfig
from snakeppo.env import END_BODY, END_STARVED, END_WALL


@dataclass(frozen=True)
class EvalStats:
    episodes: int
    score_mean: float
    score_max: float
    score_min: float
    length_mean: float
    end_wall: float  # fraction of episodes that hit a wall
    end_body: float  # fraction that ran into their own body
    end_starved: float  # fraction that starved
    steps_per_apple: float | None  # total steps / total apples; None when no apple was eaten

    def as_metrics(self, prefix: str = "eval/") -> dict[str, float]:
        metrics = {
            f"{prefix}score_mean": self.score_mean,
            f"{prefix}score_max": self.score_max,
            f"{prefix}score_min": self.score_min,
            f"{prefix}length_mean": self.length_mean,
            f"{prefix}end_wall": self.end_wall,
            f"{prefix}end_body": self.end_body,
            f"{prefix}end_starved": self.end_starved,
        }
        if self.steps_per_apple is not None:
            metrics[f"{prefix}steps_per_apple"] = self.steps_per_apple
        return metrics


@torch.no_grad()
def evaluate(
    model: PolicyNetwork,
    env: EnvConfig,
    *,
    episodes: int,
    device: torch.device,
    seed: int,
    greedy: bool = True,
) -> EvalStats:
    """Play ``episodes`` games in parallel and count each game's first finished episode.

    Games run on the full board with no goal and ignore the reward strategy, so scores
    compare across reward strategies and curricula. Starvation truncation bounds every
    episode, so this always terminates.
    """
    was_training = model.training
    model.eval()
    game = make_play_env(env, num_envs=episodes, device=device, seed=seed)
    obs = game.reset()
    finished = torch.zeros(episodes, dtype=torch.bool, device=game.device)
    scores = torch.zeros(episodes, device=game.device)
    lengths = torch.zeros(episodes, device=game.device)
    reasons = torch.zeros(episodes, dtype=torch.long, device=game.device)
    while not finished.all():
        actions, _, _ = model.act(obs, greedy=greedy)
        obs, _, _, _, info = game.step(actions)
        first = info["done"] & ~finished
        scores = torch.where(first, info["episode_score"], scores)
        lengths = torch.where(first, info["episode_length"].float(), lengths)
        reasons = torch.where(first, info["end_reason"], reasons)
        finished |= first
    model.train(was_training)
    apples = scores.sum().item()
    return EvalStats(
        episodes=episodes,
        score_mean=scores.mean().item(),
        score_max=scores.max().item(),
        score_min=scores.min().item(),
        length_mean=lengths.mean().item(),
        end_wall=(reasons == END_WALL).float().mean().item(),
        end_body=(reasons == END_BODY).float().mean().item(),
        end_starved=(reasons == END_STARVED).float().mean().item(),
        steps_per_apple=lengths.sum().item() / apples if apples else None,
    )
