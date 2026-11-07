"""Turn config sections into live objects: device, seeding, environments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from snakeppo.config import EnvConfig, RewardConfig
from snakeppo.env import TorchSnakeEnv

EVAL_SEED_OFFSET = 10_000


def resolve_device(name: str) -> torch.device:
    """``auto`` picks cuda, then mps, then cpu; any other value is passed to torch as-is."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch; ``deterministic`` also forces deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_env(
    env: EnvConfig,
    reward: RewardConfig,
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
) -> TorchSnakeEnv:
    return TorchSnakeEnv(
        num_envs,
        env.board_size,
        starvation_steps=env.starvation_steps,
        reward_apple=reward.apple,
        reward_death=reward.death,
        reward_approach=reward.approach,
        reward_retreat=reward.retreat,
        device=device,
        seed=seed,
    )
