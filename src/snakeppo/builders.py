"""Turn config sections into live objects: device, seeding and RNG state, environments."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from snakeppo.config import EnvConfig, RewardConfig
from snakeppo.env import TorchSnakeEnv
from snakeppo.rewards import SparseReward, build_reward

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


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Python, numpy, torch CPU and (on CUDA) every CUDA generator's state.

    numpy's key array is stored as a tensor so ``torch.load(weights_only=True)`` accepts it.
    """
    np_state = np.random.get_state(legacy=False)
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": np_state["bit_generator"],
            "key": torch.from_numpy(np_state["state"]["key"].astype(np.int64)),
            "pos": int(np_state["state"]["pos"]),
            "has_gauss": int(np_state["has_gauss"]),
            "gauss": float(np_state["gauss"]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Inverse of :func:`capture_rng_state`."""
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    numpy = state["numpy"]
    np.random.set_state(
        {
            "bit_generator": numpy["bit_generator"],
            "state": {"key": numpy["key"].numpy().astype(np.uint32), "pos": numpy["pos"]},
            "has_gauss": numpy["has_gauss"],
            "gauss": numpy["gauss"],
        }
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_env(
    env: EnvConfig,
    reward: RewardConfig,
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
    discount: float,
) -> TorchSnakeEnv:
    """The training env; ``discount`` is ``ppo.gamma``, which the potential strategy needs."""
    return TorchSnakeEnv(
        num_envs,
        env.board_size,
        starvation_steps=env.starvation_steps,
        reward=build_reward(reward, discount),
        device=device,
        seed=seed,
    )


def make_play_env(
    env: EnvConfig, *, num_envs: int, device: torch.device, seed: int
) -> TorchSnakeEnv:
    """An env for eval and watching: the full board, and a reward nobody reads (``sparse``)."""
    return TorchSnakeEnv(
        num_envs,
        env.board_size,
        starvation_steps=env.starvation_steps,
        reward=SparseReward(),
        device=device,
        seed=seed,
    )
