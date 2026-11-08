"""Turn config sections into live objects: device, seeding and RNG state, environments, policies.

Every place a run's network is built goes through :func:`make_policy` (a new run) or
:func:`load_policy` (a checkpoint), so its input size and channels always come from the run's
observation.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snakeppo.agents import REGISTRY, build_agent
from snakeppo.agents.networks import PolicyNetwork
from snakeppo.config import Config, EnvConfig, ObservationConfig, RewardConfig
from snakeppo.env import TorchSnakeEnv
from snakeppo.observation import ObservationSpec, describe_record
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
    observation: ObservationConfig,
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
        observation=ObservationSpec.from_config(observation),
    )


def make_play_env(
    env: EnvConfig,
    observation: ObservationConfig,
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
) -> TorchSnakeEnv:
    """An env for eval and watching: the full board, the run's observation, reward ``sparse``.

    The reward is never read; eval and watch only count apples.
    """
    return TorchSnakeEnv(
        num_envs,
        env.board_size,
        starvation_steps=env.starvation_steps,
        reward=SparseReward(),
        device=device,
        seed=seed,
        observation=ObservationSpec.from_config(observation),
    )


def observation_record(config: Config) -> dict[str, Any]:
    """What ``run.json`` and every checkpoint store about the run's observation."""
    return ObservationSpec.from_config(config.observation).record(config.env.board_size)


def make_policy(config: Config) -> PolicyNetwork:
    """The run's network, on CPU, sized from its observation (input side and channel count).

    With the default observation this is exactly ``build_agent(name, env.board_size, **kwargs)``.
    """
    spec = ObservationSpec.from_config(config.observation)
    name, size = config.agent.name, spec.size(config.env.board_size)
    agent = REGISTRY.get(name)
    if spec.view_size is not None and agent is not None and agent.requires_even_board and size % 2:
        raise ValueError(
            f"agent '{name}' needs an even input size, but observation.view: local feeds it "
            f"observation.view_size ({spec.view_size}) cells a side"
        )
    return build_agent(name, size, in_channels=len(spec.channels), **config.agent.kwargs)


def load_policy(checkpoint: Mapping[str, Any], config: Config, where: Path) -> PolicyNetwork:
    """The network ``checkpoint`` holds, on CPU, once its observation matches ``config``'s.

    ``where`` names the checkpoint in error messages. A schema-1 to 3 checkpoint records no
    observation: it trained on the default one, which is what its config describes too.
    """
    expected = observation_record(config)
    saved = checkpoint.get("observation")
    if saved is None:
        saved = observation_record(replace(config, observation=ObservationConfig()))
    if saved != expected:
        raise ValueError(
            f"{where} was trained on {describe_record(saved)}, but its config describes "
            f"{describe_record(expected)}; refusing to load it with the wrong observation"
        )
    model = make_policy(config)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as error:
        first_line = str(error).strip().splitlines()[0].rstrip(":")
        raise ValueError(
            f"{where} does not fit a '{config.agent.name}' network for "
            f"{describe_record(expected)}: {first_line}"
        ) from error
    return model
