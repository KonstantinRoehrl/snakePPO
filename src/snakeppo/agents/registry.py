"""Architecture registry: every policy network is built by name from here."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from snakeppo.agents.networks import CNNPolicy, PolicyNetwork, ResNetPolicy, SpatialCNNPolicy


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    builder: Callable[..., PolicyNetwork]
    requires_even_board: bool = False


_SHARED_ARGUMENTS = ("board_size", "in_channels", "num_actions")

REGISTRY: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"agent '{spec.name}' is already registered")
    REGISTRY[spec.name] = spec


register(AgentSpec("cnn", "5 conv layers, two stride-2, adaptive 4x4 pool, MLP head", CNNPolicy))
register(
    AgentSpec(
        "cnn-spatial",
        "6 conv layers, one stride-2 (half-resolution features), MLP head",
        SpatialCNNPolicy,
        requires_even_board=True,
    )
)
register(
    AgentSpec(
        "resnet",
        "3 residual conv blocks, one stride-2, MLP head",
        ResNetPolicy,
        requires_even_board=True,
    )
)


def available_agents() -> list[str]:
    return sorted(REGISTRY)


def build_agent(
    name: str,
    board_size: int,
    in_channels: int = 4,
    num_actions: int = 3,
    **kwargs: Any,
) -> PolicyNetwork:
    """Instantiate a registered network; raises ValueError for unknown names or odd boards."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"unknown agent '{name}'; available: {', '.join(available_agents())}")
    if spec.requires_even_board and board_size % 2:
        raise ValueError(f"agent '{name}' needs an even env.board_size, got {board_size}")
    accepted = [
        parameter
        for parameter in inspect.signature(spec.builder).parameters
        if parameter not in _SHARED_ARGUMENTS
    ]
    unexpected = [key for key in kwargs if key not in accepted]
    if unexpected:
        raise ValueError(
            f"agent '{name}' does not accept agent.kwargs "
            f"{', '.join(repr(key) for key in unexpected)}; "
            f"accepted: {', '.join(accepted) or 'none'}"
        )
    return spec.builder(
        board_size=board_size, in_channels=in_channels, num_actions=num_actions, **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
