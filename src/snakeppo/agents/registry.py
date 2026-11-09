"""Architecture registry: every policy network is built by name from here."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from snakeppo.agents.networks import (
    AttentionPolicy,
    CNNPolicy,
    MLPPolicy,
    PolicyNetwork,
    ResNetPolicy,
    ScalableCNNPolicy,
    SpatialCNNPolicy,
)
from snakeppo.agents.recurrent import RecurrentCNNPolicy


@dataclass(frozen=True)
class AgentSpec:
    """A registered architecture.

    ``size_agnostic``: a model built for one board size accepts observations of any other size
    (at least 5) and returns valid logits and values. It says nothing about how well it plays there.
    """

    name: str
    description: str
    builder: Callable[..., PolicyNetwork]
    requires_even_board: bool = False
    size_agnostic: bool = False


_SHARED_ARGUMENTS = ("board_size", "in_channels", "num_actions")

REGISTRY: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"agent '{spec.name}' is already registered")
    REGISTRY[spec.name] = spec


def _cnn_of_width(width: int) -> Callable[..., PolicyNetwork]:
    """A ``ScalableCNNPolicy`` builder with ``width`` fixed, so it exposes no agent.kwargs."""

    def build(board_size: int, in_channels: int = 4, num_actions: int = 3) -> PolicyNetwork:
        return ScalableCNNPolicy(board_size, in_channels, num_actions, width=width)

    return build


register(
    AgentSpec(
        "cnn",
        "5 conv layers, two stride-2, adaptive 4x4 pool, MLP head",
        CNNPolicy,
        size_agnostic=True,
    )
)
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
register(AgentSpec("mlp", "Flatten, 2 hidden layers of 256, MLP head; board-bound", MLPPolicy))
register(
    AgentSpec(
        "cnn-tiny",
        "3 conv layers of width 16 (one stride-2), adaptive 4x4 pool, MLP head",
        _cnn_of_width(16),
        size_agnostic=True,
    )
)
register(
    AgentSpec(
        "cnn-small",
        "3 conv layers of width 32 (one stride-2), adaptive 4x4 pool, MLP head",
        _cnn_of_width(32),
        size_agnostic=True,
    )
)
register(
    AgentSpec(
        "cnn-medium",
        "3 conv layers of width 64 (one stride-2), adaptive 4x4 pool, MLP head",
        _cnn_of_width(64),
        size_agnostic=True,
    )
)
register(
    AgentSpec(
        "attention",
        "Conv stem, one token per cell, 2 transformer layers of 64, mean-pooled head",
        AttentionPolicy,
        size_agnostic=True,
    )
)
register(
    AgentSpec(
        "recurrent",
        "cnn-small encoder, then a GRU of 256 (or an LSTM with agent.kwargs cell: lstm), MLP head",
        RecurrentCNNPolicy,
        size_agnostic=True,
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
