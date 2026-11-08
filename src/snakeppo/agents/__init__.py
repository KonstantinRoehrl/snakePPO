"""Policy networks and the registry that builds them by name."""

from snakeppo.agents.networks import PolicyNetwork
from snakeppo.agents.recurrent import PolicyPlayer, RecurrentPolicy, episode_starts
from snakeppo.agents.registry import (
    REGISTRY,
    AgentSpec,
    available_agents,
    build_agent,
    count_parameters,
    register,
)

__all__ = [
    "REGISTRY",
    "AgentSpec",
    "PolicyNetwork",
    "PolicyPlayer",
    "RecurrentPolicy",
    "available_agents",
    "build_agent",
    "count_parameters",
    "episode_starts",
    "register",
]
