"""Typed experiment configuration: dataclass defaults, YAML files and ``--set`` overrides.

Load order is defaults -> YAML file (partial allowed) -> ``section.key=value`` overrides.
Unknown sections or keys and values of the wrong type raise :class:`ConfigError`.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LR_SCHEDULES = ("constant", "linear")
ENTROPY_SCHEDULES = ("constant", "linear", "exponential")
REWARD_STRATEGIES = ("classic", "sparse", "distance", "potential", "hunger", "free_space")
CURRICULUM_GROWTH = ("off", "board", "length", "both")
# Reward coefficients that only one strategy reads; none of them may be negative.
STRATEGY_PARAMETERS = (
    "distance_scale",
    "potential_scale",
    "timeout_penalty",
    "hunger_penalty",
    "free_space_penalty",
)


class ConfigError(ValueError):
    """A config file or override names an unknown key or carries a value of the wrong type."""


@dataclass
class EnvConfig:
    board_size: int = 16
    num_envs: int = 64
    starvation_steps: int = 512
    seed: int = 42


@dataclass
class RewardConfig:
    strategy: str = "classic"
    apple: float = 1.0
    death: float = -3.0
    approach: float = 0.02
    retreat: float = 0.03
    distance_scale: float = 0.1
    potential_scale: float = 0.1
    timeout_penalty: float = 1.0
    hunger_penalty: float = 0.01
    free_space_penalty: float = 0.1


@dataclass
class AgentConfig:
    name: str = "cnn"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    entropy_coef_final: float = 0.0005
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    lr_schedule: str = "constant"
    clip_value_loss: bool = False
    entropy_schedule: str = "constant"
    target_kl: float | None = None


@dataclass
class TrainConfig:
    total_env_steps: int = 10_000_000
    eval_interval: int = 10
    eval_episodes: int = 64
    device: str = "auto"
    name: str = "run"
    runs_dir: str = "runs"
    deterministic: bool = False
    checkpoint_interval: int = 10
    group: str | None = None


@dataclass
class CurriculumConfig:
    grow: str = "off"
    stages: int = 5
    start_board: int = 8
    start_goal: int = 4
    promote_fraction: float = 0.3
    promote_window: int = 64
    min_updates: int = 10
    max_updates: int = 100


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


@dataclass(frozen=True)
class CurriculumStage:
    """One rung of the curriculum: an arena side (walls included) and an optional goal length."""

    arena: int
    goal: int | None

    @property
    def capacity(self) -> int:
        """Most apples one episode can eat here: the playable cells, or ``goal - 1``."""
        playable = (self.arena - 2) ** 2
        return playable if self.goal is None else min(playable, self.goal - 1)


def curriculum_ladder(curriculum: CurriculumConfig, board_size: int) -> tuple[CurriculumStage, ...]:
    """Every stage in order; the last is always the full board with no goal.

    A growing board widens the arena linearly from ``start_board`` to ``board_size``, rounded
    half up; a growing length doubles the goal from ``start_goal``. ``off`` is the final stage.
    """
    final = CurriculumStage(board_size, None)
    if curriculum.grow == "off":
        return (final,)
    grow_board = curriculum.grow in ("board", "both")
    grow_length = curriculum.grow in ("length", "both")
    last = curriculum.stages - 1
    span = board_size - curriculum.start_board
    stages = []
    for index in range(last):
        arena = board_size
        if grow_board:
            arena = min(
                board_size, curriculum.start_board + (2 * span * index + last) // (2 * last)
            )
        goal = curriculum.start_goal * 2**index if grow_length else None
        stages.append(CurriculumStage(arena, goal))
    return (*stages, final)


def total_updates(config: Config) -> int:
    """Number of PPO updates a run of ``config`` loops to."""
    batch = config.ppo.rollout_steps * config.env.num_envs
    return math.ceil(config.train.total_env_steps / batch)


def config_to_dict(config: Config) -> dict[str, Any]:
    return dataclasses.asdict(config)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        options = typing.get_args(annotation)
        if value is None and type(None) in options:
            return None
        (annotation,) = [option for option in options if option is not type(None)]
    if annotation is bool:
        if isinstance(value, bool):
            return value
    elif annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif annotation is float:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        elif isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif annotation is str:
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            raise ConfigError(
                f"{where} must be text, got {value!r}: YAML reads off, on, yes and no as "
                f"booleans, so quote the value (for example {where}='off')"
            )
    elif typing.get_origin(annotation) is dict and isinstance(value, dict):
        return dict(value)
    expected = getattr(annotation, "__name__", str(annotation))
    raise ConfigError(f"{where} must be {expected}, got {value!r}")


def _merge_section(section: Any, values: Mapping[str, Any], section_name: str) -> Any:
    hints = typing.get_type_hints(type(section))
    known = {f.name for f in dataclasses.fields(section)}
    updates = {}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(
                f"unknown key '{section_name}.{key}'; known keys: {', '.join(sorted(known))}"
            )
        updates[key] = _coerce(value, hints[key], f"{section_name}.{key}")
    return dataclasses.replace(section, **updates)


def config_from_dict(data: Mapping[str, Any], base: Config | None = None) -> Config:
    """Merge a (possibly partial) nested mapping onto ``base`` (defaults when omitted)."""
    return validate(_merge(data, base))


def _merge(data: Mapping[str, Any], base: Config | None) -> Config:
    config = base if base is not None else Config()
    sections = {f.name for f in dataclasses.fields(Config)}
    updates = {}
    for section_name, values in data.items():
        if section_name not in sections:
            raise ConfigError(
                f"unknown section '{section_name}'; known sections: {', '.join(sorted(sections))}"
            )
        if not isinstance(values, Mapping):
            raise ConfigError(f"section '{section_name}' must be a mapping, got {values!r}")
        updates[section_name] = _merge_section(getattr(config, section_name), values, section_name)
    return dataclasses.replace(config, **updates)


def apply_overrides(config: Config, overrides: Sequence[str]) -> Config:
    """Apply ``section.key=value`` strings in order; each value is parsed as a YAML scalar.

    The result is validated once, after the last override, so settings that only make sense
    together (``curriculum.grow`` and its budget) can be given in any order.
    """
    for override in overrides:
        path, sep, raw = override.partition("=")
        section_name, dot, key = path.strip().partition(".")
        if not sep or not dot or not section_name or not key:
            raise ConfigError(f"override '{override}' must look like section.key=value")
        config = _merge({section_name: {key: yaml.safe_load(raw)}}, base=config)
    return validate(config)


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> Config:
    config = Config()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path} must contain a mapping of config sections")
        config = config_from_dict(data, base=config)
    return apply_overrides(config, overrides)


def save_config(config: Config, path: str | Path) -> None:
    text = yaml.safe_dump(config_to_dict(config), sort_keys=False)
    Path(path).write_text(text, encoding="utf-8")


def validate(config: Config) -> Config:
    """Reject values that are well-typed but meaningless; returns ``config`` unchanged."""
    if config.ppo.lr_schedule not in LR_SCHEDULES:
        raise ConfigError(f"ppo.lr_schedule must be one of {', '.join(LR_SCHEDULES)}")
    if config.ppo.entropy_schedule not in ENTROPY_SCHEDULES:
        raise ConfigError(f"ppo.entropy_schedule must be one of {', '.join(ENTROPY_SCHEDULES)}")
    if config.ppo.entropy_schedule == "exponential" and (
        config.ppo.entropy_coef <= 0 or config.ppo.entropy_coef_final <= 0
    ):
        raise ConfigError(
            "ppo.entropy_coef and ppo.entropy_coef_final must both be > 0 "
            "for the exponential schedule"
        )
    if config.env.board_size < 5:
        raise ConfigError("env.board_size must be at least 5 (a 3x3 playable area)")
    for name in ("num_envs", "starvation_steps"):
        if getattr(config.env, name) < 1:
            raise ConfigError(f"env.{name} must be at least 1")
    for name in ("rollout_steps", "update_epochs", "minibatch_size"):
        if getattr(config.ppo, name) < 1:
            raise ConfigError(f"ppo.{name} must be at least 1")
    for name in ("total_env_steps", "eval_interval", "eval_episodes", "checkpoint_interval"):
        if getattr(config.train, name) < 1:
            raise ConfigError(f"train.{name} must be at least 1")
    if config.train.group is not None and not config.train.group.strip():
        raise ConfigError("train.group must be a non-empty name or null")
    _validate_reward(config.reward)
    _validate_curriculum(config)
    return config


def _validate_reward(reward: RewardConfig) -> None:
    if reward.strategy not in REWARD_STRATEGIES:
        raise ConfigError(f"reward.strategy must be one of {', '.join(REWARD_STRATEGIES)}")
    for name in STRATEGY_PARAMETERS:
        if getattr(reward, name) < 0:
            raise ConfigError(f"reward.{name} must be >= 0")


def _validate_curriculum(config: Config) -> None:
    """Every check past ``grow`` runs only with a curriculum on, so ``off`` never fails here."""
    curriculum, board_size = config.curriculum, config.env.board_size
    if curriculum.grow not in CURRICULUM_GROWTH:
        raise ConfigError(f"curriculum.grow must be one of {', '.join(CURRICULUM_GROWTH)}")
    if curriculum.grow == "off":
        return
    if curriculum.stages < 2:
        raise ConfigError("curriculum.stages must be at least 2")
    if not 5 <= curriculum.start_board <= board_size:
        raise ConfigError(
            f"curriculum.start_board must be between 5 and env.board_size ({board_size})"
        )
    if curriculum.start_goal < 2:
        raise ConfigError("curriculum.start_goal must be at least 2")
    if not 0 < curriculum.promote_fraction <= 1:
        raise ConfigError("curriculum.promote_fraction must be in (0, 1]")
    for name in ("promote_window", "min_updates"):
        if getattr(curriculum, name) < 1:
            raise ConfigError(f"curriculum.{name} must be at least 1")
    if curriculum.max_updates < curriculum.min_updates:
        raise ConfigError("curriculum.max_updates must be at least curriculum.min_updates")
    for index, stage in enumerate(curriculum_ladder(curriculum, board_size)):
        playable = (stage.arena - 2) ** 2
        if stage.goal is not None and stage.goal > playable:
            raise ConfigError(
                f"curriculum stage {index} goal {stage.goal} exceeds its {playable} playable cells"
            )
    budget, updates = (curriculum.stages - 1) * curriculum.max_updates, total_updates(config)
    if budget >= updates:
        raise ConfigError(
            f"curriculum needs (stages - 1) * max_updates ({budget}) below the run's "
            f"{updates} updates, so training ends on the full board"
        )
