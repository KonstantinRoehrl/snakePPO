"""Typed experiment configuration: dataclass defaults, YAML files and ``--set`` overrides.

Load order is defaults -> YAML file (partial allowed) -> ``section.key=value`` overrides.
Unknown sections or keys and values of the wrong type raise :class:`ConfigError`.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LR_SCHEDULES = ("constant", "linear")
ENTROPY_SCHEDULES = ("constant", "linear", "exponential")


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
    apple: float = 1.0
    death: float = -3.0
    approach: float = 0.02
    retreat: float = 0.03


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


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


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
    return validate(dataclasses.replace(config, **updates))


def apply_overrides(config: Config, overrides: Sequence[str]) -> Config:
    """Apply ``section.key=value`` strings in order; each value is parsed as a YAML scalar."""
    for override in overrides:
        path, sep, raw = override.partition("=")
        section_name, dot, key = path.strip().partition(".")
        if not sep or not dot or not section_name or not key:
            raise ConfigError(f"override '{override}' must look like section.key=value")
        config = config_from_dict({section_name: {key: yaml.safe_load(raw)}}, base=config)
    return config


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
    for name in ("total_env_steps", "eval_interval", "eval_episodes"):
        if getattr(config.train, name) < 1:
            raise ConfigError(f"train.{name} must be at least 1")
    return config
