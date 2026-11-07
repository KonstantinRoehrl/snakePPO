from pathlib import Path

import pytest
import yaml

from snakeppo.config import (
    Config,
    ConfigError,
    apply_overrides,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)


def test_defaults_are_the_vanilla_baseline():
    config = Config()
    assert config.env.board_size == 16
    assert config.env.num_envs == 64
    assert config.env.starvation_steps == 512
    assert config.reward.apple == 1.0
    assert config.reward.death == -3.0
    assert config.ppo.rollout_steps == 256
    assert config.ppo.minibatch_size == 2048
    assert config.ppo.lr_schedule == "constant"
    assert config.ppo.entropy_schedule == "constant"
    assert config.ppo.clip_value_loss is False
    assert config.ppo.target_kl is None
    assert config.train.deterministic is False
    assert config.agent.name == "cnn"


def test_partial_yaml_file_merges_onto_defaults(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("ppo:\n  lr: 0.001\nagent:\n  name: resnet\n", encoding="utf-8")
    config = load_config(path)
    assert config.ppo.lr == 0.001
    assert config.agent.name == "resnet"
    assert config.ppo.gamma == 0.99


def test_overrides_parse_yaml_scalars_and_apply_in_order():
    config = apply_overrides(
        Config(),
        ["ppo.target_kl=0.02", "ppo.clip_value_loss=true", "ppo.lr=3e-4", "env.seed=7"],
    )
    assert config.ppo.target_kl == 0.02
    assert config.ppo.clip_value_loss is True
    assert config.ppo.lr == 3e-4
    assert config.env.seed == 7
    assert apply_overrides(config, ["ppo.target_kl=null"]).ppo.target_kl is None


def test_int_is_accepted_for_float_fields():
    assert apply_overrides(Config(), ["reward.apple=2"]).reward.apple == 2.0


@pytest.mark.parametrize(
    "override",
    ["ppo.does_not_exist=1", "nosuchsection.lr=1", "ppo.lr", "lr=1"],
)
def test_bad_override_raises(override):
    with pytest.raises(ConfigError):
        apply_overrides(Config(), [override])


def test_unknown_key_in_file_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("ppo:\n  learning_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ppo.learning_rate"):
        load_config(path)


@pytest.mark.parametrize(
    "data",
    [
        {"env": {"num_envs": "many"}},
        {"env": {"num_envs": 2.5}},
        {"ppo": {"clip_value_loss": 1}},
        {"ppo": {"lr_schedule": "cosine"}},
        {"env": {"board_size": 3}},
    ],
)
def test_wrong_types_and_values_raise(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_saved_config_round_trips_exactly(tmp_path):
    config = apply_overrides(
        Config(), ["ppo.target_kl=0.015", "agent.kwargs={channels: 32}", "train.name=x"]
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config


PRESETS = Path(__file__).resolve().parents[1] / "configs"


def test_baseline_preset_states_exactly_the_code_defaults():
    expected = apply_overrides(Config(), ["train.name=baseline"])
    assert load_config(PRESETS / "baseline.yaml") == expected


def test_baseline_preset_names_every_config_key():
    data = yaml.safe_load((PRESETS / "baseline.yaml").read_text(encoding="utf-8"))
    defaults = config_to_dict(Config())
    assert {s: set(data[s]) for s in data} == {s: set(defaults[s]) for s in defaults}


def test_bookkeeping_keys_default_to_today_and_are_validated():
    train = Config().train
    assert train.checkpoint_interval == 10
    assert train.group is None
    assert apply_overrides(Config(), ["train.group=sweep@1"]).train.group == "sweep@1"
    with pytest.raises(ConfigError, match="train.checkpoint_interval must be at least 1"):
        apply_overrides(Config(), ["train.checkpoint_interval=0"])
    with pytest.raises(ConfigError, match="train.group must be a non-empty name or null"):
        apply_overrides(Config(), ["train.group=' '"])


def test_tuned_preset_turns_every_switch_on():
    ppo = load_config(PRESETS / "tuned.yaml").ppo
    assert ppo.lr_schedule == "linear"
    assert ppo.clip_value_loss is True
    assert ppo.entropy_schedule != "constant"
    assert ppo.target_kl is not None


@pytest.mark.parametrize("path", sorted(PRESETS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_preset_loads(path):
    load_config(path)


def test_exponential_entropy_schedule_needs_a_positive_start_coefficient():
    with pytest.raises(
        ConfigError, match=r"ppo\.entropy_coef.*ppo\.entropy_coef_final.*exponential"
    ):
        apply_overrides(Config(), ["ppo.entropy_schedule=exponential", "ppo.entropy_coef=0"])
