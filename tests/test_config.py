import dataclasses
from pathlib import Path

import pytest
import yaml

from snakeppo.config import (
    CURRICULUM_GROWTH,
    OBSERVATION_BODIES,
    OBSERVATION_VIEWS,
    REWARD_STRATEGIES,
    Config,
    ConfigError,
    CurriculumConfig,
    CurriculumStage,
    ObservationConfig,
    apply_overrides,
    config_from_dict,
    config_to_dict,
    curriculum_ladder,
    load_config,
    save_config,
    total_updates,
)
from snakeppo.results import switch_unit, switches


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


def _preset_id(path):
    return path.relative_to(PRESETS).as_posix()


@pytest.mark.parametrize("path", sorted(PRESETS.glob("**/*.yaml")), ids=_preset_id)
def test_every_preset_loads(path):
    load_config(path)


# The switch every preset in a folder turns on; each observation preset turns on its own key.
FOLDER_SWITCHES = {
    "rewards": "reward.strategy",
    "curriculum": "curriculum.grow",
    "agents": "agent.name",
}


@pytest.mark.parametrize("path", sorted(PRESETS.glob("*/*.yaml")), ids=_preset_id)
def test_every_folder_preset_turns_on_one_switch(path):
    config = load_config(path)
    units = {switch_unit(key) for key in switches(config)}
    assert len(units) == 1, units
    folder = path.parent.name
    if folder == "observation":
        assert units.pop().startswith("observation.")
    else:
        assert units == {FOLDER_SWITCHES[folder]}
    assert config.train.name.startswith(folder.removesuffix("s"))


def test_exponential_entropy_schedule_needs_a_positive_start_coefficient():
    with pytest.raises(
        ConfigError, match=r"ppo\.entropy_coef.*ppo\.entropy_coef_final.*exponential"
    ):
        apply_overrides(Config(), ["ppo.entropy_schedule=exponential", "ppo.entropy_coef=0"])


# ------------------------------------------------------------ reward and curriculum switches
def test_reward_and_curriculum_defaults_are_the_baseline():
    config = Config()
    assert config.reward.strategy == "classic"
    assert config.reward.approach == 0.02 and config.reward.retreat == 0.03
    assert config.curriculum.grow == "off"
    assert REWARD_STRATEGIES[0] == "classic" and CURRICULUM_GROWTH[0] == "off"


def test_total_updates_is_importable_from_config_and_runs():
    from snakeppo import runs

    config = apply_overrides(Config(), ["train.total_env_steps=1000", "ppo.rollout_steps=16"])
    config = apply_overrides(config, ["env.num_envs=8"])
    assert total_updates(config) == runs.total_updates(config) == 8  # 1000 / 128 = 7.8


def _curriculum(**fields):
    return CurriculumConfig(**({"grow": "both"} | fields))


def test_ladder_with_both_grows_arena_and_goal_and_ends_on_the_real_task():
    ladder = curriculum_ladder(_curriculum(), 16)
    assert [(s.arena, s.goal) for s in ladder] == [
        (8, 4),
        (10, 8),
        (12, 16),
        (14, 32),
        (16, None),
    ]
    assert [s.capacity for s in ladder] == [3, 7, 15, 31, 196]


def test_ladder_grows_only_what_grow_names():
    board = curriculum_ladder(_curriculum(grow="board"), 16)
    assert [(s.arena, s.goal) for s in board] == [
        (8, None),
        (10, None),
        (12, None),
        (14, None),
        (16, None),
    ]
    length = curriculum_ladder(_curriculum(grow="length", stages=3), 16)
    assert [(s.arena, s.goal) for s in length] == [(16, 4), (16, 8), (16, None)]
    assert curriculum_ladder(CurriculumConfig(), 16) == (CurriculumStage(16, None),)


def test_ladder_rounds_intermediate_arenas_half_up():
    # 5 + 11 * 1/2 = 10.5 rounds to 11; 5 + 2 * 1/3 = 5.67 rounds to 6
    assert curriculum_ladder(_curriculum(grow="board", start_board=5, stages=3), 16)[1].arena == 11
    sides = [s.arena for s in curriculum_ladder(_curriculum(grow="board", start_board=5), 7)]
    assert sides == [5, 6, 6, 7, 7]


def test_capacity_counts_playable_cells_or_apples_to_the_goal():
    assert CurriculumStage(8, None).capacity == 36
    assert CurriculumStage(8, 4).capacity == 3
    assert CurriculumStage(5, 20).capacity == 9


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("reward.strategy=shaped", "reward.strategy must be one of classic, sparse"),
        ("reward.hunger_penalty=-0.1", "reward.hunger_penalty must be >= 0"),
        ("reward.distance_scale=-1", "reward.distance_scale must be >= 0"),
        ("curriculum.grow=sideways", "curriculum.grow must be one of off, board, length, both"),
        ("curriculum.grow=off", r"curriculum.grow must be text, got False.*quote"),
    ],
)
def test_reward_and_curriculum_selectors_are_validated(override, message):
    with pytest.raises(ConfigError, match=message):
        apply_overrides(Config(), [override])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["curriculum.stages=1"], "curriculum.stages must be at least 2"),
        (
            ["curriculum.start_board=4"],
            r"curriculum.start_board must be between 5 and env.board_size \(16\)",
        ),
        (
            ["curriculum.start_board=17"],
            r"curriculum.start_board must be between 5 and env.board_size \(16\)",
        ),
        (["curriculum.start_goal=1"], "curriculum.start_goal must be at least 2"),
        (["curriculum.promote_fraction=0"], r"curriculum.promote_fraction must be in \(0, 1\]"),
        (["curriculum.promote_fraction=1.5"], r"curriculum.promote_fraction must be in \(0, 1\]"),
        (["curriculum.promote_window=0"], "curriculum.promote_window must be at least 1"),
        (["curriculum.min_updates=0"], "curriculum.min_updates must be at least 1"),
        (
            ["curriculum.max_updates=5"],
            "curriculum.max_updates must be at least curriculum.min_updates",
        ),
        (["curriculum.start_goal=64"], "curriculum stage 0 goal 64 exceeds its 36 playable cells"),
        (
            ["curriculum.max_updates=200"],
            r"curriculum needs \(stages - 1\) \* max_updates \(800\) below the run's 611 updates",
        ),
    ],
)
def test_curriculum_settings_are_validated_when_it_is_on(overrides, message):
    with pytest.raises(ConfigError, match=message):
        apply_overrides(Config(), ["curriculum.grow=both", *overrides])


def test_curriculum_settings_are_not_checked_while_it_is_off():
    small = apply_overrides(Config(), ["env.board_size=5", "curriculum.stages=0"])
    assert small.curriculum.grow == "off"
    assert (
        config_from_dict({"curriculum": {"grow": "off", "max_updates": 0}}).curriculum.grow == "off"
    )


def test_curriculum_section_round_trips_through_a_saved_config(tmp_path):
    config = apply_overrides(Config(), ["curriculum.grow=board", "reward.strategy=potential"])
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config
    save_config(Config(), path)
    assert load_config(path).curriculum.grow == "off"


def test_overrides_are_validated_together_so_their_order_does_not_matter():
    config = apply_overrides(Config(), ["curriculum.grow=both", "curriculum.max_updates=50"])
    assert (config.curriculum.grow, config.curriculum.max_updates) == ("both", 50)
    with pytest.raises(ConfigError, match="curriculum needs"):
        apply_overrides(
            Config(),
            ["curriculum.max_updates=50", "curriculum.grow=both", "curriculum.max_updates=500"],
        )


# ------------------------------------------------------------ observation switches
def test_observation_defaults_are_the_baseline_and_the_section_comes_last():
    observation = Config().observation
    assert observation == ObservationConfig(
        body="binary", tail=False, hunger=False, view="full", view_size=11
    )
    assert OBSERVATION_BODIES == ("binary", "order")
    assert OBSERVATION_VIEWS == ("full", "local")
    assert [f.name for f in dataclasses.fields(Config)][-2:] == ["curriculum", "observation"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("observation.body=onehot", "observation.body must be one of binary, order"),
        ("observation.view=global", "observation.view must be one of full, local"),
        (
            "observation.view_size=4",
            "observation.view_size must be an odd number of at least 5, got 4",
        ),
        (
            "observation.view_size=3",
            "observation.view_size must be an odd number of at least 5, got 3",
        ),
        (
            "observation.view_size=12",
            "observation.view_size must be an odd number of at least 5, got 12",
        ),
        ("observation.tail=1", "observation.tail must be bool, got 1"),
        ("observation.view_size=7.0", "observation.view_size must be int, got 7.0"),
    ],
)
def test_observation_values_are_validated(override, message):
    with pytest.raises(ConfigError, match=message):
        apply_overrides(Config(), [override])


def test_view_size_is_checked_even_with_the_full_view():
    with pytest.raises(ConfigError, match="observation.view_size must be an odd number"):
        config_from_dict({"observation": {"view": "full", "view_size": 6}})


@pytest.mark.parametrize(
    "overrides",
    [
        ["observation.body=order"],
        ["observation.tail=true", "observation.hunger=true"],
        ["observation.view=local", "observation.view_size=5"],
        ["observation.view=local", "observation.view_size=31"],
    ],
)
def test_every_meaningful_observation_is_accepted(overrides):
    config = apply_overrides(Config(), overrides)
    assert config.observation != ObservationConfig()


def test_observation_section_round_trips_through_a_saved_config(tmp_path):
    config = apply_overrides(
        Config(),
        [
            "observation.body=order",
            "observation.tail=true",
            "observation.hunger=true",
            "observation.view=local",
            "observation.view_size=7",
        ],
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["observation"]["view_size"] == 7


def test_a_config_without_an_observation_section_gets_the_default_observation():
    assert config_from_dict({"env": {"board_size": 8}}).observation == ObservationConfig()
