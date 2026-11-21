from dataclasses import replace

import pytest

from snakeppo.builders import play_env_config, plays_any_size
from snakeppo.config import config_from_dict

LOCAL_VIEW = {"view": "local", "view_size": 5}


def _config(agent, observation=None):
    return config_from_dict(
        {"env": {"board_size": 8}, "agent": {"name": agent}, "observation": observation or {}}
    )


@pytest.mark.parametrize(
    ("agent", "observation", "expected"),
    [
        ("cnn", None, True),
        ("recurrent", None, True),
        ("cnn-spatial", None, False),
        ("mlp", None, False),
        ("mlp", LOCAL_VIEW, True),
    ],
)
def test_plays_any_size_needs_a_size_agnostic_network_or_a_local_view(agent, observation, expected):
    assert plays_any_size(_config(agent, observation)) is expected


def test_play_env_config_keeps_the_trained_board_when_no_size_is_given():
    config = _config("cnn-spatial")
    assert play_env_config(config, None) is config.env


def test_play_env_config_moves_a_size_agnostic_run_to_another_board():
    config = _config("cnn")
    assert play_env_config(config, 12) == replace(config.env, board_size=12)
    assert config.env.board_size == 8


def test_play_env_config_lets_a_board_bound_run_play_its_own_board_and_a_local_view_any():
    assert play_env_config(_config("cnn-spatial"), 8).board_size == 8
    assert play_env_config(_config("mlp", LOCAL_VIEW), 12).board_size == 12


def test_play_env_config_refuses_a_board_smaller_than_5():
    with pytest.raises(ValueError, match="^--board-size must be at least 5, got 4$"):
        play_env_config(_config("cnn"), 4)


def test_play_env_config_refuses_another_board_for_a_board_bound_run():
    with pytest.raises(ValueError) as error:
        play_env_config(_config("cnn-spatial"), 12)
    assert str(error.value) == (
        "agent 'cnn-spatial' is board-bound: it trained on a 8x8 board and cannot play 12x12; "
        "use a size-agnostic architecture (see 'snakeppo agents') or train with "
        "observation.view: local"
    )
