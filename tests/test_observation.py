import subprocess
import sys

import pytest

from snakeppo.config import ObservationConfig
from snakeppo.observation import BODY_ORDER_TAIL, ObservationSpec, describe_record

DISPLAY_CHANNELS = ("body", "head", "food", "wall")


def test_the_default_spec_is_todays_binary_full_board_observation():
    spec = ObservationSpec.from_config(ObservationConfig())
    assert spec == ObservationSpec()
    assert spec.view_size is None
    assert spec.channels == DISPLAY_CHANNELS
    assert spec.is_binary is True
    assert spec.size(16) == 16
    assert spec.record(16) == {"channels": list(DISPLAY_CHANNELS), "size": 16, "view": "full"}


def test_from_config_keeps_view_size_only_for_the_local_view():
    assert ObservationSpec.from_config(ObservationConfig(view_size=7)).view_size is None
    local = ObservationSpec.from_config(ObservationConfig(view="local", view_size=7))
    assert local == ObservationSpec(view_size=7)
    full = ObservationConfig(body="order", tail=True, hunger=True)
    assert ObservationSpec.from_config(full) == ObservationSpec(
        body="order", tail=True, hunger=True
    )


@pytest.mark.parametrize(
    ("spec", "channels"),
    [
        (ObservationSpec(body="order"), ("body_order", "head", "food", "wall")),
        (ObservationSpec(tail=True), (*DISPLAY_CHANNELS, "tail")),
        (ObservationSpec(hunger=True), (*DISPLAY_CHANNELS, "hunger")),
        (ObservationSpec(view_size=5), DISPLAY_CHANNELS),
        (
            ObservationSpec(body="order", tail=True, hunger=True, view_size=9),
            ("body_order", "head", "food", "wall", "tail", "hunger"),
        ),
        (ObservationSpec(tail=True, hunger=True), (*DISPLAY_CHANNELS, "tail", "hunger")),
    ],
)
def test_every_switch_names_its_channels_in_the_fixed_order_and_is_not_binary(spec, channels):
    assert spec.channels == channels
    assert spec.is_binary is False


def test_the_local_view_sets_the_size_whatever_the_board():
    spec = ObservationSpec(view_size=11)
    assert [spec.size(board) for board in (8, 16, 30)] == [11, 11, 11]
    assert spec.record(30) == {"channels": list(DISPLAY_CHANNELS), "size": 11, "view": "local"}


def test_a_local_view_as_large_as_the_board_still_records_a_different_observation():
    assert ObservationSpec(view_size=15).record(15) != ObservationSpec().record(15)


def test_describe_record_names_the_channels_the_size_and_the_view():
    record = ObservationSpec(body="order").record(16)
    assert describe_record(record) == "channels body_order, head, food, wall on a 16x16 full view"
    local = ObservationSpec(tail=True, view_size=7).record(16)
    assert describe_record(local) == "channels body, head, food, wall, tail on a 7x7 local view"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"body": "onehot"}, "ObservationSpec body must be one of binary, order, got 'onehot'"),
        ({"view_size": 4}, "ObservationSpec view_size must be None or an odd number of at least 5"),
        ({"view_size": 3}, "ObservationSpec view_size must be None or an odd number of at least 5"),
    ],
)
def test_a_spec_built_directly_rejects_meaningless_values(fields, message):
    with pytest.raises(ValueError, match=message):
        ObservationSpec(**fields)


def test_body_order_runs_from_one_at_the_head_to_a_fixed_tail_value():
    assert BODY_ORDER_TAIL == 0.1


def test_importing_the_observation_module_does_not_import_the_env():
    code = "import sys, snakeppo.observation; assert 'snakeppo.env' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
