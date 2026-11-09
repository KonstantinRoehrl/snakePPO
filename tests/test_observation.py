import subprocess
import sys

import pytest
import torch

from snakeppo.config import ObservationConfig
from snakeppo.observation import (
    BODY_ORDER_TAIL,
    FOOD_PROJECTED,
    TAIL_NEXT,
    ObservationSpec,
    body_order_grid,
    describe_record,
    local_view,
    tail_grid,
)

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


# ------------------------------------------------------------ tensor helpers
def _snakes(*snakes, max_len=36):
    """``body_positions`` (head first, -1 past the tail) and ``snake_len`` for hand-built snakes."""
    positions = torch.full((len(snakes), max_len, 2), -1, dtype=torch.long)
    for index, snake in enumerate(snakes):
        positions[index, : len(snake)] = torch.tensor(snake)
    return positions, torch.tensor([len(snake) for snake in snakes])


def _plane(cells, size=8):
    plane = torch.zeros(size, size)
    for (row, col), value in cells.items():
        plane[row, col] = value
    return plane


def test_body_order_runs_evenly_from_the_head_to_the_tail():
    long_snake = [(4, 4), (4, 5), (5, 5), (5, 4), (5, 3)]
    positions, lengths = _snakes(long_snake, [(2, 2)])
    grid = body_order_grid(positions, lengths, 8)
    assert grid.shape == (2, 8, 8) and grid.dtype == torch.float32
    values = [grid[0, row, col].item() for row, col in long_snake]
    assert values == pytest.approx([1.0, 0.775, 0.55, 0.325, 0.1])
    assert values[-1] == pytest.approx(BODY_ORDER_TAIL)
    assert int((grid[0] > 0).sum()) == 5
    assert torch.equal(grid[1], _plane({(2, 2): 1.0}))  # a length-1 snake is all head


def test_tail_plane_marks_the_tail_and_the_cell_it_moves_into_next():
    one, two = [(4, 4)], [(4, 4), (5, 4)]
    four = [(4, 4), (5, 4), (5, 5), (5, 6)]
    positions, lengths = _snakes(one, two, four)
    grid = tail_grid(positions, lengths, 8)
    assert grid.shape == (3, 8, 8) and grid.dtype == torch.float32
    assert TAIL_NEXT == 0.5
    assert torch.equal(grid[0], _plane({(4, 4): 1.0}))
    assert torch.equal(grid[1], _plane({(5, 4): 1.0, (4, 4): 0.5}))  # length 2: the head is next
    assert torch.equal(grid[2], _plane({(5, 6): 1.0, (5, 5): 0.5}))


def _display_grid(head, food, size=8, extra=None):
    """One env's unrotated (1, C, S, S) grid: body = head only, food, the wall ring, extras."""
    grid = torch.zeros(1, 4 + (extra is not None), size, size)
    grid[0, 0, head[0], head[1]] = 1.0
    grid[0, 1, head[0], head[1]] = 1.0
    grid[0, 2, food[0], food[1]] = 1.0
    grid[0, 3] = 1.0
    grid[0, 3, 1:-1, 1:-1] = 0.0
    if extra is not None:
        grid[0, 4] = extra
    return grid


def test_local_view_centres_the_head_and_draws_off_board_cells_as_wall():
    tail = _plane({(1, 2): 1.0})
    grid = _display_grid((1, 1), (3, 2), extra=tail)
    crop = local_view(grid, torch.tensor([[1, 1]]), torch.tensor([[3, 2]]), 5)
    assert crop.shape == (1, 5, 5, 5)
    walls = torch.zeros(5, 5)
    walls[:2, :] = 1.0  # row -1 is off the board, row 0 is the wall ring
    walls[:, :2] = 1.0
    assert torch.equal(crop[0, 3], walls)
    assert torch.equal(crop[0, 1], _plane({(2, 2): 1.0}, size=5))
    assert torch.equal(crop[0, 0], _plane({(2, 2): 1.0}, size=5))
    assert torch.equal(crop[0, 2], _plane({(4, 3): 1.0}, size=5))  # food inside stays 1.0
    assert torch.equal(crop[0, 4], _plane({(2, 3): 1.0}, size=5))  # extra channels are cropped


@pytest.mark.parametrize(
    ("food", "cell"),
    [
        ((1, 10), (0, 3)),  # d = (-7, 2): 2 * 2 / 7 = 0.57 rounds to 1
        ((8, 14), (2, 4)),  # straight right
        ((9, 12), (3, 4)),  # d = (1, 4): 1 * 2 / 4 = 0.5 rounds away from zero to 1
        ((7, 12), (1, 4)),  # d = (-1, 4): -0.5 rounds away from zero to -1
        ((13, 3), (4, 0)),  # d = (5, -5): the corner
        ((11, 14), (3, 4)),  # d = (3, 6): exactly 1
        ((8, 3), (2, 0)),  # straight left
    ],
)
def test_food_outside_the_crop_is_projected_onto_its_edge(food, cell):
    grid = _display_grid((8, 8), food, size=16)
    crop = local_view(grid, torch.tensor([[8, 8]]), torch.tensor([food]), 5)
    assert FOOD_PROJECTED == 0.5
    assert torch.equal(crop[0, 2], _plane({cell: 0.5}, size=5))
    assert 0 in cell or 4 in cell


def test_local_view_handles_every_env_of_a_batch_on_its_own():
    heads, foods = [(8, 8), (1, 1), (14, 7), (8, 8)], [(1, 10), (3, 2), (13, 7), (7, 9)]
    grid = torch.cat([_display_grid(h, f, size=16) for h, f in zip(heads, foods, strict=True)])
    batch = local_view(grid, torch.tensor(heads), torch.tensor(foods), 7)
    for index, (head, food) in enumerate(zip(heads, foods, strict=True)):
        single = local_view(grid[index : index + 1], torch.tensor([head]), torch.tensor([food]), 7)
        assert torch.equal(batch[index], single[0])
