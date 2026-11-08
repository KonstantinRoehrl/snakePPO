import subprocess
import sys

import pytest
import torch

from snakeppo.observation import BODY_ORDER_TAIL
from snakeppo.viewer import (
    BODY_AT_TAIL,
    BODY_NEAR_HEAD,
    HEAD,
    HEADER_HEIGHT,
    PADDING,
    body_colour,
)


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from snakeppo.viewer import Viewer

    with Viewer(board_size=8, num_games=4) as v:
        yield v


def _grids(n=4):
    grids = torch.zeros(n, 4, 8, 8)
    grids[:, 3, 0, :] = 1
    grids[:, 1, 4, 4] = 1
    return grids


def _orders(n=4):
    return torch.zeros(n, 8, 8)


def test_importing_the_viewer_does_not_import_pygame():
    code = "import sys, snakeppo.viewer; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_draw_keeps_one_font_and_reports_open(viewer):
    font = viewer._font
    assert viewer.draw(_grids(), [0, 1, 2, 3], _orders()) is True
    assert viewer.draw(_grids(), [0, 1, 2, 3], _orders()) is True
    assert viewer._font is font


def test_closing_the_window_stops_drawing_cleanly(viewer):
    import pygame

    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert viewer.draw(_grids(), [0, 0, 0, 0], _orders()) is False
    assert viewer.is_open is False
    assert viewer.draw(_grids(), [0, 0, 0, 0], _orders()) is False
    viewer.close()


def test_body_colour_runs_from_bright_at_the_head_to_dark_at_the_tail():
    assert body_colour(1.0) == BODY_NEAR_HEAD == (50, 200, 50)
    assert body_colour(BODY_ORDER_TAIL) == BODY_AT_TAIL == (20, 80, 20)
    assert body_colour(0.0) == BODY_AT_TAIL  # clamped below the tail value
    assert body_colour(1.5) == BODY_NEAR_HEAD


def test_body_colours_of_a_snake_are_distinct_and_darken_towards_the_tail():
    orders = [1 - 0.9 * i / 5 for i in range(1, 6)]  # a length-6 snake behind its head
    colours = [body_colour(order) for order in orders]
    assert len(set(colours)) == len(colours)
    for nearer, further in zip(colours, colours[1:], strict=False):
        assert all(a >= b for a, b in zip(nearer, further, strict=True))
    assert colours[-1] == BODY_AT_TAIL


def _pixel(viewer, row, col):
    """The colour at the centre of one cell of game 0."""
    x = PADDING + col * viewer.cell_size + viewer.cell_size // 2
    y = PADDING + HEADER_HEIGHT + row * viewer.cell_size + viewer.cell_size // 2
    return tuple(viewer._screen.get_at((x, y)))[:3]


def test_draw_shades_each_body_cell_by_its_order(viewer):
    grids, orders = _grids(), _orders()
    snake = [(4, 4), (4, 5), (4, 6)]  # head first, as body-order values 1.0, 0.55, 0.1
    for (row, col), order in zip(snake, (1.0, 0.55, BODY_ORDER_TAIL), strict=True):
        grids[0, 0, row, col] = 1
        orders[0, row, col] = order
    assert viewer.draw(grids, [0, 0, 0, 0], orders) is True
    middle, tail = _pixel(viewer, 4, 5), _pixel(viewer, 4, 6)
    assert middle == body_colour(0.55) and tail == BODY_AT_TAIL
    assert sum(middle) > sum(tail)
    assert _pixel(viewer, 4, 4) == HEAD  # the head keeps its own colour
