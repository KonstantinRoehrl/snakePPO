import subprocess
import sys

import pytest
import torch


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


def test_importing_the_viewer_does_not_import_pygame():
    code = "import sys, snakeppo.viewer; assert 'pygame' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_draw_keeps_one_font_and_reports_open(viewer):
    font = viewer._font
    assert viewer.draw(_grids(), [0, 1, 2, 3]) is True
    assert viewer.draw(_grids(), [0, 1, 2, 3]) is True
    assert viewer._font is font


def test_closing_the_window_stops_drawing_cleanly(viewer):
    import pygame

    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert viewer.draw(_grids(), [0, 0, 0, 0]) is False
    assert viewer.is_open is False
    assert viewer.draw(_grids(), [0, 0, 0, 0]) is False
    viewer.close()
