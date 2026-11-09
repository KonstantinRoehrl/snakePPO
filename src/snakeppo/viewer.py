"""Pygame window showing up to four games side by side; pygame is imported only when used.

The body is shaded by its body order (1.0 at the head down to ``BODY_ORDER_TAIL`` at the tail),
from bright green next to the head to dark green at the tail, whatever the policy observes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from snakeppo.observation import BODY_ORDER_TAIL

MAX_GAMES = 4
PADDING = 10
HEADER_HEIGHT = 40
WALL, FOOD, HEAD, EMPTY, GRID_LINE = (
    (50, 50, 50),
    (255, 50, 50),
    (120, 255, 120),
    (0, 0, 0),
    (30, 30, 30),
)
BODY_NEAR_HEAD = (50, 200, 50)  # body order 1.0
BODY_AT_TAIL = (20, 80, 20)  # body order BODY_ORDER_TAIL
TEXT = (255, 255, 255)


def body_colour(order: float) -> tuple[int, int, int]:
    """Body colour at a body-order value, linear from the head's 1.0 to the tail, clamped."""
    fraction = min(max((1.0 - order) / (1.0 - BODY_ORDER_TAIL), 0.0), 1.0)
    red, green, blue = (
        round(near + (tail - near) * fraction)
        for near, tail in zip(BODY_NEAR_HEAD, BODY_AT_TAIL, strict=True)
    )
    return red, green, blue


class Viewer:
    def __init__(self, board_size: int, num_games: int = MAX_GAMES, cell_size: int = 20) -> None:
        import pygame

        self._pygame: Any = pygame
        self.board_size = board_size
        self.num_games = max(1, min(num_games, MAX_GAMES))
        self.cell_size = cell_size
        columns = min(self.num_games, 2)
        rows = (self.num_games + 1) // 2
        board_px = board_size * cell_size
        width = columns * board_px + (columns + 1) * PADDING
        height = rows * (board_px + HEADER_HEIGHT) + (rows + 1) * PADDING
        pygame.init()
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("snakeppo")
        self._font = pygame.font.Font(None, 28)
        self._clock = pygame.time.Clock()
        self.is_open = True

    def draw(self, grids: torch.Tensor, scores: Sequence[float], body_order: torch.Tensor) -> bool:
        """Draw unrotated grids (N, 4, S, S) with their body order (N, S, S).

        ``body_order`` comes from ``env.render_body_order()``. Returns False once the window has
        been closed.
        """
        pygame = self._pygame
        if not self.is_open:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        cells = grids[: self.num_games].detach().cpu().numpy() > 0
        orders = body_order[: self.num_games].detach().cpu().numpy()
        board_px = self.board_size * self.cell_size
        self._screen.fill(EMPTY)
        for game, (body, head, food, wall) in enumerate(cells):
            x0 = PADDING + (game % 2) * (board_px + PADDING)
            y0 = PADDING + (game // 2) * (board_px + HEADER_HEIGHT + PADDING) + HEADER_HEIGHT
            label = self._font.render(f"Game {game} | Score: {int(scores[game])}", True, TEXT)
            self._screen.blit(label, (x0, y0 - HEADER_HEIGHT + 8))
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if wall[row, col]:
                        color = WALL
                    elif food[row, col]:
                        color = FOOD
                    elif head[row, col]:
                        color = HEAD
                    elif body[row, col]:
                        color = body_colour(float(orders[game, row, col]))
                    else:
                        color = EMPTY
                    rect = pygame.Rect(
                        x0 + col * self.cell_size,
                        y0 + row * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self._screen, color, rect)
                    pygame.draw.rect(self._screen, GRID_LINE, rect, 1)
        pygame.display.flip()
        self._clock.tick(100)
        return True

    def close(self) -> None:
        if self.is_open:
            self.is_open = False
            self._pygame.display.quit()
            self._pygame.quit()

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
