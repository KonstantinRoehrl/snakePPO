"""What the policy sees: the observation switches as one :class:`ObservationSpec`.

The channel order is fixed: the body (``body`` or ``body_order``), ``head``, ``food`` and
``wall``, then ``tail`` and ``hunger`` when they are on. The full view is the whole board; the
local view is a head-centred square of ``view_size`` cells, so the network's input side is
:meth:`ObservationSpec.size`. A run records its observation as :meth:`ObservationSpec.record`,
and a checkpoint is only ever loaded against an equal record.

The tensor helpers build the switched channels from the env's state; they are pure, so the env
calls them and the tests check them directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from snakeppo.config import OBSERVATION_BODIES, ObservationConfig

# Body-order value of the tail segment; the head is 1.0 and the segments between are even steps.
BODY_ORDER_TAIL = 0.1
# Tail-channel value of the segment the tail moves into next (the tail cell itself is 1.0).
TAIL_NEXT = 0.5
# Food-channel value of food outside a local view, drawn on the crop edge in its direction.
FOOD_PROJECTED = 0.5


@dataclass(frozen=True)
class ObservationSpec:
    """The observation switches the env reads; the defaults are today's 4-channel full grid.

    ``view_size`` is ``None`` for the full view and the crop's odd side for the local view.
    """

    body: str = "binary"
    tail: bool = False
    hunger: bool = False
    view_size: int | None = None

    def __post_init__(self) -> None:
        if self.body not in OBSERVATION_BODIES:
            raise ValueError(
                f"ObservationSpec body must be one of {', '.join(OBSERVATION_BODIES)}, "
                f"got {self.body!r}"
            )
        if self.view_size is not None and (self.view_size < 5 or self.view_size % 2 == 0):
            raise ValueError(
                "ObservationSpec view_size must be None or an odd number of at least 5, "
                f"got {self.view_size}"
            )

    @classmethod
    def from_config(cls, config: ObservationConfig) -> ObservationSpec:
        return cls(
            body=config.body,
            tail=config.tail,
            hunger=config.hunger,
            view_size=config.view_size if config.view == "local" else None,
        )

    @property
    def channels(self) -> tuple[str, ...]:
        body = "body_order" if self.body == "order" else "body"
        extra = (("tail",) if self.tail else ()) + (("hunger",) if self.hunger else ())
        return (body, "head", "food", "wall", *extra)

    @property
    def is_binary(self) -> bool:
        """True when every value is 0 or 1, so the rollout buffer may store ``uint8``."""
        return (
            self.body == "binary" and not self.tail and not self.hunger and self.view_size is None
        )

    def size(self, board_size: int) -> int:
        """The observation's spatial side on a ``board_size`` board."""
        return board_size if self.view_size is None else self.view_size

    def record(self, board_size: int) -> dict[str, Any]:
        """What ``run.json`` and every checkpoint store about the observation."""
        return {
            "channels": list(self.channels),
            "size": self.size(board_size),
            "view": "full" if self.view_size is None else "local",
        }


def describe_record(record: Mapping[str, Any]) -> str:
    """One readable line for an observation record, as error messages quote it."""
    size = record["size"]
    return f"channels {', '.join(record['channels'])} on a {size}x{size} {record['view']} view"


def body_order_grid(
    body_positions: torch.Tensor, snake_len: torch.Tensor, board_size: int
) -> torch.Tensor:
    """(N, S, S) float32: segment ``i`` of a length-``L`` snake gets ``1 - 0.9 * i / (L - 1)``.

    The head is 1.0 and the tail :data:`BODY_ORDER_TAIL`; a length-1 snake is 1.0, and cells off
    the body are 0. ``body_positions`` is (N, M, 2), head first.
    """
    num_envs, max_len = body_positions.shape[:2]
    index = torch.arange(max_len, device=body_positions.device)
    length = snake_len.unsqueeze(1)
    on_body = index.unsqueeze(0) < length
    order = 1 - (1 - BODY_ORDER_TAIL) * index.unsqueeze(0) / (length - 1).clamp(min=1)
    env_ids = torch.arange(num_envs, device=body_positions.device).unsqueeze(1).expand_as(on_body)
    grid = torch.zeros((num_envs, board_size, board_size), device=body_positions.device)
    grid[env_ids[on_body], body_positions[..., 0][on_body], body_positions[..., 1][on_body]] = (
        order[on_body].float()
    )
    return grid


def tail_grid(
    body_positions: torch.Tensor, snake_len: torch.Tensor, board_size: int
) -> torch.Tensor:
    """(N, S, S) float32: the tail cell 1.0 and the segment before it :data:`TAIL_NEXT`.

    The next segment is written first, so a length-1 snake (whose next segment is its tail) shows
    1.0, and a length-2 snake marks its head :data:`TAIL_NEXT`.
    """
    num_envs = body_positions.shape[0]
    env_ids = torch.arange(num_envs, device=body_positions.device)
    tail = body_positions[env_ids, snake_len - 1]
    following = body_positions[env_ids, (snake_len - 2).clamp(min=0)]
    grid = torch.zeros((num_envs, board_size, board_size), device=body_positions.device)
    grid[env_ids, following[:, 0], following[:, 1]] = TAIL_NEXT
    grid[env_ids, tail[:, 0], tail[:, 1]] = 1.0
    return grid


def local_view(
    grid: torch.Tensor, head: torch.Tensor, food: torch.Tensor, view_size: int
) -> torch.Tensor:
    """Crop (N, C, S, S) to (N, C, V, V) centred on each head; channel 2 is food, 3 is wall.

    Cells off the board are wall (1.0 in channel 3, 0 elsewhere). Food inside the crop keeps its
    1.0; food outside is drawn at :data:`FOOD_PROJECTED` on the edge cell where the line from the
    head leaves the crop: with ``d = food - head``, ``r = V // 2`` and ``m = max(|d|)``, each
    offset is ``d * r / m`` rounded half away from zero, in exact integer arithmetic.
    """
    num_envs, channels, size = grid.shape[0], grid.shape[1], grid.shape[2]
    radius = view_size // 2
    padded = grid.new_zeros((num_envs, channels, size + 2 * radius, size + 2 * radius))
    padded[:, 3] = 1.0
    padded[:, :, radius : radius + size, radius : radius + size] = grid
    # the head sits at padded index head + radius, so its crop starts at padded index head
    offsets = torch.arange(view_size, device=grid.device)
    rows = (head[:, 0:1] + offsets).view(num_envs, view_size, 1)
    cols = (head[:, 1:2] + offsets).view(num_envs, 1, view_size)
    env_ids = torch.arange(num_envs, device=grid.device).view(num_envs, 1, 1)
    crop = padded.permute(0, 2, 3, 1)[env_ids, rows, cols].permute(0, 3, 1, 2).contiguous()

    delta = food - head
    inside = (delta.abs() <= radius).all(dim=1)
    reach = delta.abs().amax(dim=1, keepdim=True).clamp(min=1)
    projected = delta.sign() * ((2 * delta.abs() * radius + reach) // (2 * reach))
    cell = torch.where(inside.unsqueeze(1), delta, projected) + radius
    value = torch.where(inside, 1.0, FOOD_PROJECTED).to(crop.dtype)
    crop[torch.arange(num_envs, device=grid.device), 2, cell[:, 0], cell[:, 1]] = value
    return crop
