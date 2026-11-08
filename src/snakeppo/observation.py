"""What the policy sees: the observation switches as one :class:`ObservationSpec`.

The channel order is fixed: the body (``body`` or ``body_order``), ``head``, ``food`` and
``wall``, then ``tail`` and ``hunger`` when they are on. The full view is the whole board; the
local view is a head-centred square of ``view_size`` cells, so the network's input side is
:meth:`ObservationSpec.size`. A run records its observation as :meth:`ObservationSpec.record`,
and a checkpoint is only ever loaded against an equal record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from snakeppo.config import OBSERVATION_BODIES, ObservationConfig

# Body-order value of the tail segment; the head is 1.0 and the segments between are even steps.
BODY_ORDER_TAIL = 0.1


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
