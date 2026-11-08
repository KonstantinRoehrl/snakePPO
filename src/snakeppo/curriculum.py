"""Staged curriculum: which stage a run trains in, and when it is promoted to the next one.

Pure bookkeeping; the trainer applies each stage to the env between updates. A stage is left
when the mean score of its last ``promote_window`` finished episodes reaches
``promote_fraction`` of the stage's capacity, never before ``min_updates`` updates in the stage,
and always after ``max_updates``. The final stage (full board, no goal) is never left.
"""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from snakeppo.config import CurriculumConfig, CurriculumStage, curriculum_ladder

PROMOTED_NONE, PROMOTED_GATE, PROMOTED_FORCED = range(3)


@dataclass(frozen=True)
class UpdateResult:
    promoted: int  # a PROMOTED_* code
    metrics: dict[str, float]  # the update's curriculum/* metrics


class Curriculum:
    def __init__(self, config: CurriculumConfig, board_size: int) -> None:
        self.config = config
        self.stages = curriculum_ladder(config, board_size)
        self.stage = 0
        self.updates_in_stage = 0
        self._window: deque[float] = deque(maxlen=config.promote_window)

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.stage]

    @property
    def is_final(self) -> bool:
        return self.stage == len(self.stages) - 1

    @property
    def window_score(self) -> float | None:
        """Mean score of the window once it holds ``promote_window`` episodes, else ``None``."""
        if len(self._window) < self.config.promote_window:
            return None
        return statistics.fmean(self._window)

    def end_update(self, scores: Sequence[float]) -> UpdateResult:
        """Add one update's finished-episode scores, then decide whether to promote.

        The metrics describe the stage the update ran in, with the window score after
        ``scores`` were added; on promotion the window starts empty in the next stage.
        """
        self._window.extend(scores)
        self.updates_in_stage += 1
        stage, window_score = self.current, self.window_score
        metrics = {
            "curriculum/stage": float(self.stage),
            "curriculum/board_size": float(stage.arena),
            "curriculum/goal_length": float(stage.goal or 0),
        }
        if window_score is not None:
            metrics["curriculum/window_score"] = window_score
        promoted = self._promotion(window_score)
        if promoted != PROMOTED_NONE:
            self.stage += 1
            self.updates_in_stage = 0
            self._window.clear()
        metrics["curriculum/promoted"] = float(promoted)
        return UpdateResult(promoted, metrics)

    def _promotion(self, window_score: float | None) -> int:
        if self.is_final:
            return PROMOTED_NONE
        threshold = self.config.promote_fraction * self.current.capacity
        if (
            self.updates_in_stage >= self.config.min_updates
            and window_score is not None
            and window_score >= threshold
        ):
            return PROMOTED_GATE
        if self.updates_in_stage >= self.config.max_updates:
            return PROMOTED_FORCED
        return PROMOTED_NONE

    def state_dict(self) -> dict[str, Any]:
        """Primitives only, so a checkpoint holding it loads with ``weights_only=True``."""
        return {
            "stage": self.stage,
            "updates_in_stage": self.updates_in_stage,
            "window": list(self._window),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        stage = int(state["stage"])
        if not 0 <= stage < len(self.stages):
            raise ValueError(
                f"curriculum stage {stage} is outside this {len(self.stages)}-stage ladder"
            )
        self.stage = stage
        self.updates_in_stage = int(state["updates_in_stage"])
        self._window = deque((float(s) for s in state["window"]), maxlen=self.config.promote_window)
