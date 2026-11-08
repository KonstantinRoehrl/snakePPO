import pytest
import torch

from snakeppo.config import CurriculumConfig
from snakeppo.curriculum import PROMOTED_FORCED, PROMOTED_GATE, PROMOTED_NONE, Curriculum


def _curriculum(**fields):
    # board 16, both: stage 0 is an 8x8 arena with goal 4, capacity 3, gate at 0.3 * 3 = 0.9
    defaults = {"grow": "both", "promote_window": 4, "min_updates": 2, "max_updates": 10}
    return Curriculum(CurriculumConfig(**(defaults | fields)), board_size=16)


def test_gate_promotes_once_the_window_mean_reaches_its_share_of_capacity():
    curriculum = _curriculum()
    assert curriculum.end_update([1, 1]).promoted == PROMOTED_NONE  # window not full
    result = curriculum.end_update([0.5, 1.5])
    assert result.promoted == PROMOTED_GATE
    assert result.metrics == {
        "curriculum/stage": 0.0,
        "curriculum/board_size": 8.0,
        "curriculum/goal_length": 4.0,
        "curriculum/window_score": 1.0,
        "curriculum/promoted": 1.0,
    }
    assert (curriculum.stage, curriculum.current.arena, curriculum.current.goal) == (1, 10, 8)
    assert curriculum.updates_in_stage == 0 and curriculum.window_score is None


def test_gate_waits_below_the_threshold_and_for_a_full_window():
    curriculum = _curriculum()
    curriculum.end_update([0.5, 1, 1, 1])  # mean 0.875 < 0.9
    assert curriculum.end_update([]).promoted == PROMOTED_NONE
    curriculum = _curriculum(promote_window=5)
    curriculum.end_update([3, 3, 3])
    result = curriculum.end_update([3])
    assert result.promoted == PROMOTED_NONE and "curriculum/window_score" not in result.metrics


def test_gate_never_fires_before_min_updates():
    curriculum = _curriculum(min_updates=3)
    assert curriculum.end_update([3, 3, 3, 3]).promoted == PROMOTED_NONE
    assert curriculum.end_update([]).promoted == PROMOTED_NONE
    assert curriculum.end_update([]).promoted == PROMOTED_GATE


def test_promotion_is_forced_at_max_updates():
    curriculum = _curriculum(max_updates=3)
    codes = [curriculum.end_update([0]).promoted for _ in range(3)]
    assert codes == [PROMOTED_NONE, PROMOTED_NONE, PROMOTED_FORCED]
    assert curriculum.stage == 1 and curriculum.window_score is None


def test_the_window_keeps_only_the_latest_episodes():
    curriculum = _curriculum(min_updates=1)
    curriculum.end_update([3, 3])
    result = curriculum.end_update([0, 0, 0, 1])  # all six average 7/6, the last four 0.25
    assert result.promoted == PROMOTED_NONE
    assert result.metrics["curriculum/window_score"] == 0.25


def test_the_final_stage_is_never_left():
    curriculum = _curriculum(stages=2, max_updates=1)
    assert curriculum.end_update([]).promoted == PROMOTED_FORCED
    assert curriculum.is_final and (curriculum.current.arena, curriculum.current.goal) == (16, None)
    for _ in range(5):
        result = curriculum.end_update([196] * 4)
        assert result.promoted == PROMOTED_NONE
    assert result.metrics["curriculum/goal_length"] == 0.0
    assert result.metrics["curriculum/board_size"] == 16.0


def test_state_round_trips_through_a_weights_only_checkpoint(tmp_path):
    curriculum = _curriculum(max_updates=2)
    curriculum.end_update([0])
    curriculum.end_update([0])  # forced to stage 1
    curriculum.end_update([0.5, 2.5])
    path = tmp_path / "state.pt"
    torch.save({"curriculum": curriculum.state_dict()}, path)

    restored = _curriculum(max_updates=2)
    restored.load_state_dict(torch.load(path, weights_only=True)["curriculum"])
    assert restored.state_dict() == {"stage": 1, "updates_in_stage": 1, "window": [0.5, 2.5]}
    assert restored.end_update([9]) == curriculum.end_update([9])
    with pytest.raises(ValueError, match="curriculum stage 7 is outside this 5-stage ladder"):
        restored.load_state_dict({"stage": 7, "updates_in_stage": 0, "window": []})
