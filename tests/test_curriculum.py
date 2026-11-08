from pathlib import Path

import pytest
import torch

from snakeppo.config import CurriculumConfig, load_config
from snakeppo.curriculum import PROMOTED_FORCED, PROMOTED_GATE, PROMOTED_NONE, Curriculum
from snakeppo.ppo import trainer
from snakeppo.ppo.trainer import resume, train
from snakeppo.runs import load_checkpoint, read_metrics

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
# smoke.yaml: 8x8 board, 4 updates, eval and checkpoint at 2 and 4. Stage 0 is a 6x6 arena with
# goal 4; promotion is forced after 2 updates at the latest, so updates 3 and 4 run on the full
# board with no goal.
CURRICULUM = (
    "curriculum.grow=both",
    "curriculum.stages=2",
    "curriculum.start_board=6",
    "curriculum.min_updates=1",
    "curriculum.max_updates=2",
)


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


def _curriculum_run(runs_dir, *overrides):
    config = load_config(SMOKE, [f"train.runs_dir={runs_dir.as_posix()}", *CURRICULUM, *overrides])
    return train(config, progress=False)


def _curve(run_dir):
    timing = ("wall_time", "train/fps")
    return [{k: v for k, v in line.items() if k not in timing} for line in read_metrics(run_dir)]


def test_a_curriculum_run_logs_its_stages_and_ends_on_the_full_board(tmp_path):
    lines = read_metrics(_curriculum_run(tmp_path))
    assert lines[0]["curriculum/stage"] == 0
    assert (lines[0]["curriculum/board_size"], lines[0]["curriculum/goal_length"]) == (6, 4)
    assert [line["curriculum/promoted"] for line in lines].count(PROMOTED_NONE) == 3
    assert lines[-1]["curriculum/stage"] == 1
    assert (lines[-1]["curriculum/board_size"], lines[-1]["curriculum/goal_length"]) == (8, 0)
    assert all("eval/score_mean" in line for line in lines[1::2])


def test_promotion_is_applied_before_the_same_updates_checkpoint(tmp_path, monkeypatch):
    saved = []
    real_save = trainer.save_checkpoint

    def save(path, **state):
        saved.append((path.name, state["update"], state["curriculum"]["stage"]))
        real_save(path, **state)

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    run_dir = _curriculum_run(tmp_path)
    promoted_at = next(
        line["update"] for line in read_metrics(run_dir) if line["curriculum/promoted"]
    )
    for name, update, stage in saved:
        assert stage == (1 if update >= promoted_at else 0), (name, update)
    last = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert last["curriculum"]["stage"] == 1


def test_a_resumed_curriculum_run_matches_an_uninterrupted_one_on_cpu(tmp_path, monkeypatch):
    reference = _curriculum_run(tmp_path / "reference")
    real_log = trainer.MetricsLogger.log

    def log(self, update, env_steps, metrics):
        real_log(self, update, env_steps, metrics)
        if update == 3:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(trainer.MetricsLogger, "log", log)
        with pytest.raises(KeyboardInterrupt):
            _curriculum_run(tmp_path / "cut")
    (run_dir,) = (tmp_path / "cut").iterdir()
    resume(run_dir, progress=False)
    assert _curve(run_dir) == _curve(reference)
