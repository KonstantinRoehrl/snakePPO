import json
import os
import time
from pathlib import Path

import pytest
import torch

from snakeppo.cli import main
from snakeppo.config import load_config
from snakeppo.ppo import trainer
from snakeppo.ppo.trainer import ResumeError, resume, train
from snakeppo.runs import load_checkpoint, read_metrics, read_run_info, update_run_info

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
TIMING_KEYS = ("wall_time", "train/fps")


def _smoke_config(runs_dir, *overrides):
    return load_config(SMOKE, [f"train.runs_dir={runs_dir.as_posix()}", *overrides])


def _curve(run_dir):
    return [
        {k: v for k, v in line.items() if k not in TIMING_KEYS} for line in read_metrics(run_dir)
    ]


def _interrupted_run(runs_dir, monkeypatch, after_update):
    """A smoke run (4 updates, evals and checkpoints at 2 and 4) stopped by Ctrl+C."""
    real_log = trainer.MetricsLogger.log

    def log(self, update, env_steps, metrics):
        real_log(self, update, env_steps, metrics)
        if update == after_update:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(trainer.MetricsLogger, "log", log)
        with pytest.raises(KeyboardInterrupt):
            train(_smoke_config(runs_dir), progress=False)
    (run_dir,) = runs_dir.iterdir()
    return run_dir


def test_resumed_run_matches_an_uninterrupted_one_on_cpu(tmp_path, monkeypatch):
    reference = train(_smoke_config(tmp_path / "reference"), progress=False)
    run_dir = _interrupted_run(tmp_path / "cut", monkeypatch, after_update=3)
    assert read_run_info(run_dir)["status"] == "interrupted"
    assert [line["update"] for line in read_metrics(run_dir)] == [1, 2, 3]

    assert resume(run_dir, progress=False) == run_dir
    assert _curve(run_dir) == _curve(reference)
    wall_times = [line["wall_time"] for line in read_metrics(run_dir)]
    assert wall_times == sorted(wall_times)
    info = read_run_info(run_dir)
    assert info["status"] == "finished" and len(info["resumed"]) == 1
    for name in ("last", "best"):
        ours = load_checkpoint(run_dir / "checkpoints" / f"{name}.pt")["model"]
        theirs = load_checkpoint(reference / "checkpoints" / f"{name}.pt")["model"]
        assert all(torch.equal(ours[key], theirs[key]) for key in theirs)


def test_resume_refuses_a_finished_run(tmp_path):
    run_dir = train(_smoke_config(tmp_path), progress=False)
    with pytest.raises(ResumeError, match="already finished all 4 updates"):
        resume(run_dir, progress=False)


def test_resume_refuses_a_run_without_last_checkpoint(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch, after_update=1)
    with pytest.raises(ResumeError, match="no checkpoints/last.pt"):
        resume(run_dir, progress=False)


def test_resume_refuses_a_schema_1_run(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch, after_update=3)
    info = read_run_info(run_dir)
    info["schema_version"] = 1
    (run_dir / "run.json").write_text(json.dumps(info), "utf-8")
    with pytest.raises(ResumeError, match="written before resume support"):
        resume(run_dir, progress=False)


def test_resume_refuses_a_run_another_process_is_still_training(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch, after_update=3)
    update_run_info(run_dir, status="running")
    with pytest.raises(ResumeError, match="still being trained"):
        resume(run_dir, progress=False)
    stale = time.time() - 120
    os.utime(run_dir / "metrics.jsonl", (stale, stale))
    resume(run_dir, progress=False)  # stalled now, so it continues
    assert read_run_info(run_dir)["status"] == "finished"


def test_resume_refuses_a_different_device_type(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch, after_update=3)
    update_run_info(run_dir, device="cuda")
    with pytest.raises(ResumeError, match="trained on cuda but would resume on cpu"):
        resume(run_dir, progress=False)


def test_resume_refuses_a_folder_that_is_not_a_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a run folder"):
        resume(tmp_path, progress=False)


def test_cli_resume_takes_no_other_option_and_reports_refusals(tmp_path, capsys):
    run_dir = train(_smoke_config(tmp_path), progress=False)
    capsys.readouterr()
    assert main(["train", "--resume", str(run_dir), "--seed", "0", "--set", "env.seed=1"]) == 2
    err = capsys.readouterr().err
    assert "--resume continues a run exactly and cannot be combined with --set, --seed" in err
    assert main(["train", "--resume", str(run_dir)]) == 2
    assert "already finished" in capsys.readouterr().err


def test_cli_group_flag_is_recorded_as_text(tmp_path):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main([*args, "--group", "2"]) == 0
    (run_dir,) = tmp_path.iterdir()
    assert read_run_info(run_dir)["group"] == "2"


def test_cli_ctrl_c_exits_130(tmp_path, monkeypatch, capsys):
    def interrupted(config, progress=True):
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "train", interrupted)
    assert main(["train", "--config", str(SMOKE)]) == 130
    assert "snakeppo train: interrupted" in capsys.readouterr().err
