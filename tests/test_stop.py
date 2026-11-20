import io
import os
import re
import sys
from pathlib import Path

import pytest

from snakeppo.cli import main
from snakeppo.config import load_config
from snakeppo.ppo import trainer
from snakeppo.ppo.trainer import ResumeError, StopRequested, resume, train
from snakeppo.processes import (
    LAUNCH_ENV,
    STOP_REQUEST_FILE,
    is_alive,
    process_record,
    request_stop,
)
from snakeppo.runs import load_checkpoint, load_run, read_metrics, read_run_info, update_run_info

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
TIMING_KEYS = ("wall_time", "train/fps")


def _smoke_config(runs_dir):
    return load_config(SMOKE, [f"train.runs_dir={runs_dir.as_posix()}"])


def _curve(run_dir):
    return [
        {k: v for k, v in line.items() if k not in TIMING_KEYS} for line in read_metrics(run_dir)
    ]


def _after_update(monkeypatch, runs_dir, update, action):
    """Call ``action(run_dir)`` on the one run in ``runs_dir`` right after it logs ``update``."""
    real_log = trainer.MetricsLogger.log

    def log(self, number, env_steps, metrics):
        real_log(self, number, env_steps, metrics)
        if number == update:
            (run_dir,) = runs_dir.iterdir()
            action(run_dir)

    monkeypatch.setattr(trainer.MetricsLogger, "log", log)


def _interrupted_run(runs_dir, monkeypatch):
    """A smoke run (4 updates, evals and checkpoints at 2 and 4) interrupted after update 3."""

    def ctrl_c(run_dir):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        _after_update(patch, runs_dir, 3, ctrl_c)
        with pytest.raises(KeyboardInterrupt):
            train(_smoke_config(runs_dir), progress=False)
    (run_dir,) = runs_dir.iterdir()
    return run_dir


def test_a_stop_request_ends_the_run_like_ctrl_c_and_resume_matches_an_uninterrupted_run(
    tmp_path, monkeypatch, capsys
):
    reference = train(_smoke_config(tmp_path / "reference"), progress=False)
    with monkeypatch.context() as patch:
        _after_update(patch, tmp_path / "cut", 3, request_stop)
        with pytest.raises(StopRequested):
            train(_smoke_config(tmp_path / "cut"), progress=False)
    (run_dir,) = (tmp_path / "cut").iterdir()
    info = read_run_info(run_dir)
    assert info["status"] == "interrupted" and info["process"] is None
    assert not (run_dir / STOP_REQUEST_FILE).exists()
    assert f"Continue with: snakeppo train --resume {run_dir}" in capsys.readouterr().out
    assert [line["update"] for line in read_metrics(run_dir)] == [1, 2, 3]
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["update"] == 3
    assert load_checkpoint(run_dir / "checkpoints" / "best.pt")["update"] == 2

    resume(run_dir, progress=False)
    assert _curve(run_dir) == _curve(reference)
    info = read_run_info(run_dir)
    assert info["status"] == "finished" and info["process"] is None


def test_cli_train_exits_130_on_a_stop_request(tmp_path, monkeypatch, capsys):
    _after_update(monkeypatch, tmp_path, 3, request_stop)
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main(args) == 130
    assert "snakeppo train: interrupted" in capsys.readouterr().err
    (run_dir,) = tmp_path.iterdir()
    assert read_run_info(run_dir)["status"] == "interrupted"


def test_a_stop_request_at_the_last_update_lets_the_run_finish(tmp_path, monkeypatch):
    _after_update(monkeypatch, tmp_path, 4, request_stop)
    run_dir = train(_smoke_config(tmp_path), progress=False)
    assert read_run_info(run_dir)["status"] == "finished"
    assert not (run_dir / STOP_REQUEST_FILE).exists()


@pytest.mark.parametrize(("value", "launch"), [("20261119-abc", "20261119-abc"), ("", None)])
def test_run_json_records_the_training_process_and_launch_while_it_runs(
    tmp_path, monkeypatch, value, launch
):
    monkeypatch.setenv(LAUNCH_ENV, value)
    seen = []
    _after_update(monkeypatch, tmp_path, 1, lambda run_dir: seen.append(read_run_info(run_dir)))
    run_dir = train(_smoke_config(tmp_path), progress=False)
    (during,) = seen
    assert is_alive(during["process"]) is True and during["process"]["pid"] == os.getpid()
    assert during["launch"] == launch
    info = read_run_info(run_dir)
    assert info["process"] is None and info["launch"] == launch


def test_run_json_has_no_launch_without_the_launch_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(LAUNCH_ENV, raising=False)
    run_dir = train(_smoke_config(tmp_path), progress=False)
    assert read_run_info(run_dir)["launch"] is None


class _Boom(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "status"), [(KeyboardInterrupt(), "interrupted"), (_Boom(), "failed")]
)
def test_every_aborted_run_clears_its_process_record(tmp_path, monkeypatch, error, status):
    def fail(run_dir):
        request_stop(run_dir)  # left behind by a stop that came too late
        raise error

    _after_update(monkeypatch, tmp_path, 3, fail)
    with pytest.raises(type(error)):
        train(_smoke_config(tmp_path), progress=False)
    (run_dir,) = tmp_path.iterdir()
    info = read_run_info(run_dir)
    assert info["status"] == status and info["process"] is None
    assert not (run_dir / STOP_REQUEST_FILE).exists()


def test_resume_clears_a_leftover_stop_request_records_its_process_and_keeps_the_launch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(LAUNCH_ENV, "20261119-abc")
    run_dir = _interrupted_run(tmp_path, monkeypatch)
    monkeypatch.delenv(LAUNCH_ENV)
    request_stop(run_dir)
    seen = []
    _after_update(monkeypatch, tmp_path, 3, lambda folder: seen.append(read_run_info(folder)))
    resume(run_dir, progress=False)
    (during,) = seen
    assert is_alive(during["process"]) is True and during["process"]["pid"] == os.getpid()
    assert during["launch"] == "20261119-abc"
    info = read_run_info(run_dir)
    assert info["status"] == "finished" and info["process"] is None
    assert info["launch"] == "20261119-abc"


def test_resume_refuses_a_run_whose_process_is_alive(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch)
    update_run_info(run_dir, status="running", process=process_record())
    message = f"is still being trained by process {os.getpid()}; stop it before resuming"
    with pytest.raises(ResumeError, match=re.escape(message)):
        resume(run_dir, progress=False)


def test_resume_accepts_a_running_run_whose_process_is_gone_at_once(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch)
    record = process_record()
    gone = {**record, "started": record["started"] - 3600}  # the pid now belongs to another process
    update_run_info(run_dir, status="running", process=gone)
    assert load_run(run_dir).status == "stalled"  # metrics were written just now
    resume(run_dir, progress=False)
    assert read_run_info(run_dir)["status"] == "finished"


class _Stream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.mark.parametrize(
    ("progress", "tty", "disabled"), [(True, True, False), (True, False, True), (False, True, True)]
)
def test_the_progress_bar_shows_only_on_a_terminal(tmp_path, monkeypatch, progress, tty, disabled):
    seen = []
    real_tqdm = trainer.tqdm

    def tqdm(*args, **kwargs):
        seen.append(kwargs["disable"])
        return real_tqdm(*args, **kwargs)

    monkeypatch.setattr(trainer, "tqdm", tqdm)
    monkeypatch.setattr(sys, "stderr", _Stream(tty))
    train(_smoke_config(tmp_path), progress=progress)
    assert seen == [disabled]
