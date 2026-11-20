import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from snakeppo import sweep
from snakeppo.cli import main
from snakeppo.processes import LAUNCH_ENV, STOP_REQUEST_FILE, child_creationflags, launch_dir
from snakeppo.runs import METRICS_FILE, find_runs, load_run, update_run_info

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def _args(runs_dir, *extra):
    return [
        "sweep",
        "--config",
        str(SMOKE),
        "--set",
        f"train.runs_dir={runs_dir.as_posix()}",
        *extra,
    ]


def test_group_id_is_name_at_start_time():
    assert sweep.group_id("baseline", datetime(2026, 11, 7, 9, 5, 3)) == "baseline@20261107-090503"


def test_child_command_runs_train_through_this_interpreter():
    command = sweep.child_command(3, "g@1", ["--config", "x.yaml"])
    assert command[:4] == [sys.executable, "-m", "snakeppo.cli", "train"]
    assert command[4:] == ["--config", "x.yaml", "--seed", "3", "--group", "g@1"]


def test_sweep_trains_every_seed_into_one_group(tmp_path, capsys):
    assert main(_args(tmp_path, "--seeds", "1,2", "--name", "sw")) == 0
    records = [load_run(run) for run in find_runs([tmp_path])]
    assert sorted(r.seed for r in records) == [1, 2]
    (group,) = {r.group for r in records}
    assert group.startswith("sw@") and all(r.status == "finished" for r in records)
    out = capsys.readouterr().out
    assert f"Sweep {group}: 2 of 2 seeds trained successfully." in out
    assert "seed 1: finished" in out and "seed 2: finished" in out


def test_a_failing_seed_does_not_stop_the_others(tmp_path, monkeypatch, capsys):
    real_command = sweep.child_command

    def command(seed, group, train_args):
        if seed == 2:
            return [sys.executable, "-c", "import sys; print('seed two broke'); sys.exit(3)"]
        return real_command(seed, group, train_args)

    monkeypatch.setattr(sweep, "child_command", command)
    assert main(_args(tmp_path, "--seeds", "1,2", "--jobs", "2")) == 1
    (record,) = [load_run(run) for run in find_runs([tmp_path])]
    assert record.seed == 1 and record.status == "finished"
    out = capsys.readouterr().out
    assert "seed 2: failed with exit code 3; last output:" in out
    assert "    seed two broke" in out
    assert "1 of 2 seeds trained successfully" in out
    assert "seed 2: no run folder, exit code 3  -" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--seeds", "1,1"], "--seeds lists a seed twice"),
        (["--seeds", "1", "--jobs", "0"], "--jobs must be at least 1"),
        (["--seeds", "1", "--agent", "nope"], "unknown agent 'nope'"),
        (["--seeds", "1", "--set", "ppo.nope=1"], "unknown key 'ppo.nope'"),
    ],
)
def test_bad_sweep_options_fail_before_any_training(tmp_path, capsys, extra, message):
    assert main(_args(tmp_path, *extra)) == 2
    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_resume_command_continues_the_folder_through_this_interpreter(tmp_path):
    run_dir = tmp_path / "a-run"
    assert sweep.resume_command(run_dir) == [
        sys.executable,
        "-m",
        "snakeppo.cli",
        "train",
        "--resume",
        str(run_dir),
    ]


def test_joining_a_group_continues_only_its_unfinished_seeds(tmp_path, monkeypatch, capsys):
    assert main(_args(tmp_path, "--seeds", "1,2,3", "--jobs", "3", "--group", "g@1")) == 0
    records = [load_run(run) for run in find_runs([tmp_path])]
    assert {r.group for r in records} == {"g@1"}
    folders = {r.seed: r.path for r in records}
    update_run_info(folders[2], status="interrupted")
    update_run_info(folders[3], status="running")
    an_hour_ago = time.time() - 3600
    os.utime(folders[3] / METRICS_FILE, (an_hour_ago, an_hour_ago))
    assert load_run(folders[3]).status == "stalled"
    capsys.readouterr()

    started = []
    finishes_at_once = [sys.executable, "-c", "pass"]

    def resume_command(run_dir):
        started.append(("resume", run_dir))
        return finishes_at_once

    def child_command(seed, group, train_args):
        started.append(("train", seed, group))
        return finishes_at_once

    monkeypatch.setattr(sweep, "resume_command", resume_command)
    monkeypatch.setattr(sweep, "child_command", child_command)
    assert main(_args(tmp_path, "--seeds", "1,2,3,4", "--group", "g@1")) == 0
    assert started == [("resume", folders[2]), ("resume", folders[3]), ("train", 4, "g@1")]
    out = capsys.readouterr().out
    assert "Sweep g@1: 1 of 4 seeds trained successfully." in out
    assert f"  seed 1: finished, already in the group  {folders[1]}" in out
    assert f"  seed 2: interrupted  {folders[2]}" in out
    assert "  seed 4: no run folder  -" in out


def test_a_stop_request_starts_no_further_seeds(tmp_path):
    stop_file = tmp_path / STOP_REQUEST_FILE

    def requests_a_stop(seed, group, train_args):
        return [sys.executable, "-c", f"open({str(stop_file)!r}, 'w').close()"]

    outcomes = sweep.run_sweep(
        [1, 2, 3], [], runs_dir=tmp_path, group="g@1", command=requests_a_stop, stop_file=stop_file
    )
    assert outcomes == [
        sweep.SeedOutcome(1, 0, None, "no run folder"),
        sweep.SeedOutcome(2, None, None, "skipped"),
        sweep.SeedOutcome(3, None, None, "skipped"),
    ]
    assert sweep.format_outcomes("g@1", outcomes).splitlines() == [
        "Sweep g@1: 0 of 3 seeds trained successfully.",
        "  seed 1: no run folder  -",
        "  seed 2: skipped  -",
        "  seed 3: skipped  -",
    ]


def test_a_launched_sweep_with_a_stop_request_exits_130(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(LAUNCH_ENV, "launch-1")
    stop_file = launch_dir(tmp_path, "launch-1") / STOP_REQUEST_FILE
    stop_file.parent.mkdir(parents=True)
    stop_file.write_text("{}", encoding="utf-8")
    assert main(_args(tmp_path, "--seeds", "1,2")) == 130
    out = capsys.readouterr().out
    assert "  seed 1: skipped  -" in out and "  seed 2: skipped  -" in out
    assert find_runs([tmp_path]) == []


class _FinishedProcess:
    returncode = 0

    def poll(self):
        return self.returncode


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_detached_sweep_starts_seeds_without_a_console_into_its_own_output(
    tmp_path, monkeypatch, jobs
):
    started = []

    def popen(argv, **kwargs):
        started.append(kwargs)
        return _FinishedProcess()

    monkeypatch.setattr(sweep.subprocess, "Popen", popen)
    sweep.run_sweep([1, 2], [], runs_dir=tmp_path, group="g@1", jobs=jobs, detached=True)
    detached = {"stdout": sys.stdout, "stderr": subprocess.STDOUT}
    assert started == [detached | {"creationflags": child_creationflags()}] * 2
