import sys
from datetime import datetime
from pathlib import Path

import pytest

from snakeppo import sweep
from snakeppo.cli import main
from snakeppo.runs import find_runs, load_run

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
