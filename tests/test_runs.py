import json
import math
import os
from datetime import datetime

import pytest
import torch
import yaml

from snakeppo.builders import capture_rng_state
from snakeppo.config import Config, apply_overrides, save_config
from snakeppo.runs import (
    SCHEMA_VERSION,
    MetricsLogger,
    create_run_dir,
    derive_status,
    find_runs,
    load_checkpoint,
    load_run,
    read_metrics,
    read_metrics_since,
    read_run_info,
    resolve_checkpoint,
    save_checkpoint,
    seconds_per_update,
    stall_window,
    summarize_run,
    total_updates,
    truncate_metrics,
    update_run_info,
    write_run_info,
)

NOW = datetime(2026, 11, 7, 9, 30, 0)
RECORD = {"channels": ["body", "head", "food", "wall"], "size": 16, "view": "full"}
RUN_INFO = {
    "agent": "cnn",
    "param_count": 10,
    "device": "cpu",
    "total_updates": 3,
    "group": None,
    "observation": RECORD,
}


def _fake_run(tmp_path, name="demo", seed=3, evals=(1.0, 4.0, 2.0)):
    run_dir = create_run_dir(tmp_path, name, seed, now=NOW)
    write_run_info(run_dir, name=name, **RUN_INFO)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"env": {"seed": seed}}), "utf-8")
    with MetricsLogger(run_dir) as logger:
        for update, score in enumerate(evals, start=1):
            logger.log(update, update * 100, {"loss/total": 0.5, "eval/score_mean": score})
    return run_dir


def _lines(*updates, gap=1.0):
    return [{"update": u, "env_steps": u * 10, "wall_time": u * gap} for u in updates]


def test_run_dir_name_is_unique_and_has_checkpoint_folder(tmp_path):
    first = create_run_dir(tmp_path, "demo", 3, now=NOW)
    second = create_run_dir(tmp_path, "demo", 3, now=NOW)
    assert first.name == "2026-11-07_093000_demo_seed3"
    assert second.name == "2026-11-07_093000_demo_seed3_2"
    assert (first / "checkpoints").is_dir()


def test_run_info_records_schema_identity_and_v2_fields(tmp_path):
    run_dir = create_run_dir(tmp_path, "demo", 3, now=NOW)
    local = {"channels": ["body", "head", "food", "wall", "hunger"], "size": 11, "view": "local"}
    info = write_run_info(
        run_dir,
        name="demo",
        agent="resnet",
        param_count=123,
        device="cpu",
        total_updates=40,
        group="demo@20261107-093000",
        observation=local,
    )
    assert json.loads((run_dir / "run.json").read_text("utf-8")) == info == read_run_info(run_dir)
    assert info["schema_version"] == SCHEMA_VERSION == 4
    assert info["observation"] == local
    assert {"name", "created", "git_sha", "agent", "param_count", "device"} <= info.keys()
    assert info["status"] == "running"
    assert info["total_updates"] == 40
    assert info["group"] == "demo@20261107-093000"
    assert info["resumed"] == []
    assert not list(run_dir.glob(".*.tmp"))


def test_update_run_info_rewrites_fields_and_rejects_derived_statuses(tmp_path):
    run_dir = _fake_run(tmp_path)
    updated = update_run_info(run_dir, status="finished", resumed=["2026-11-07T10:00:00+01:00"])
    assert read_run_info(run_dir) == updated
    assert updated["status"] == "finished" and updated["agent"] == "cnn"
    with pytest.raises(ValueError, match="status must be one of"):
        update_run_info(run_dir, status="stalled")


def test_total_updates_rounds_up_to_whole_rollouts():
    config = apply_overrides(
        Config(), ["train.total_env_steps=1000", "ppo.rollout_steps=16", "env.num_envs=8"]
    )
    assert total_updates(config) == 8  # 1000 / 128 = 7.8


def test_metrics_lines_follow_the_contract_and_tensorboard_is_written(tmp_path):
    run_dir = _fake_run(tmp_path)
    lines = read_metrics(run_dir)
    assert [line["update"] for line in lines] == [1, 2, 3]
    assert lines[0]["env_steps"] == 100
    assert {"update", "env_steps", "wall_time", "loss/total", "eval/score_mean"} == set(lines[0])
    assert any((run_dir / "tb").iterdir())


def test_metrics_logger_continues_the_wall_clock_after_an_offset(tmp_path):
    run_dir = create_run_dir(tmp_path, "demo", 0, now=NOW)
    with MetricsLogger(run_dir, tensorboard=False, wall_time_offset=100.0) as logger:
        assert logger.elapsed() >= 100.0
        logger.log(1, 10, {})
    assert read_metrics(run_dir)[0]["wall_time"] >= 100.0


def test_read_metrics_since_returns_only_new_complete_lines(tmp_path):
    run_dir = _fake_run(tmp_path, evals=(1.0, 2.0))
    first = read_metrics_since(run_dir, 0)
    assert [line["update"] for line in first.lines] == [1, 2] and not first.reset
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_lines(3)[0]) + "\n" + '{"update": 4, "env_')
    second = read_metrics_since(run_dir, first.offset)
    assert [line["update"] for line in second.lines] == [3]
    assert second.offset < (run_dir / "metrics.jsonl").stat().st_size  # partial line held back
    assert read_metrics_since(run_dir, second.offset).lines == []


def test_read_metrics_since_resets_after_the_file_was_rewritten(tmp_path):
    run_dir = _fake_run(tmp_path)
    end = read_metrics_since(run_dir, 0).offset
    truncate_metrics(run_dir, after_update=1)
    past_end = read_metrics_since(run_dir, end)
    assert past_end.reset and [line["update"] for line in past_end.lines] == [1]
    mid_line = read_metrics_since(run_dir, 3)
    assert mid_line.reset and [line["update"] for line in mid_line.lines] == [1]
    missing = read_metrics_since(tmp_path / "nowhere", 10)
    assert missing.lines == [] and missing.offset == 0 and missing.reset
    with pytest.raises(ValueError, match="offset"):
        read_metrics_since(run_dir, -1)


def test_truncate_metrics_drops_lines_after_the_update(tmp_path):
    run_dir = _fake_run(tmp_path)
    assert truncate_metrics(run_dir, after_update=2) == 1
    assert [line["update"] for line in read_metrics(run_dir)] == [1, 2]
    assert truncate_metrics(run_dir, after_update=2) == 0


def test_stall_window_is_five_median_gaps_but_at_least_a_minute():
    assert seconds_per_update(_lines(1)) is None
    assert seconds_per_update(_lines(1, 2, 3, gap=4.0)) == 4.0
    assert stall_window([]) == 60.0
    assert stall_window(_lines(1, 2, 3, gap=1.0)) == 60.0
    assert stall_window(_lines(*range(1, 30), gap=20.0)) == 100.0


@pytest.mark.parametrize(
    ("stored", "seconds_since_write", "expected"),
    [
        ("running", 10.0, "running"),
        ("running", 61.0, "stalled"),
        ("finished", 1e6, "finished"),
        ("interrupted", 1e6, "interrupted"),
        ("failed", 1e6, "failed"),
    ],
)
def test_derive_status_reports_stored_status_and_stalls(stored, seconds_since_write, expected):
    info = {"schema_version": 2, "status": stored}
    status = derive_status(info, Config(), _lines(1, 2), 1000.0, 1000.0 + seconds_since_write)
    assert status == expected


def test_schema_1_status_is_inferred_from_progress():
    config = apply_overrides(
        Config(), ["train.total_env_steps=40", "ppo.rollout_steps=1", "env.num_envs=10"]
    )
    info = {"schema_version": 1}
    assert derive_status(info, config, _lines(1, 2, 3, 4), 0.0, 1e9) == "finished"
    assert derive_status(info, config, _lines(1, 2), 0.0, 1e9) == "unknown"
    assert derive_status(info, config, [], 0.0, 1e9) == "unknown"


def test_load_run_reads_a_v2_run_and_a_v1_run(tmp_path):
    v2 = _fake_run(tmp_path / "a")
    record = load_run(v2)
    assert record.id == v2.name and record.seed == 3 and record.group is None
    assert record.status == "running" and record.update == 3 and record.env_steps == 300
    assert record.total_updates == 3
    assert load_run(v2, now=(v2 / "metrics.jsonl").stat().st_mtime + 120).status == "stalled"

    v1 = create_run_dir(tmp_path / "b", "old", 1, now=NOW)
    old_info = {"schema_version": 1, "name": "old", "agent": "cnn", "device": "cpu"}
    (v1 / "run.json").write_text(json.dumps(old_info), "utf-8")
    config = apply_overrides(
        Config(), ["train.total_env_steps=20", "ppo.rollout_steps=1", "env.num_envs=10"]
    )
    save_config(config, v1 / "config.yaml")
    with MetricsLogger(v1, tensorboard=False) as logger:
        logger.log(1, 10, {})
        logger.log(2, 20, {})
    old = load_run(v1)
    assert old.status == "finished" and old.group is None and old.total_updates == 2


def test_checkpoint_round_trip_and_resolution(tmp_path):
    run_dir = create_run_dir(tmp_path, "demo", 0, now=NOW)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = run_dir / "checkpoints" / "best.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        update=5,
        env_steps=640,
        config={"a": 1},
        best_eval=-math.inf,
        wall_time=12.5,
        rng=capture_rng_state(torch.device("cpu")),
        env={"head": torch.zeros(2, 2, dtype=torch.long)},
        observation=RECORD,
    )
    loaded = load_checkpoint(path)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["observation"] == RECORD and "curriculum" not in loaded
    assert loaded["update"] == 5 and loaded["env_steps"] == 640 and loaded["config"] == {"a": 1}
    assert loaded["best_eval"] == -math.inf and loaded["wall_time"] == 12.5
    assert loaded["rng"]["cuda"] is None and torch.equal(loaded["env"]["head"], torch.zeros(2, 2))
    assert torch.equal(loaded["model"]["weight"], model.weight)
    assert not list(path.parent.glob(".*.tmp"))
    assert resolve_checkpoint(run_dir) == (run_dir, path)
    assert resolve_checkpoint(path) == (run_dir, path)
    with pytest.raises(FileNotFoundError, match="last"):
        resolve_checkpoint(run_dir, "last")


def test_summaries_find_runs_and_report_best_and_last_eval(tmp_path):
    run_dir = _fake_run(tmp_path)
    assert find_runs([tmp_path]) == [run_dir] == find_runs([run_dir])
    row = summarize_run(run_dir)
    assert row == {
        "run": run_dir.name,
        "agent": "cnn",
        "seed": 3,
        "updates": 3,
        "env_steps": 300,
        "best_eval": 4.0,
        "last_eval": 2.0,
    }


def test_atomic_writes_leave_no_temp_files(tmp_path):
    run_dir = _fake_run(tmp_path)
    update_run_info(run_dir, status="finished")
    truncate_metrics(run_dir, after_update=1)
    assert sorted(p.name for p in run_dir.iterdir() if p.name.startswith(".")) == []
    assert os.path.getsize(run_dir / "metrics.jsonl") > 0
