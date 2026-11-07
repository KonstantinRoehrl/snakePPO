import json
from datetime import datetime

import pytest
import torch
import yaml

from snakeppo.runs import (
    SCHEMA_VERSION,
    MetricsLogger,
    create_run_dir,
    find_runs,
    load_checkpoint,
    read_metrics,
    resolve_checkpoint,
    save_checkpoint,
    summarize_run,
    write_run_info,
)

NOW = datetime(2026, 11, 7, 9, 30, 0)


def _fake_run(tmp_path, name="demo", seed=3, evals=(1.0, 4.0, 2.0)):
    run_dir = create_run_dir(tmp_path, name, seed, now=NOW)
    write_run_info(run_dir, name=name, agent="cnn", param_count=10, device="cpu")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"env": {"seed": seed}}), "utf-8")
    with MetricsLogger(run_dir) as logger:
        for update, score in enumerate(evals, start=1):
            logger.log(update, update * 100, {"loss/total": 0.5, "eval/score_mean": score})
    return run_dir


def test_run_dir_name_is_unique_and_has_checkpoint_folder(tmp_path):
    first = create_run_dir(tmp_path, "demo", 3, now=NOW)
    second = create_run_dir(tmp_path, "demo", 3, now=NOW)
    assert first.name == "2026-11-07_093000_demo_seed3"
    assert second.name == "2026-11-07_093000_demo_seed3_2"
    assert (first / "checkpoints").is_dir()


def test_run_info_records_schema_and_identity(tmp_path):
    run_dir = create_run_dir(tmp_path, "demo", 3, now=NOW)
    write_run_info(run_dir, name="demo", agent="resnet", param_count=123, device="cpu")
    info = json.loads((run_dir / "run.json").read_text("utf-8"))
    assert info["schema_version"] == SCHEMA_VERSION == 1
    assert {"name", "created", "git_sha", "agent", "param_count", "device"} <= info.keys()
    assert info["agent"] == "resnet" and info["param_count"] == 123


def test_metrics_lines_follow_the_contract_and_tensorboard_is_written(tmp_path):
    run_dir = _fake_run(tmp_path)
    lines = read_metrics(run_dir)
    assert [line["update"] for line in lines] == [1, 2, 3]
    assert lines[0]["env_steps"] == 100
    assert {"update", "env_steps", "wall_time", "loss/total", "eval/score_mean"} == set(lines[0])
    assert any((run_dir / "tb").iterdir())


def test_checkpoint_round_trip_and_resolution(tmp_path):
    run_dir = create_run_dir(tmp_path, "demo", 0, now=NOW)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = run_dir / "checkpoints" / "best.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, update=5, env_steps=640, config={"a": 1}
    )
    loaded = load_checkpoint(path)
    assert loaded["update"] == 5 and loaded["env_steps"] == 640 and loaded["config"] == {"a": 1}
    assert torch.equal(loaded["model"]["weight"], model.weight)
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
