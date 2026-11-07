import json
import os
from pathlib import Path

import pytest
import torch

from snakeppo.agents import build_agent
from snakeppo.config import EnvConfig, RewardConfig, apply_overrides, load_config
from snakeppo.evaluation import evaluate
from snakeppo.ppo.trainer import train
from snakeppo.runs import load_checkpoint

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def _smoke_config(tmp_path, *overrides):
    return load_config(SMOKE, [f"train.runs_dir={tmp_path.as_posix()}", *overrides])


def test_evaluate_counts_one_finished_episode_per_game_and_is_reproducible():
    torch.manual_seed(0)
    model = build_agent("cnn", board_size=8)
    env = EnvConfig(board_size=8, starvation_steps=16)
    kwargs = {"episodes": 6, "device": torch.device("cpu"), "seed": 5}
    first = evaluate(model, env, RewardConfig(), **kwargs)
    assert first.episodes == 6
    assert first.length_mean >= 1
    assert first.score_min <= first.score_mean <= first.score_max
    assert evaluate(model, env, RewardConfig(), **kwargs) == first
    assert model.training


def test_smoke_training_writes_a_complete_run_folder(tmp_path):
    run_dir = train(_smoke_config(tmp_path), progress=False)
    assert {p.name for p in run_dir.iterdir()} == {
        "run.json",
        "config.yaml",
        "metrics.jsonl",
        "tb",
        "checkpoints",
    }
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()]
    assert [line["update"] for line in lines] == [1, 2, 3, 4]
    assert lines[-1]["env_steps"] == 512
    assert "eval/score_mean" in lines[1] and "eval/score_mean" not in lines[0]
    assert {"loss/policy", "ppo/approx_kl", "ppo/explained_variance", "train/fps"} <= set(lines[0])
    assert load_config(run_dir / "config.yaml") == _smoke_config(tmp_path)
    for name in ("best", "last"):
        checkpoint = load_checkpoint(run_dir / "checkpoints" / f"{name}.pt")
        assert checkpoint["config"]["agent"]["name"] == "cnn"
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["update"] == 4


def test_same_seed_gives_identical_curves_on_cpu(tmp_path):
    def curve(sub):
        run_dir = train(_smoke_config(tmp_path / sub), progress=False)
        lines = (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
        return [
            {k: v for k, v in json.loads(x).items() if k not in ("wall_time", "train/fps")}
            for x in lines
        ]

    assert curve("a") == curve("b")


def test_switches_run_end_to_end(tmp_path):
    config = apply_overrides(
        _smoke_config(tmp_path),
        [
            "ppo.lr_schedule=linear",
            "ppo.clip_value_loss=true",
            "ppo.entropy_schedule=exponential",
            "ppo.target_kl=0.0001",
        ],
    )
    run_dir = train(config, progress=False)
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()]
    assert lines[1]["ppo/lr"] < lines[0]["ppo/lr"]
    assert lines[1]["ppo/entropy_coef"] < lines[0]["ppo/entropy_coef"]
    assert all(line["ppo/epochs"] <= 2 for line in lines)


def _metric_lines(run_dir):
    return [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()]


def test_lr_schedule_starts_at_the_base_lr_and_ends_at_zero(tmp_path):
    config = _smoke_config(tmp_path, "ppo.lr_schedule=linear")
    lines = _metric_lines(train(config, progress=False))
    assert lines[0]["ppo/lr"] == config.ppo.lr
    assert lines[-1]["ppo/lr"] == 0.0


@pytest.mark.parametrize("schedule", ["linear", "exponential"])
def test_entropy_schedule_starts_at_the_start_value_and_ends_at_the_final_value(tmp_path, schedule):
    config = _smoke_config(tmp_path, f"ppo.entropy_schedule={schedule}")
    lines = _metric_lines(train(config, progress=False))
    assert lines[0]["ppo/entropy_coef"] == pytest.approx(config.ppo.entropy_coef)
    assert lines[-1]["ppo/entropy_coef"] == pytest.approx(config.ppo.entropy_coef_final)


def test_constant_schedules_never_move(tmp_path):
    config = _smoke_config(tmp_path)
    lines = _metric_lines(train(config, progress=False))
    assert {line["ppo/lr"] for line in lines} == {config.ppo.lr}
    assert {line["ppo/entropy_coef"] for line in lines} == {config.ppo.entropy_coef}


@pytest.fixture
def restore_deterministic_settings():
    """``train.deterministic`` flips process-global torch switches; undo them after the test."""
    cudnn = torch.backends.cudnn
    benchmark, deterministic = cudnn.benchmark, cudnn.deterministic
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(False)
    cudnn.benchmark, cudnn.deterministic = benchmark, deterministic
    if cublas is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas


def test_deterministic_training_runs_with_the_default_architecture(
    tmp_path, restore_deterministic_settings
):
    config = _smoke_config(tmp_path, "train.deterministic=true")
    assert len(_metric_lines(train(config, progress=False))) == 4
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("agent", ["cnn", "cnn-spatial", "resnet"])
def test_deterministic_training_runs_on_cuda(tmp_path, restore_deterministic_settings, agent):
    config = _smoke_config(
        tmp_path, "train.device=cuda", "train.deterministic=true", f"agent.name={agent}"
    )
    assert len(_metric_lines(train(config, progress=False))) == 4
