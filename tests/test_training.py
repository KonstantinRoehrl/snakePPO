import json
from pathlib import Path

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
