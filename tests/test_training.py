import json
import math
import os
from pathlib import Path

import pytest
import torch

from snakeppo.agents import available_agents, build_agent
from snakeppo.builders import make_policy, observation_record
from snakeppo.config import (
    REWARD_STRATEGIES,
    EnvConfig,
    ObservationConfig,
    apply_overrides,
    load_config,
)
from snakeppo.evaluation import evaluate
from snakeppo.ppo import trainer
from snakeppo.ppo.trainer import train
from snakeppo.runs import load_checkpoint, read_run_info

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
# The observation record of the smoke config: today's four binary channels on its 8x8 board.
SMOKE_RECORD = {"channels": ["body", "head", "food", "wall"], "size": 8, "view": "full"}
OBSERVATION_SWITCHES = {
    "body-order": ("observation.body=order",),
    "tail": ("observation.tail=true",),
    "hunger": ("observation.hunger=true",),
    "local-view": ("observation.view=local", "observation.view_size=5"),
    "local-view-mlp": ("observation.view=local", "observation.view_size=5", "agent.name=mlp"),
}


def _smoke_config(tmp_path, *overrides):
    return load_config(SMOKE, [f"train.runs_dir={tmp_path.as_posix()}", *overrides])


def test_evaluate_counts_one_finished_episode_per_game_and_is_reproducible():
    torch.manual_seed(0)
    model = build_agent("cnn", board_size=8)
    env = EnvConfig(board_size=8, starvation_steps=16)
    kwargs = {"episodes": 6, "device": torch.device("cpu"), "seed": 5}
    first = evaluate(model, env, ObservationConfig(), **kwargs)
    assert first.episodes == 6
    assert first.length_mean >= 1
    assert first.score_min <= first.score_mean <= first.score_max
    assert evaluate(model, env, ObservationConfig(), **kwargs) == first
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
    assert not any(key.startswith("curriculum/") for line in lines for key in line)
    assert {"loss/policy", "ppo/approx_kl", "ppo/explained_variance", "train/fps"} <= set(lines[0])
    assert load_config(run_dir / "config.yaml") == _smoke_config(tmp_path)
    for name in ("best", "last"):
        checkpoint = load_checkpoint(run_dir / "checkpoints" / f"{name}.pt")
        assert checkpoint["config"]["agent"]["name"] == "cnn"
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["update"] == 4


def test_finished_run_records_schema_4_run_info_and_checkpoints(tmp_path):
    run_dir = train(_smoke_config(tmp_path, "train.group=g@1"), progress=False)
    info = json.loads((run_dir / "run.json").read_text("utf-8"))
    assert info["schema_version"] == 4
    assert info["observation"] == SMOKE_RECORD
    assert info["status"] == "finished"
    assert info["total_updates"] == 4
    assert info["group"] == "g@1"
    assert info["resumed"] == []
    last = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert last["schema_version"] == 4 and last["update"] == 4 and last["env_steps"] == 512
    assert last["observation"] == SMOKE_RECORD
    assert load_checkpoint(run_dir / "checkpoints" / "best.pt")["observation"] == SMOKE_RECORD
    assert {"best_eval", "wall_time", "rng", "env"} <= last.keys()
    assert "curriculum" not in last  # only runs with curriculum.grow on store it
    assert "policy_state" not in last  # only recurrent runs store it
    assert last["best_eval"] == max(
        line["eval/score_mean"] for line in _metric_lines(run_dir) if "eval/score_mean" in line
    )
    assert last["wall_time"] >= _metric_lines(run_dir)[-1]["wall_time"]
    assert last["rng"]["cuda"] is None
    assert set(last["env"]) >= {"head", "food", "generator"}


class _Boom(Exception):
    pass


def _raise_at_update(monkeypatch, update, error):
    """Make the trainer's metrics logger raise ``error`` right after logging ``update``."""
    real_log = trainer.MetricsLogger.log

    def log(self, number, env_steps, metrics):
        real_log(self, number, env_steps, metrics)
        if number == update:
            raise error

    monkeypatch.setattr(trainer.MetricsLogger, "log", log)


@pytest.mark.parametrize(
    ("error", "status"), [(KeyboardInterrupt(), "interrupted"), (_Boom(), "failed")]
)
def test_an_aborted_run_records_why_and_re_raises(tmp_path, monkeypatch, capsys, error, status):
    _raise_at_update(monkeypatch, 3, error)
    with pytest.raises(type(error)):
        train(_smoke_config(tmp_path), progress=False)
    (run_dir,) = tmp_path.iterdir()
    assert json.loads((run_dir / "run.json").read_text("utf-8"))["status"] == status
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["update"] == 2
    if status == "interrupted":
        assert f"snakeppo train --resume {run_dir}" in capsys.readouterr().out


def test_checkpoint_interval_writes_last_between_evals(tmp_path, monkeypatch):
    saved = []
    real_save = trainer.save_checkpoint

    def save(path, **state):
        saved.append((path.name, state["update"]))
        real_save(path, **state)

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    config = _smoke_config(
        tmp_path,
        "train.total_env_steps=1280",
        "train.eval_interval=4",
        "train.checkpoint_interval=3",
    )
    train(config, progress=False)  # 10 updates: evals at 4, 8, 10; interval hits 3, 6, 9
    assert [u for name, u in saved if name == "last.pt"] == [3, 4, 6, 8, 9, 10]
    assert {u for name, u in saved if name == "best.pt"} <= {4, 8, 10}


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


# every registered architecture, plus the recurrent one with its other cell
CUDA_AGENTS = {name: (f"agent.name={name}",) for name in available_agents()} | {
    "recurrent-lstm": ("agent.name=recurrent", "agent.kwargs={cell: lstm}")
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("agent", CUDA_AGENTS.values(), ids=CUDA_AGENTS.keys())
def test_deterministic_training_runs_on_cuda(tmp_path, restore_deterministic_settings, agent):
    config = _smoke_config(tmp_path, "train.device=cuda", "train.deterministic=true", *agent)
    assert len(_metric_lines(train(config, progress=False))) == 4


@pytest.mark.parametrize("strategy", REWARD_STRATEGIES)
def test_every_reward_strategy_trains_end_to_end(tmp_path, strategy):
    lines = _metric_lines(
        train(_smoke_config(tmp_path, f"reward.strategy={strategy}"), progress=False)
    )
    assert len(lines) == 4 and "eval/score_mean" in lines[-1]
    assert all(math.isfinite(line["train/reward_mean"]) for line in lines)


def test_the_default_policy_is_built_exactly_as_before():
    config = load_config(SMOKE)
    torch.manual_seed(0)
    built = make_policy(config)
    torch.manual_seed(0)
    reference = build_agent("cnn", 8)
    assert built.state_dict().keys() == reference.state_dict().keys()
    assert all(
        torch.equal(built.state_dict()[k], reference.state_dict()[k]) for k in built.state_dict()
    )
    assert observation_record(config) == SMOKE_RECORD


def test_make_policy_sizes_the_network_from_the_observation():
    config = apply_overrides(
        load_config(SMOKE),
        [
            "observation.tail=true",
            "observation.view=local",
            "observation.view_size=5",
            "agent.name=mlp",
        ],
    )
    assert observation_record(config) == {
        "channels": ["body", "head", "food", "wall", "tail"],
        "size": 5,
        "view": "local",
    }
    logits, values = make_policy(config)(torch.zeros(2, 5, 5, 5))
    assert logits.shape == (2, 3) and values.shape == (2,)


@pytest.mark.parametrize(
    ("overrides", "dtype"),
    [
        ((), torch.uint8),
        (("observation.tail=true",), torch.float32),
        (("observation.hunger=true",), torch.float32),
    ],
    ids=["default", "tail", "hunger"],
)
def test_the_rollout_buffer_keeps_uint8_only_for_an_all_binary_observation(
    tmp_path, monkeypatch, overrides, dtype
):
    buffers = []
    real_buffer = trainer.RolloutBuffer

    def spy(*args, **kwargs):
        buffers.append(real_buffer(*args, **kwargs))
        return buffers[-1]

    monkeypatch.setattr(trainer, "RolloutBuffer", spy)
    train(_smoke_config(tmp_path, *overrides), progress=False)
    (buffer,) = buffers
    assert buffer.obs.dtype == dtype
    if "observation.hunger=true" in overrides:
        # the hunger plane is steps_since_food / starvation_steps: its fractions must survive
        assert (buffer.obs.frac() != 0).any()


@pytest.mark.parametrize(
    "overrides", OBSERVATION_SWITCHES.values(), ids=OBSERVATION_SWITCHES.keys()
)
def test_every_observation_switch_trains_end_to_end(tmp_path, overrides):
    config = _smoke_config(tmp_path, *overrides)
    run_dir = train(config, progress=False)
    lines = _metric_lines(run_dir)
    assert len(lines) == 4 and "eval/score_mean" in lines[-1]
    record = observation_record(config)
    assert record != SMOKE_RECORD
    assert read_run_info(run_dir)["observation"] == record
    for name in ("best", "last"):
        assert load_checkpoint(run_dir / "checkpoints" / f"{name}.pt")["observation"] == record
