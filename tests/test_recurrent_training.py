import json
from pathlib import Path

import pytest
import torch

from snakeppo.agents import RecurrentPolicy, build_agent
from snakeppo.builders import make_policy, recurrent_minibatches
from snakeppo.config import EnvConfig, ObservationConfig, load_config
from snakeppo.env import TorchSnakeEnv
from snakeppo.evaluation import evaluate
from snakeppo.ppo import trainer
from snakeppo.ppo.trainer import ResumeError, resume, train
from snakeppo.runs import load_checkpoint, read_metrics

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
# smoke.yaml: 8 envs, 16 rollout steps, minibatch 64, so 2 groups of 4 envs; 4 updates, evals and
# checkpoints at 2 and 4.
NUM_ENVS, STEPS, UPDATES = 8, 16, 4
SLOTS = {"gru": 1, "lstm": 2}
TIMING_KEYS = ("wall_time", "train/fps")
# Stage 0 is a 6x6 arena with goal 4, promoted after 2 updates at the latest.
CURRICULUM = (
    "curriculum.grow=both",
    "curriculum.stages=2",
    "curriculum.start_board=6",
    "curriculum.min_updates=1",
    "curriculum.max_updates=2",
)


def _config(runs_dir, cell="gru", *overrides):
    return load_config(
        SMOKE,
        [
            f"train.runs_dir={runs_dir.as_posix()}",
            "agent.name=recurrent",
            f"agent.kwargs={{cell: {cell}}}",
            *overrides,
        ],
    )


def _curve(run_dir):
    return [
        {k: v for k, v in line.items() if k not in TIMING_KEYS} for line in read_metrics(run_dir)
    ]


def test_recurrent_minibatches_counts_groups_of_whole_envs(tmp_path):
    assert recurrent_minibatches(_config(tmp_path)) == 2  # 16 * 8 // 64
    assert recurrent_minibatches(_config(tmp_path, "gru", "ppo.minibatch_size=4096")) == 1
    baseline = load_config(Path(SMOKE).parent / "baseline.yaml")
    assert recurrent_minibatches(baseline) == 8  # 256 * 64 // 2048


def test_a_batching_that_splits_an_env_is_refused_before_any_run_folder(tmp_path):
    overrides = ("env.num_envs=5", "ppo.minibatch_size=32")  # 16 * 5 // 32 = 2 groups
    config = _config(tmp_path, "gru", *overrides)
    message = (
        r"agent 'recurrent' trains on whole envs: ppo.rollout_steps \* env.num_envs // "
        r"ppo.minibatch_size gives 2 minibatches, which must divide env.num_envs \(5\)"
    )
    with pytest.raises(ValueError, match=message):
        make_policy(config)
    with pytest.raises(ValueError, match=message):
        train(config, progress=False)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())
    feed_forward = load_config(SMOKE, ["agent.name=cnn", *overrides])
    assert not isinstance(make_policy(feed_forward), RecurrentPolicy)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_a_recurrent_run_trains_and_checkpoints_its_state(tmp_path, cell):
    run_dir = train(_config(tmp_path, cell), progress=False)
    lines = read_metrics(run_dir)
    assert [line["update"] for line in lines] == [1, 2, 3, 4]
    assert "eval/score_mean" in lines[1] and "eval/score_mean" in lines[-1]
    for name in ("best", "last"):
        state = load_checkpoint(run_dir / "checkpoints" / f"{name}.pt")["policy_state"]
        assert state.shape == (NUM_ENVS, SLOTS[cell], 256)
        assert state.dtype == torch.float32 and state.device.type == "cpu"


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_the_update_replays_each_rollout_from_its_start_state(tmp_path, monkeypatch, cell):
    calls = []
    real_loss = trainer.ppo_loss

    def spy(*args, **kwargs):
        calls.append([arg.detach().clone() for arg in args])
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(trainer, "ppo_loss", spy)
    train(_config(tmp_path, cell), progress=False)
    per_update = 2 * 2  # update_epochs * groups
    assert len(calls) == UPDATES * per_update
    for first in calls[::per_update]:
        # before the update's first optimizer step the weights are the rollout's weights, so
        # replaying whole envs from their start state reproduces the rollout's numbers
        new_log_probs, old_log_probs, _, new_values, old_values, *_ = first
        assert new_log_probs.shape == (STEPS * NUM_ENVS // 2,)
        assert torch.allclose(new_log_probs, old_log_probs, atol=1e-5)
        assert torch.allclose(new_values, old_values, atol=1e-5)


def _record_training(monkeypatch):
    """Spy on the recurrent model and the 8-env training env, in call order.

    Events: ``("act", state_in, new_state)``, ``("step", done, truncated)`` and
    ``("forward", state, starts)`` for model calls made outside ``act`` (the bootstraps).
    """
    events, inside_act = [], []
    real_act, real_forward = RecurrentPolicy.act, RecurrentPolicy.forward
    real_step = TorchSnakeEnv.step

    def act(self, obs, state, starts, greedy=False):
        inside_act.append(True)
        try:
            result = real_act(self, obs, state, starts, greedy)
        finally:
            inside_act.pop()
        if len(obs) == NUM_ENVS:
            events.append(("act", state, result[3]))
        return result

    def forward(self, obs, state, starts):
        if not inside_act:
            events.append(("forward", state, starts))
        return real_forward(self, obs, state, starts)

    def step(self, actions):
        result = real_step(self, actions)
        if self.num_envs == NUM_ENVS:
            events.append(("step", result[4]["done"].clone(), result[3].clone()))
        return result

    monkeypatch.setattr(RecurrentPolicy, "act", act)
    monkeypatch.setattr(RecurrentPolicy, "forward", forward)
    monkeypatch.setattr(TorchSnakeEnv, "step", step)
    return events


def test_the_rollout_resets_state_at_episode_starts_and_after_a_promotion(tmp_path, monkeypatch):
    events = _record_training(monkeypatch)
    # short starvation, so the rollout sees truncations as well as deaths
    config = _config(tmp_path, "gru", "env.starvation_steps=8", *CURRICULUM)
    run_dir = train(config, progress=False)
    (promoted_after,) = [
        line["update"] for line in read_metrics(run_dir) if line.get("curriculum/promoted")
    ]
    acts = [event for event in events if event[0] == "act"]
    steps = [event for event in events if event[0] == "step"]
    assert len(acts) == len(steps) == UPDATES * STEPS
    assert not acts[0][1].any()  # a new run starts from the zero state
    for k in range(len(acts) - 1):
        _, _, new_state = acts[k]
        _, done, _ = steps[k]
        if k + 1 == promoted_after * STEPS:
            expected = torch.zeros_like(new_state)  # the promotion's reset
        else:
            expected = torch.where(done[:, None, None], 0.0, new_state)
        assert torch.equal(acts[k + 1][1], expected)
    assert any(done.any() for _, done, _ in steps)

    truncations = 0
    for i, event in enumerate(events):
        if event[0] == "step" and event[2].any():
            truncations += 1
            kind, state, starts = events[i + 1]
            assert kind == "forward" and not starts.any()
            # the state that produced the truncated step, not the reset one
            assert torch.equal(state, events[i - 1][2][event[2]])
            assert state.abs().sum() > 0
    assert truncations > 0


def _interrupted_run(runs_dir, monkeypatch, after_update, *overrides):
    real_log = trainer.MetricsLogger.log

    def log(self, update, env_steps, metrics):
        real_log(self, update, env_steps, metrics)
        if update == after_update:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(trainer.MetricsLogger, "log", log)
        with pytest.raises(KeyboardInterrupt):
            train(_config(runs_dir, "gru", *overrides), progress=False)
    (run_dir,) = runs_dir.iterdir()
    return run_dir


def test_a_resumed_recurrent_run_matches_an_uninterrupted_one_on_cpu(tmp_path, monkeypatch):
    reference = train(_config(tmp_path / "reference"), progress=False)
    run_dir = _interrupted_run(tmp_path / "cut", monkeypatch, 3)
    assert load_checkpoint(run_dir / "checkpoints" / "last.pt")["update"] == 2

    resume(run_dir, progress=False)
    assert _curve(run_dir) == _curve(reference)
    for name in ("last", "best"):
        ours = load_checkpoint(run_dir / "checkpoints" / f"{name}.pt")
        theirs = load_checkpoint(reference / "checkpoints" / f"{name}.pt")
        assert all(torch.equal(ours["model"][key], theirs["model"][key]) for key in theirs["model"])
        assert torch.equal(ours["policy_state"], theirs["policy_state"])


def test_resume_refuses_a_recurrent_checkpoint_without_its_state(tmp_path, monkeypatch):
    run_dir = _interrupted_run(tmp_path, monkeypatch, 3)
    last = run_dir / "checkpoints" / "last.pt"
    checkpoint = load_checkpoint(last)
    del checkpoint["policy_state"]
    torch.save(checkpoint, last)
    with pytest.raises(ResumeError, match="has no policy_state to continue a recurrent run from"):
        resume(run_dir, progress=False)
    assert json.loads((run_dir / "run.json").read_text("utf-8"))["status"] == "interrupted"


def test_evaluate_plays_a_recurrent_model_from_a_fresh_state_every_time():
    torch.manual_seed(0)
    model = build_agent("recurrent", board_size=8, cell="lstm")
    env = EnvConfig(board_size=8, starvation_steps=16)
    kwargs = {"episodes": 6, "device": torch.device("cpu"), "seed": 5}
    first = evaluate(model, env, ObservationConfig(), **kwargs)
    assert first.episodes == 6 and first.length_mean >= 1
    assert evaluate(model, env, ObservationConfig(), **kwargs) == first
    assert model.training
