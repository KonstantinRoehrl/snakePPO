import pytest
import torch
from torch import nn

from snakeppo.agents import (
    REGISTRY,
    PolicyPlayer,
    RecurrentPolicy,
    available_agents,
    build_agent,
    episode_starts,
)
from snakeppo.agents.recurrent import CELLS

SLOTS = {"gru": 1, "lstm": 2}


def _model(cell: str, in_channels: int = 4) -> RecurrentPolicy:
    torch.manual_seed(0)
    model = build_agent("recurrent", board_size=8, in_channels=in_channels, cell=cell)
    assert isinstance(model, RecurrentPolicy)
    return model


def _random_state(model: RecurrentPolicy, batch: int) -> torch.Tensor:
    return torch.randn(batch, model.state_slots, model.hidden_size)


def test_recurrent_is_registered_and_size_agnostic():
    assert "recurrent" in available_agents()
    assert REGISTRY["recurrent"].size_agnostic
    assert not REGISTRY["recurrent"].requires_even_board


def test_cells_are_gru_and_lstm():
    assert CELLS == ("gru", "lstm")
    assert isinstance(_model("gru").rnn, nn.GRU)
    assert isinstance(_model("lstm").rnn, nn.LSTM)


def test_an_unknown_cell_names_the_valid_ones():
    with pytest.raises(
        ValueError, match=r"agent 'recurrent' cell must be one of gru, lstm, got 'rnn'"
    ):
        build_agent("recurrent", board_size=8, cell="rnn")


@pytest.mark.parametrize("cell", ["gru", "lstm"])
@pytest.mark.parametrize("in_channels", [4, 6])
def test_state_and_step_shapes(cell, in_channels):
    model = _model(cell, in_channels)
    state = model.initial_state(5, "cpu")
    assert state.shape == (5, SLOTS[cell], 256)
    assert not state.any()
    obs = torch.rand(5, in_channels, 8, 8)
    logits, value, new_state = model(obs, state, torch.zeros(5, dtype=torch.bool))
    assert logits.shape == (5, 3) and value.shape == (5,)
    assert new_state.shape == state.shape and new_state.any()
    actions, log_probs, values, act_state = model.act(obs, state, torch.zeros(5, dtype=torch.bool))
    assert actions.shape == log_probs.shape == values.shape == (5,)
    assert act_state.shape == state.shape
    greedy, _, _, _ = model.act(obs, state, torch.zeros(5, dtype=torch.bool), greedy=True)
    assert torch.equal(greedy, logits.argmax(dim=-1))


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_a_start_runs_the_step_from_a_zero_state(cell):
    model = _model(cell)
    obs = torch.rand(4, 4, 8, 8)
    carried = _random_state(model, 4)
    starts = torch.tensor([True, False, True, False])
    logits, value, state = model(obs, carried, starts)
    fresh_logits, fresh_value, fresh_state = model(obs, model.initial_state(4, "cpu"), starts)
    kept_logits, kept_value, kept_state = model(obs, carried, torch.zeros(4, dtype=torch.bool))
    for got, fresh, kept in (
        (logits, fresh_logits, kept_logits),
        (value, fresh_value, kept_value),
        (state, fresh_state, kept_state),
    ):
        assert torch.allclose(got[starts], fresh[starts], atol=1e-6)
        assert torch.allclose(got[~starts], kept[~starts], atol=1e-6)
    assert not torch.allclose(kept_logits[starts], fresh_logits[starts], atol=1e-6)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_unroll_equals_one_forward_call_per_step(cell):
    model = _model(cell)
    generator = torch.Generator().manual_seed(1)
    obs = torch.rand(6, 4, 4, 8, 8, generator=generator)
    starts = torch.rand(6, 4, generator=generator) < 0.3
    start_state = _random_state(model, 4)
    logits, values, final = model.unroll(obs, start_state, starts)
    assert logits.shape == (6, 4, 3) and values.shape == (6, 4) and final.shape == start_state.shape
    state = start_state
    for t in range(6):
        step_logits, step_value, state = model(obs[t], state, starts[t])
        assert torch.allclose(logits[t], step_logits, atol=1e-5)
        assert torch.allclose(values[t], step_value, atol=1e-5)
    assert torch.allclose(final, state, atol=1e-5)


def test_episode_starts_marks_the_step_after_each_done():
    done = torch.tensor([[False, True], [True, False], [False, False]])
    starts = episode_starts(done)
    assert starts.dtype == torch.bool
    assert starts.tolist() == [[False, False], [False, True], [True, False]]


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_replaying_a_rollout_reproduces_its_log_probs_and_values(cell):
    """The update's replay from the saved start state matches the rollout: no state drift."""
    model = _model(cell)
    steps, envs = 10, 4
    generator = torch.Generator().manual_seed(2)
    state = model.initial_state(envs, "cpu")
    start_state = state.clone()
    no_starts = torch.zeros(envs, dtype=torch.bool)
    observations, actions, log_probs, values, dones = [], [], [], [], []
    with torch.no_grad():
        for _ in range(steps):
            obs = torch.rand(envs, 4, 8, 8, generator=generator)
            action, log_prob, value, state = model.act(obs, state, no_starts)
            done = torch.rand(envs, generator=generator) < 0.3
            state = state.masked_fill(done[:, None, None], 0.0)  # eager reset after the step
            for record, item in zip(
                (observations, actions, log_probs, values, dones),
                (obs, action, log_prob, value, done),
                strict=True,
            ):
                record.append(item)
    done = torch.stack(dones)
    assert done[:-1].any(), "the rollout must contain mid-rollout episode ends"
    logits, replayed_values, final = model.unroll(
        torch.stack(observations), start_state, episode_starts(done)
    )
    replayed = torch.distributions.Categorical(logits=logits).log_prob(torch.stack(actions))
    assert torch.allclose(replayed, torch.stack(log_probs), atol=1e-5)
    assert torch.allclose(replayed_values, torch.stack(values), atol=1e-5)
    assert torch.allclose(final.masked_fill(done[-1][:, None, None], 0.0), state, atol=1e-5)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_a_recurrent_model_runs_on_any_board_size(cell):
    model = _model(cell)
    for size in (5, 12, 20):
        logits, value, state = model(
            torch.rand(2, 4, size, size), model.initial_state(2, "cpu"), torch.zeros(2, dtype=bool)
        )
        assert logits.shape == (2, 3) and value.shape == (2,)
        assert state.shape == (2, SLOTS[cell], 256)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_recurrent_networks_avoid_torchs_adaptive_pool(cell):
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in _model(cell).modules())


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_policy_player_carries_state_and_resets_only_finished_games(cell):
    model = _model(cell)
    player = PolicyPlayer(model, num_games=3, device="cpu")
    assert player.state is not None and not player.state.any()
    obs = torch.rand(3, 4, 8, 8)
    with torch.no_grad():
        expected, _, _, expected_state = model.act(
            obs, model.initial_state(3, "cpu"), torch.zeros(3, dtype=torch.bool), greedy=True
        )
        actions = player.act(obs, greedy=True)
    assert torch.equal(actions, expected)
    assert torch.equal(player.state, expected_state)
    before = player.state.clone()
    player.end_episodes(torch.tensor([False, True, False]))
    assert not player.state[1].any()
    assert torch.equal(player.state[[0, 2]], before[[0, 2]])


def test_policy_player_plays_a_feed_forward_model_statelessly():
    torch.manual_seed(0)
    model = build_agent("cnn", board_size=8)
    player = PolicyPlayer(model, num_games=3, device="cpu")
    obs = torch.rand(3, 4, 8, 8)
    with torch.no_grad():
        assert torch.equal(player.act(obs, greedy=True), model(obs)[0].argmax(dim=-1))
    player.end_episodes(torch.tensor([True, False, True]))
    assert player.state is None


def test_policy_player_accepts_any_object_with_a_feed_forward_act():
    class Straight:
        def act(self, obs, greedy=False):
            n = obs.shape[0]
            return torch.zeros(n, dtype=torch.long), torch.zeros(n), torch.zeros(n)

    player = PolicyPlayer(Straight(), num_games=2, device="cpu")
    assert player.act(torch.rand(2, 4, 8, 8), greedy=True).tolist() == [0, 0]
    player.end_episodes(torch.tensor([True, True]))
