import pytest
import torch

from snakeppo.agents import available_agents, build_agent, count_parameters


def test_registry_lists_the_c1_architectures():
    assert available_agents() == ["cnn", "cnn-spatial", "resnet"]


@pytest.mark.parametrize("name", ["cnn", "cnn-spatial", "resnet"])
@pytest.mark.parametrize("board_size", [8, 16])
def test_every_agent_maps_observations_to_logits_and_values(name, board_size):
    model = build_agent(name, board_size=board_size)
    obs = torch.zeros(5, 4, board_size, board_size)
    logits, value = model(obs)
    assert logits.shape == (5, 3)
    assert value.shape == (5,)
    actions, log_probs, values = model.act(obs)
    assert actions.shape == log_probs.shape == values.shape == (5,)
    greedy, _, _ = model.act(obs, greedy=True)
    assert torch.equal(greedy, logits.argmax(dim=-1))
    assert count_parameters(model) > 0


def test_unknown_agent_names_the_available_ones():
    with pytest.raises(ValueError, match="cnn, cnn-spatial, resnet"):
        build_agent("lstm", board_size=16)


@pytest.mark.parametrize("name", ["cnn-spatial", "resnet"])
def test_board_dependent_agents_reject_odd_boards(name):
    with pytest.raises(ValueError, match="even"):
        build_agent(name, board_size=9)


def test_cnn_accepts_odd_boards():
    logits, _ = build_agent("cnn", board_size=9)(torch.zeros(1, 4, 9, 9))
    assert logits.shape == (1, 3)
