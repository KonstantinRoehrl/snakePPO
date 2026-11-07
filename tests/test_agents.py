import pytest
import torch
from torch import nn

from snakeppo.agents import available_agents, build_agent, count_parameters
from snakeppo.agents.networks import DeterministicAdaptiveAvgPool2d


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


def _cnn_pool_input_size(board_size: int) -> int:
    """Feature-map size entering the pool of ``cnn`` (two stride-2, kernel-3, padding-1 convs)."""
    size = board_size
    for _ in range(2):
        size = (size - 1) // 2 + 1
    return size


@pytest.mark.parametrize("board_size", [6, 8, 10, 12, 14, 16, 20])
def test_deterministic_pool_matches_torch_adaptive_pool(board_size):
    size = _cnn_pool_input_size(board_size)
    torch.manual_seed(0)
    data = torch.randn(3, 5, size, size)
    reference_in = data.clone().requires_grad_()
    pooled_in = data.clone().requires_grad_()
    reference = nn.AdaptiveAvgPool2d((4, 4))(reference_in)
    pooled = DeterministicAdaptiveAvgPool2d((4, 4))(pooled_in)
    assert pooled.shape == reference.shape == (3, 5, 4, 4)
    assert torch.allclose(pooled, reference, atol=1e-6)
    weights = torch.randn_like(reference)
    (reference * weights).sum().backward()
    (pooled * weights).sum().backward()
    assert torch.allclose(pooled_in.grad, reference_in.grad, atol=1e-6)


def test_cnn_has_no_adaptive_pool_without_a_deterministic_backward():
    model = build_agent("cnn", board_size=16)
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())


def test_unexpected_agent_kwargs_name_the_agent_the_key_and_the_accepted_ones():
    with pytest.raises(ValueError, match=r"agent 'cnn'.*'foo'.*accepted.*none"):
        build_agent("cnn", board_size=16, foo=1)
