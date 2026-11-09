import pytest
import torch
from torch import nn

from snakeppo.agents import REGISTRY, available_agents, build_agent, count_parameters
from snakeppo.agents.networks import (
    DeterministicAdaptiveAvgPool2d,
    conv_features,
    sinusoidal_positions_2d,
)

SIZE_AGNOSTIC = ["attention", "cnn", "cnn-medium", "cnn-small", "cnn-tiny"]
BOARD_BOUND = ["cnn-spatial", "mlp", "resnet"]
FEED_FORWARD = sorted(SIZE_AGNOSTIC + BOARD_BOUND)


def test_registry_lists_every_architecture():
    assert available_agents() == [
        "attention",
        "cnn",
        "cnn-medium",
        "cnn-small",
        "cnn-spatial",
        "cnn-tiny",
        "mlp",
        "recurrent",
        "resnet",
    ]


def _shape_cases() -> list[tuple[str, int, int]]:
    """Every feed-forward agent on boards 8 and 16, plus odd 9 where the agent accepts it."""
    return [
        (name, board_size, in_channels)
        for name in FEED_FORWARD
        for board_size in (8, 9, 16)
        if not (board_size % 2 and name in ("cnn-spatial", "resnet"))
        for in_channels in (4, 6)
    ]


@pytest.mark.parametrize(("name", "board_size", "in_channels"), _shape_cases())
def test_every_agent_maps_observations_to_logits_and_values(name, board_size, in_channels):
    model = build_agent(name, board_size=board_size, in_channels=in_channels)
    obs = torch.rand(5, in_channels, board_size, board_size)
    logits, value = model(obs)
    assert logits.shape == (5, 3)
    assert value.shape == (5,)
    actions, log_probs, values = model.act(obs)
    assert actions.shape == log_probs.shape == values.shape == (5,)
    greedy, _, _ = model.act(obs, greedy=True)
    assert torch.equal(greedy, logits.argmax(dim=-1))
    assert count_parameters(model) > 0


@pytest.mark.parametrize("name", SIZE_AGNOSTIC)
def test_size_agnostic_agents_run_on_any_board_size(name):
    assert REGISTRY[name].size_agnostic
    model = build_agent(name, board_size=8)
    for size in (5, 7, 12, 20):
        logits, value = model(torch.rand(2, 4, size, size))
        assert logits.shape == (2, 3) and value.shape == (2,)
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("name", BOARD_BOUND)
def test_board_bound_agents_are_flagged_and_reject_another_size(name):
    assert not REGISTRY[name].size_agnostic
    model = build_agent(name, board_size=8)
    with pytest.raises(RuntimeError):
        model(torch.rand(2, 4, 12, 12))


def test_unknown_agent_names_the_available_ones():
    with pytest.raises(
        ValueError,
        match=(
            "attention, cnn, cnn-medium, cnn-small, cnn-spatial, cnn-tiny, mlp, recurrent, resnet"
        ),
    ):
        build_agent("lstm", board_size=16)


@pytest.mark.parametrize("name", ["cnn-spatial", "resnet"])
def test_board_dependent_agents_reject_odd_boards(name):
    with pytest.raises(ValueError, match="even"):
        build_agent(name, board_size=9)


def test_cnn_widths_give_increasing_capacity():
    tiny, small, medium = (
        count_parameters(build_agent(name, board_size=16))
        for name in ("cnn-tiny", "cnn-small", "cnn-medium")
    )
    assert tiny < small < medium


@pytest.mark.parametrize("width", [16, 32, 64])
@pytest.mark.parametrize("board_size", [5, 8, 9, 16])
def test_conv_features_pool_to_a_fixed_feature_count(width, board_size):
    features = conv_features(4, width)(torch.rand(3, 4, board_size, board_size))
    assert features.shape == (3, 2 * width * 16)


def test_sinusoidal_positions_encode_row_and_column_row_major():
    positions = sinusoidal_positions_2d(5, 64)
    assert positions.shape == (25, 64)
    assert torch.unique(positions, dim=0).shape[0] == 25
    grid = positions.view(5, 5, 64)
    # the first half of the dims depends only on the row, the second half only on the column
    assert torch.equal(grid[:, :, :32], grid[:, :1, :32].expand(5, 5, 32))
    assert torch.equal(grid[:, :, 32:], grid[:1, :, 32:].expand(5, 5, 32))
    assert not torch.equal(grid[0, 0], grid[0, 1]) and not torch.equal(grid[0, 0], grid[1, 0])


def test_sinusoidal_positions_need_a_dimension_divisible_by_four():
    with pytest.raises(ValueError, match="divisible by 4"):
        sinusoidal_positions_2d(5, 62)


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


@pytest.mark.parametrize("name", available_agents())
def test_no_registered_network_uses_torchs_adaptive_pool(name):
    """``nn.AdaptiveAvgPool2d`` has no deterministic CUDA backward (``train.deterministic``)."""
    model = build_agent(name, board_size=8)
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())


def test_unexpected_agent_kwargs_name_the_agent_the_key_and_the_accepted_ones():
    with pytest.raises(ValueError, match=r"agent 'cnn'.*'foo'.*accepted.*none"):
        build_agent("cnn", board_size=16, foo=1)


def test_cnn_capacities_accept_no_agent_kwargs():
    with pytest.raises(ValueError, match=r"agent 'cnn-small'.*'width'.*accepted.*none"):
        build_agent("cnn-small", board_size=16, width=8)
