import random
from collections import deque

import pytest
import torch

from snakeppo.config import REWARD_STRATEGIES, RewardConfig
from snakeppo.rewards import (
    REWARDS,
    ClassicReward,
    DistanceReward,
    FreeSpaceReward,
    HungerReward,
    PotentialReward,
    RewardSpec,
    SparseReward,
    StepContext,
    available_rewards,
    build_reward,
    manhattan,
    reachable_fraction,
    register,
)

BOARD = 8


def _cells(*cells):
    return torch.tensor(cells, dtype=torch.long)


def _playable(size=BOARD):
    playable = torch.zeros((size, size), dtype=torch.bool)
    playable[1:-1, 1:-1] = True
    return playable


def _step(
    old_head=(4, 4),
    new_head=(3, 4),
    food=(1, 4),
    *,
    next_food=None,
    ate=False,
    terminated=False,
    starved=False,
    steps_since_food=1,
    starvation_steps=10,
    body=None,
):
    """A one-env step on an open 8x8 board; the head moves unless the step terminated."""
    head = old_head if terminated else new_head
    if body is None:
        body = torch.zeros((1, BOARD, BOARD), dtype=torch.bool)
        body[0, head[0], head[1]] = True
    return StepContext(
        old_head=_cells(old_head),
        new_head=_cells(new_head),
        head=_cells(head),
        old_food=_cells(food),
        food=_cells(next_food or food),
        ate=torch.tensor([ate]),
        terminated=torch.tensor([terminated]),
        starved=torch.tensor([starved]),
        steps_since_food=torch.tensor([steps_since_food]),
        starvation_steps=starvation_steps,
        body=body,
        playable=_playable(),
    )


def test_classic_shapes_on_squared_distance_and_is_replaced_on_eating_and_dying():
    classic = ClassicReward(apple=1.0, death=-3.0, approach=0.5, retreat=0.25)
    assert classic(_step()).item() == 0.5 * 5  # squared distance 9 -> 4
    assert classic(_step(new_head=(5, 4))).item() == 0.25 * -7  # 9 -> 16
    assert classic(_step(new_head=(1, 4), ate=True)).item() == 1.0
    assert classic(_step(new_head=(0, 4), terminated=True)).item() == -3.0


def test_sparse_pays_only_apples_and_deaths():
    sparse = SparseReward(apple=2.0, death=-5.0)
    assert sparse(_step()).item() == 0.0
    assert sparse(_step(ate=True)).item() == 2.0
    assert sparse(_step(terminated=True)).item() == -5.0


def test_distance_pays_scale_per_manhattan_cell():
    distance = DistanceReward(apple=1.0, death=-3.0, scale=0.25)
    assert distance(_step()).item() == 0.25
    assert distance(_step(new_head=(4, 5))).item() == -0.25
    assert distance(_step(ate=True)).item() == 1.0
    assert distance(_step(terminated=True)).item() == -3.0
    assert manhattan(_cells((4, 4)), _cells((1, 6))).item() == 5.0


def test_potential_adds_discounted_shaping_on_every_step():
    potential = PotentialReward(apple=1.0, death=-3.0, scale=0.1, discount=0.9)
    # phi(s) = -0.3, phi(s') = -0.2: F = 0.9 * -0.2 + 0.3
    assert potential(_step()).item() == pytest.approx(0.12)
    # eating: phi(s') uses the fresh food at (6, 2), 6 cells from the head at (2, 4)
    eaten = _step(new_head=(2, 4), food=(2, 4), next_food=(6, 2), ate=True)
    assert potential(eaten).item() == pytest.approx(1.0 + 0.9 * -0.6 + 0.2)
    # a terminal state has phi = 0, and the death reward is not replaced
    assert potential(_step(terminated=True)).item() == pytest.approx(-3.0 + 0.3)


def test_potential_shaping_telescopes_over_a_terminated_episode():
    gamma, scale = 0.97, 0.4
    shaping_only = PotentialReward(apple=0.0, death=0.0, scale=scale, discount=gamma)
    rng = random.Random(0)
    heads = [(rng.randrange(1, 7), rng.randrange(1, 7)) for _ in range(13)]
    foods = [(rng.randrange(1, 7), rng.randrange(1, 7)) for _ in range(13)]
    total = 0.0
    for t in range(12):
        last = t == 11
        step = StepContext(
            old_head=_cells(heads[t]),
            new_head=_cells(heads[t + 1]),
            head=_cells(heads[t] if last else heads[t + 1]),
            old_food=_cells(foods[t]),
            food=_cells(foods[t + 1]),
            ate=torch.tensor([False]),
            terminated=torch.tensor([last]),
            starved=torch.tensor([False]),
            steps_since_food=torch.tensor([t + 1]),
            starvation_steps=100,
            body=torch.zeros((1, BOARD, BOARD), dtype=torch.bool),
            playable=_playable(),
        )
        total += gamma**t * shaping_only(step).item()
    d0 = abs(heads[0][0] - foods[0][0]) + abs(heads[0][1] - foods[0][1])
    assert total == pytest.approx(scale * d0, abs=1e-5)  # -phi(s0)


def test_hunger_adds_a_ramp_and_a_timeout_penalty_to_classic():
    classic = ClassicReward()
    hunger = HungerReward(classic, timeout_penalty=1.0, hunger_penalty=0.5)
    base = classic(_step(steps_since_food=4)).item()
    assert hunger(_step(steps_since_food=4)).item() == pytest.approx(base - 0.5 * 4 / 10)
    starving = _step(steps_since_food=10, starved=True)
    assert hunger(starving).item() == pytest.approx(classic(starving).item() - 0.5 - 1.0)
    assert hunger(_step(ate=True, steps_since_food=0)).item() == 1.0
    assert hunger(_step(terminated=True, steps_since_food=7)).item() == -3.0


def test_free_space_is_classic_while_the_board_is_open_and_penalises_a_trap():
    classic = ClassicReward()
    free_space = FreeSpaceReward(classic, penalty=0.5)
    assert free_space(_step()).item() == classic(_step()).item()
    # a wall of body across row 4 cuts the head at (5, 1) off from the 18 free cells above it
    body = torch.zeros((1, BOARD, BOARD), dtype=torch.bool)
    body[0, 4, 1:7] = True
    body[0, 5, 1] = True
    trapped = _step(old_head=(5, 2), new_head=(5, 1), food=(1, 1), body=body)
    reachable = reachable_fraction(body, _cells((5, 1)), _playable()).item()
    assert reachable == pytest.approx(11 / 29)
    expected = classic(trapped).item() - 0.5 * (1 - reachable)
    assert free_space(trapped).item() == pytest.approx(expected)
    assert free_space(_step(ate=True, body=body)).item() == 1.0


def _bfs_fraction(body, head, playable):
    free = {
        (r, c)
        for r in range(playable.shape[0])
        for c in range(playable.shape[1])
        if playable[r, c] and not body[r, c]
    }
    if not free:
        return 1.0
    seen, queue = set(), deque([head])
    while queue:
        r, c = queue.popleft()
        for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if cell in free and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen) / len(free)


def test_reachable_fraction_matches_a_reference_bfs_on_random_boards():
    generator = torch.Generator().manual_seed(0)
    size, n = 10, 64
    playable = torch.zeros((size, size), dtype=torch.bool)
    playable[2:-2, 1:-1] = True  # an arena narrower than the board
    body = (torch.rand((n, size, size), generator=generator) < 0.45) & playable
    heads = torch.stack(
        [
            body[i].nonzero()[torch.randint(len(body[i].nonzero()), (1,), generator=generator)][0]
            for i in range(n)
        ]
    )
    body[0] = False  # one board with the head alone
    body[0, heads[0, 0], heads[0, 1]] = True
    body[1] = playable  # one board with no free cell
    heads[1] = body[1].nonzero()[0]
    fractions = reachable_fraction(body, heads, playable)
    for i in range(n):
        expected = _bfs_fraction(body[i], tuple(heads[i].tolist()), playable)
        assert fractions[i].item() == pytest.approx(expected), i


def test_registry_lists_every_configurable_strategy():
    assert available_rewards() == sorted(REWARD_STRATEGIES)
    assert all(REWARDS[name].description for name in available_rewards())
    with pytest.raises(ValueError, match="already registered"):
        register(RewardSpec("classic", "again", lambda config, discount: SparseReward()))


def test_build_reward_reads_its_own_config_values_and_the_discount():
    config = RewardConfig(apple=2.0, death=-1.0, potential_scale=0.3, free_space_penalty=0.7)
    assert build_reward(config, 0.9) == ClassicReward(2.0, -1.0, 0.02, 0.03)
    potential = build_reward(RewardConfig(strategy="potential", potential_scale=0.3), 0.95)
    assert potential == PotentialReward(1.0, -3.0, 0.3, 0.95)
    free_space = build_reward(RewardConfig(strategy="free_space", free_space_penalty=0.7), 0.9)
    assert free_space == FreeSpaceReward(ClassicReward(), 0.7)
    assert build_reward(RewardConfig(strategy="hunger"), 0.9) == HungerReward(
        ClassicReward(), 1.0, 0.01
    )
    assert build_reward(RewardConfig(strategy="distance"), 0.9) == DistanceReward(1.0, -3.0, 0.1)
    assert build_reward(RewardConfig(strategy="sparse"), 0.9) == SparseReward(1.0, -3.0)
    with pytest.raises(ValueError, match="unknown reward strategy 'nope'; available: classic"):
        build_reward(RewardConfig(strategy="nope"), 0.9)
