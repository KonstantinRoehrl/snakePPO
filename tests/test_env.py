import pytest
import torch

from snakeppo.env import (
    DOWN,
    END_BODY,
    END_GOAL,
    END_NONE,
    END_REASONS,
    END_STARVED,
    END_WALL,
    LEFT,
    RIGHT,
    STRAIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    UP,
    TorchSnakeEnv,
)
from snakeppo.rewards import ClassicReward, SparseReward


def _env(**kwargs):
    defaults = {"num_envs": 1, "board_size": 8, "seed": 0}
    return TorchSnakeEnv(**(defaults | kwargs))


def _act(env, action):
    return env.step(torch.full((env.num_envs,), action, dtype=torch.long))


def test_reset_starts_every_snake_in_the_centre_facing_up():
    env = _env(num_envs=3)
    obs = env.reset()
    assert obs.shape == (3, 4, 8, 8)
    assert obs.dtype == torch.float32
    assert env.head.tolist() == [[4, 4]] * 3
    assert env.heading.tolist() == [UP] * 3
    assert env.snake_len.tolist() == [1, 1, 1]
    grid = env.render_grid()
    assert grid[:, 3].sum().item() == 3 * (8 * 4 - 4)
    assert (grid[:, 2].sum(dim=(1, 2)) == 1).all()


def test_relative_actions_turn_the_snake():
    env = _env()
    env.place_snake(0, [(4, 4)], UP, food=(1, 1))
    _act(env, STRAIGHT)
    assert env.head.tolist() == [[3, 4]]
    _act(env, TURN_RIGHT)
    assert env.heading.tolist() == [RIGHT] and env.head.tolist() == [[3, 5]]
    _act(env, TURN_LEFT)
    assert env.heading.tolist() == [UP] and env.head.tolist() == [[2, 5]]
    _act(env, TURN_LEFT)
    assert env.heading.tolist() == [LEFT] and env.head.tolist() == [[2, 4]]


def test_eating_grows_scores_and_respawns_food_off_the_body():
    env = _env()
    env.place_snake(0, [(4, 4)], UP, food=(3, 4))
    _, reward, terminated, truncated, _ = _act(env, STRAIGHT)
    assert reward.item() == 1.0
    assert env.score.item() == 1.0 and env.snake_len.item() == 2
    assert not terminated.item() and not truncated.item()
    food = tuple(env.food[0].tolist())
    assert food not in {(3, 4), (4, 4)}
    assert not env.body[0, food[0], food[1]]


def test_shaping_rewards_approach_and_penalises_retreat():
    env = _env(reward=ClassicReward(approach=0.5, retreat=0.25))
    env.place_snake(0, [(4, 4)], UP, food=(1, 4))
    _, reward, *_ = _act(env, STRAIGHT)  # squared distance 9 -> 4
    assert reward.item() == 0.5 * 5
    env.place_snake(0, [(4, 4)], DOWN, food=(1, 4))
    _, reward, *_ = _act(env, STRAIGHT)  # squared distance 9 -> 16
    assert reward.item() == 0.25 * -7


def test_wall_death_terminates_and_auto_resets_in_the_same_step():
    env = _env()
    env.place_snake(0, [(1, 4)], UP, food=(6, 6))
    env.score[0] = 3
    obs, reward, terminated, truncated, info = _act(env, STRAIGHT)
    assert terminated.tolist() == [True] and truncated.tolist() == [False]
    assert reward.item() == -3.0
    assert info["done"].tolist() == [True]
    assert info["episode_score"].tolist() == [3.0]
    assert info["episode_length"].tolist() == [1]
    assert env.head.tolist() == [[4, 4]] and env.score.item() == 0.0
    assert torch.equal(obs, env.observe())


def test_running_into_own_body_terminates():
    env = _env()
    body = [(4, 4), (4, 5), (5, 5), (5, 4), (5, 3)]
    env.place_snake(0, body, LEFT, food=(1, 1))
    _, _, terminated, *_ = _act(env, TURN_LEFT)  # LEFT -> DOWN into (5, 4)
    assert terminated.tolist() == [True]


def test_starvation_truncates_with_bootstrappable_final_obs():
    env = _env(starvation_steps=2)
    env.place_snake(0, [(4, 4)], UP, food=(6, 6))
    _, _, terminated, truncated, _ = _act(env, STRAIGHT)
    assert not truncated.item()
    obs, _, terminated, truncated, info = _act(env, STRAIGHT)
    assert truncated.tolist() == [True] and terminated.tolist() == [False]
    assert info["final_obs"][0, 1].sum().item() == 1  # head present before reset
    assert not torch.equal(info["final_obs"], obs)
    assert env.head.tolist() == [[4, 4]]


def test_only_finished_envs_reset():
    env = _env(num_envs=2)
    env.place_snake(0, [(1, 4)], UP, food=(6, 6))
    env.place_snake(1, [(4, 4)], UP, food=(6, 6))
    _act(env, STRAIGHT)
    assert env.head.tolist() == [[4, 4], [3, 4]]


def test_observation_is_rotated_so_the_snake_faces_up():
    env = _env()
    env.place_snake(0, [(4, 4)], RIGHT, food=(4, 6))
    obs = env.observe()
    expected = torch.rot90(env.render_grid(), k=1, dims=(2, 3))
    assert torch.equal(obs, expected)
    food_row = obs[0, 2].nonzero()[0, 0].item()
    head_row = obs[0, 1].nonzero()[0, 0].item()
    assert food_row < head_row  # food straight ahead appears above the head


def _food_after_resets(seed, resets=3):
    env = _env(num_envs=4, seed=seed)
    sequence = []
    for _ in range(resets):
        env.reset()
        sequence.append(env.food.clone())
    return torch.stack(sequence)


def test_same_seed_reproduces_food_and_different_seed_does_not():
    assert torch.equal(_food_after_resets(1), _food_after_resets(1))
    assert not torch.equal(_food_after_resets(1), _food_after_resets(2))


def _random_actions(steps, num_envs, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 3, (steps, num_envs), generator=generator)


def _trajectory(env, actions):
    out = []
    for step_actions in actions:
        obs, reward, terminated, truncated, info = env.step(step_actions)
        out.append((obs, reward, terminated, truncated, info["episode_score"]))
    return out


def test_state_dict_round_trip_continues_the_same_trajectory(tmp_path):
    original = _env(num_envs=4, seed=3, starvation_steps=20)
    _trajectory(original, _random_actions(60, 4, seed=0))
    path = tmp_path / "env.pt"
    torch.save(original.state_dict(), path)

    restored = _env(num_envs=4, seed=99, starvation_steps=20)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(restored.observe(), original.observe())
    later = _random_actions(200, 4, seed=1)
    for a, b in zip(_trajectory(original, later), _trajectory(restored, later), strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_load_state_dict_rejects_a_different_env_shape():
    state = _env(num_envs=4).state_dict()
    with pytest.raises(ValueError, match="num_envs or board_size differ"):
        _env(num_envs=2).load_state_dict(state)
    del state["generator"]
    with pytest.raises(ValueError, match="missing generator"):
        _env(num_envs=4).load_state_dict(state)


# ------------------------------------------------------------ baseline bit-identity
class _LegacySnakeEnv(TorchSnakeEnv):
    """Today's step, reset and food draw, copied verbatim from before reward strategies."""

    reward_apple, reward_death, reward_approach, reward_retreat = 1.0, -3.0, 0.02, 0.03

    def step(self, actions):
        actions = actions.to(self.device)
        turn = torch.where(actions == TURN_LEFT, -1, torch.where(actions == TURN_RIGHT, 1, 0))
        self.heading = (self.heading + turn) % 4
        new_head = self.head + self._dirs[self.heading]

        old_dist = (self.head - self.food).float().pow(2).sum(dim=1)
        new_dist = (new_head - self.food).float().pow(2).sum(dim=1)
        delta = old_dist - new_dist
        reward = (
            delta.clamp(min=0) * self.reward_approach + delta.clamp(max=0) * self.reward_retreat
        )

        last = self.board_size - 1
        hit_wall = (new_head <= 0).any(dim=1) | (new_head >= last).any(dim=1)
        clamped = new_head.clamp(0, last)
        hit_body = self.body[self._env_ids, clamped[:, 0], clamped[:, 1]] & ~hit_wall
        terminated = hit_wall | hit_body
        alive = ~terminated

        ate = (new_head == self.food).all(dim=1) & alive
        reward = torch.where(ate, torch.full_like(reward, self.reward_apple), reward)
        reward = torch.where(terminated, torch.full_like(reward, self.reward_death), reward)

        self.head = torch.where(alive.unsqueeze(1), new_head, self.head)
        self.snake_len += ate.long()
        self.score += ate.float()
        shifted = self.body_positions.roll(1, dims=1)
        shifted[:, 0] = self.head
        self.body_positions = torch.where(alive.view(-1, 1, 1), shifted, self.body_positions)
        self._rebuild_body()
        if ate.any():
            self._spawn_food(ate)

        self.steps_since_food = torch.where(ate, 0, self.steps_since_food + 1)
        truncated = alive & (self.steps_since_food >= self.starvation_steps)
        done = terminated | truncated
        self.episode_length += 1
        self.episode_return += reward

        obs = self.observe()
        info = {
            "done": done,
            "final_obs": obs,
            "episode_score": self.score.clone(),
            "episode_length": self.episode_length.clone(),
            "episode_return": self.episode_return.clone(),
        }
        if done.any():
            info["final_obs"] = obs.clone()
            self._reset_envs(done)
            obs = self.observe()
        return obs, reward, terminated, truncated, info

    def _reset_envs(self, mask):
        center = self.board_size // 2
        m1 = mask.view(-1, 1)
        start = torch.full_like(self.head, center)
        self.head = torch.where(m1, start, self.head)
        self.heading = torch.where(mask, UP, self.heading)
        fresh_positions = torch.full_like(self.body_positions, -1)
        fresh_positions[:, 0] = center
        self.body_positions = torch.where(mask.view(-1, 1, 1), fresh_positions, self.body_positions)
        self.snake_len = torch.where(mask, 1, self.snake_len)
        zeros_f = torch.zeros_like(self.score)
        self.score = torch.where(mask, zeros_f, self.score)
        self.episode_return = torch.where(mask, zeros_f, self.episode_return)
        self.steps_since_food = torch.where(mask, 0, self.steps_since_food)
        self.episode_length = torch.where(mask, 0, self.episode_length)
        self._rebuild_body()
        self._spawn_food(mask)

    def _spawn_food(self, mask):
        inner = self.board_size - 2
        free = ~self.body[:, 1:-1, 1:-1].reshape(self.num_envs, -1)
        scores = torch.rand(free.shape, device=self.device, generator=self.generator)
        scores = scores.masked_fill(~free, -1.0)
        index = scores.argmax(dim=1)
        food = torch.stack([index // inner + 1, index % inner + 1], dim=1)
        self.food = torch.where(mask.view(-1, 1), food, self.food)


def test_classic_default_env_is_bit_identical_to_the_legacy_env():
    kwargs = {"num_envs": 16, "board_size": 8, "starvation_steps": 24, "seed": 7}
    legacy, current = _LegacySnakeEnv(**kwargs), TorchSnakeEnv(**kwargs)
    assert torch.equal(legacy.reset(), current.reset())
    episodes = 0
    for step_actions in _random_actions(600, 16, seed=3):
        old, new = legacy.step(step_actions), current.step(step_actions)
        for a, b in zip(old[:4], new[:4], strict=True):
            assert torch.equal(a, b)
        assert all(torch.equal(old[4][key], new[4][key]) for key in old[4])
        assert torch.equal(legacy.food, current.food)
        episodes += int(old[4]["done"].sum())
    assert episodes > 50  # the run crossed many deaths, starvations and food respawns
    assert torch.equal(legacy.generator.get_state(), current.generator.get_state())


# ------------------------------------------------------------ arena, goal and end causes
def test_end_reasons_name_wall_body_and_starvation():
    env = _env(starvation_steps=2)
    env.place_snake(0, [(1, 4)], UP, food=(6, 6))
    assert _act(env, STRAIGHT)[4]["end_reason"].tolist() == [END_WALL]
    env.place_snake(0, [(4, 4), (4, 5), (5, 5), (5, 4), (5, 3)], LEFT, food=(1, 1))
    assert _act(env, TURN_LEFT)[4]["end_reason"].tolist() == [END_BODY]
    env.place_snake(0, [(4, 4)], UP, food=(6, 6))
    assert _act(env, STRAIGHT)[4]["end_reason"].tolist() == [END_NONE]
    assert _act(env, STRAIGHT)[4]["end_reason"].tolist() == [END_STARVED]
    assert END_REASONS[END_GOAL - 1] == "goal"


def test_arena_walls_off_everything_outside_a_centred_square():
    env = _env(num_envs=2, board_size=10)
    env.set_curriculum(6, None)
    env.reset()
    assert (env.arena_size, env.goal_length) == (6, None)
    assert env.head.tolist() == [[5, 5], [5, 5]]  # arena rows/cols 2..7, centre 2 + 6 // 2
    walls = env.render_grid()[0, 3]
    assert walls[3:7, 3:7].sum().item() == 0
    assert walls.sum().item() == 100 - 16
    for _ in range(20):
        env.reset()
        assert ((env.food >= 3) & (env.food <= 6)).all()


def test_arena_border_kills_like_the_outer_wall():
    env = _env(board_size=10)
    env.set_curriculum(6, None)
    env.place_snake(0, [(3, 5)], UP, food=(6, 6))
    _, reward, terminated, _, info = _act(env, STRAIGHT)  # into row 2, the arena wall
    assert terminated.tolist() == [True] and reward.item() == -3.0
    assert info["end_reason"].tolist() == [END_WALL]


def test_full_board_arena_spawns_where_the_plain_game_does():
    env = _env(board_size=9)
    env.set_curriculum(7, None)
    env.set_curriculum(9, None)
    env.reset()
    assert env.head.tolist() == [[9 // 2, 9 // 2]]
    assert env.render_grid()[0, 3].sum().item() == 9 * 4 - 4


def test_reaching_the_goal_length_truncates_without_a_terminal_state():
    env = _env()
    env.set_curriculum(8, 2)
    env.place_snake(0, [(4, 4)], UP, food=(3, 4))
    _, reward, terminated, truncated, info = _act(env, STRAIGHT)
    assert truncated.tolist() == [True] and terminated.tolist() == [False]
    assert reward.item() == 1.0  # the apple, and nothing for the goal
    assert info["end_reason"].tolist() == [END_GOAL]
    assert info["episode_score"].tolist() == [1.0]
    assert env.snake_len.tolist() == [1]  # auto-reset


def test_set_curriculum_rejects_impossible_arenas_and_goals():
    env = _env()
    with pytest.raises(ValueError, match="arena_size must be between 5 and 8"):
        env.set_curriculum(4, None)
    with pytest.raises(ValueError, match="arena_size must be between 5 and 8"):
        env.set_curriculum(9, None)
    with pytest.raises(ValueError, match="goal_length must be at least 2"):
        env.set_curriculum(8, 1)


def test_env_uses_the_reward_strategy_it_is_given():
    env = _env(reward=SparseReward(apple=5.0))
    env.place_snake(0, [(4, 4)], UP, food=(1, 4))
    assert _act(env, STRAIGHT)[1].item() == 0.0
    env.place_snake(0, [(4, 4)], UP, food=(3, 4))
    assert _act(env, STRAIGHT)[1].item() == 5.0
