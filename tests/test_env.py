import pytest
import torch

from snakeppo.env import DOWN, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, TorchSnakeEnv


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
    env = _env(reward_approach=0.5, reward_retreat=0.25)
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
