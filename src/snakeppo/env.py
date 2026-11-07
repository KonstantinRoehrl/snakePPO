"""Vectorized Snake on a square board with a wall ring, running entirely in torch.

Every env auto-resets: a snake that dies (terminated) or starves (truncated) is reset inside
the same ``step`` call, and ``info`` carries the pre-reset final observation and the finished
episode's statistics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

# Heading index -> (row, col) delta. Index order is clockwise, so +1 turns right.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
UP, RIGHT, DOWN, LEFT = range(4)
STRAIGHT, TURN_LEFT, TURN_RIGHT = range(3)
NUM_ACTIONS = 3
NUM_CHANNELS = 4  # body, head, food, wall ring
# Every attribute ``step`` and ``_reset_envs`` reassign; ``state_dict`` saves exactly these.
STATE_TENSORS = (
    "head",
    "heading",
    "body",
    "body_positions",
    "snake_len",
    "food",
    "score",
    "steps_since_food",
    "episode_length",
    "episode_return",
)


class TorchSnakeEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int = 16,
        *,
        starvation_steps: int = 512,
        reward_apple: float = 1.0,
        reward_death: float = -3.0,
        reward_approach: float = 0.02,
        reward_retreat: float = 0.03,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.board_size = board_size
        self.starvation_steps = starvation_steps
        self.reward_apple = reward_apple
        self.reward_death = reward_death
        self.reward_approach = reward_approach
        self.reward_retreat = reward_retreat
        self.device = torch.device(device)
        self.obs_shape = (NUM_CHANNELS, board_size, board_size)
        self.max_len = (board_size - 2) ** 2

        self.generator = torch.Generator(device=self.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)

        self._dirs = torch.tensor(DIRECTIONS, device=self.device, dtype=torch.long)
        self._env_ids = torch.arange(num_envs, device=self.device)
        self._len_range = torch.arange(self.max_len, device=self.device)
        self._walls = torch.zeros((board_size, board_size), device=self.device)
        self._walls[0, :] = self._walls[-1, :] = self._walls[:, 0] = self._walls[:, -1] = 1

        n, s = num_envs, board_size
        self.head = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.heading = torch.zeros(n, dtype=torch.long, device=self.device)
        self.body = torch.zeros((n, s, s), dtype=torch.bool, device=self.device)
        self.body_positions = torch.full((n, self.max_len, 2), -1, device=self.device)
        self.snake_len = torch.ones(n, dtype=torch.long, device=self.device)
        self.food = torch.zeros((n, 2), dtype=torch.long, device=self.device)
        self.score = torch.zeros(n, device=self.device)
        self.steps_since_food = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_length = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(n, device=self.device)
        self.reset()

    # ----------------------------------------------------------------- public API
    def reset(self) -> torch.Tensor:
        """Reset every env; returns the policy observation (N, 4, S, S) float32."""
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.observe()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Advance every env by one relative action (0 straight, 1 left, 2 right).

        Returns ``(obs, reward, terminated, truncated, info)``. ``obs`` is already the first
        observation of the next episode for envs that finished. ``info`` holds ``done``,
        ``final_obs`` (pre-reset observation; equal to ``obs`` where not done),
        ``episode_score``, ``episode_length`` and ``episode_return`` (valid where done).
        """
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

    def observe(self) -> torch.Tensor:
        """Policy observation: the grid rotated so every snake faces up."""
        grid = self.render_grid()
        rotations = torch.stack([torch.rot90(grid, k=k, dims=(2, 3)) for k in range(4)], dim=1)
        return rotations[self._env_ids, self.heading]

    def render_grid(self) -> torch.Tensor:
        """Unrotated grid (N, 4, S, S) float32 for display: body, head, food, walls."""
        grid = torch.zeros((self.num_envs, *self.obs_shape), device=self.device)
        grid[:, 0] = self.body.float()
        grid[self._env_ids, 1, self.head[:, 0], self.head[:, 1]] = 1.0
        grid[self._env_ids, 2, self.food[:, 0], self.food[:, 1]] = 1.0
        grid[:, 3] = self._walls
        return grid

    def place_snake(
        self,
        env_index: int,
        body: Sequence[tuple[int, int]],
        heading: int,
        food: tuple[int, int],
    ) -> None:
        """Put one env into an exact state (head first in ``body``); for tests and demos."""
        self.body_positions[env_index] = -1
        self.body_positions[env_index, : len(body)] = torch.tensor(body, device=self.device)
        self.snake_len[env_index] = len(body)
        self.head[env_index] = torch.tensor(body[0], device=self.device)
        self.heading[env_index] = heading
        self.food[env_index] = torch.tensor(food, device=self.device)
        self._rebuild_body()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Everything ``step`` mutates, as CPU tensors, plus the food generator's state."""
        state = {name: getattr(self, name).detach().cpu().clone() for name in STATE_TENSORS}
        state["generator"] = self.generator.get_state()
        return state

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Restore a :meth:`state_dict` taken from an env with the same ``num_envs`` and board."""
        missing = [name for name in (*STATE_TENSORS, "generator") if name not in state]
        if missing:
            raise ValueError(f"env state is missing {', '.join(missing)}")
        for name in STATE_TENSORS:
            current, saved = getattr(self, name), state[name]
            if saved.shape != current.shape:
                raise ValueError(
                    f"env state '{name}' has shape {tuple(saved.shape)}, this env needs "
                    f"{tuple(current.shape)} (num_envs or board_size differ)"
                )
            setattr(self, name, saved.to(device=self.device, dtype=current.dtype).clone())
        self.generator.set_state(state["generator"])

    # ---------------------------------------------------------------- internals
    def _reset_envs(self, mask: torch.Tensor) -> None:
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

    def _rebuild_body(self) -> None:
        keep = self._len_range.unsqueeze(0) < self.snake_len.unsqueeze(1)
        coords = self.body_positions
        env_ids = self._env_ids.unsqueeze(1).expand_as(keep)
        self.body.zero_()
        self.body[env_ids[keep], coords[..., 0][keep], coords[..., 1][keep]] = True

    def _spawn_food(self, mask: torch.Tensor) -> None:
        inner = self.board_size - 2
        free = ~self.body[:, 1:-1, 1:-1].reshape(self.num_envs, -1)
        scores = torch.rand(free.shape, device=self.device, generator=self.generator)
        scores = scores.masked_fill(~free, -1.0)
        index = scores.argmax(dim=1)
        food = torch.stack([index // inner + 1, index % inner + 1], dim=1)
        self.food = torch.where(mask.view(-1, 1), food, self.food)
