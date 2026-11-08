"""Vectorized Snake on a square board with a wall ring, running entirely in torch.

Every env auto-resets: a snake that dies (terminated), starves or reaches its goal length
(truncated) is reset inside the same ``step`` call, and ``info`` carries the pre-reset final
observation and the finished episode's statistics. A curriculum can shrink the playable area to a
centred walled arena and set a goal length (:meth:`TorchSnakeEnv.set_curriculum`); the tensors
keep the full board size either way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from snakeppo.rewards import ClassicReward, RewardStrategy, StepContext

# Heading index -> (row, col) delta. Index order is clockwise, so +1 turns right.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
UP, RIGHT, DOWN, LEFT = range(4)
STRAIGHT, TURN_LEFT, TURN_RIGHT = range(3)
NUM_ACTIONS = 3
NUM_CHANNELS = 4  # body, head, food, wall ring
# Why an episode ended, per env in ``info["end_reason"]``; END_NONE where it did not end.
END_NONE, END_WALL, END_BODY, END_STARVED, END_GOAL = range(5)
END_REASONS = ("wall", "body", "starved", "goal")  # END_REASONS[code - 1]
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
        reward: RewardStrategy | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.board_size = board_size
        self.starvation_steps = starvation_steps
        self.reward = ClassicReward() if reward is None else reward
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
        self.arena_size = board_size
        self.goal_length: int | None = None
        self._build_arena()

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

    def set_curriculum(self, arena_size: int, goal_length: int | None) -> None:
        """Play inside a centred ``arena_size`` square (walls included), ending at ``goal_length``.

        Walls, collisions, the spawn point and food placement follow at once; call
        :meth:`reset` afterwards. ``set_curriculum(board_size, None)`` is the plain game.
        """
        if not 5 <= arena_size <= self.board_size:
            raise ValueError(
                f"arena_size must be between 5 and {self.board_size}, got {arena_size}"
            )
        if goal_length is not None and goal_length < 2:
            raise ValueError(f"goal_length must be at least 2 or None, got {goal_length}")
        self.arena_size, self.goal_length = arena_size, goal_length
        self._build_arena()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Advance every env by one relative action (0 straight, 1 left, 2 right).

        Returns ``(obs, reward, terminated, truncated, info)``. ``terminated`` means the snake hit
        a wall or itself; ``truncated`` means it starved or reached the goal length, so the
        episode ends without a terminal state and its value is bootstrapped. ``obs`` is already
        the first observation of the next episode for envs that finished. ``info`` holds
        ``done``, ``final_obs`` (pre-reset observation; equal to ``obs`` where not done),
        ``episode_score``, ``episode_length``, ``episode_return`` and ``end_reason`` (an
        ``END_*`` code; all valid where done).
        """
        actions = actions.to(self.device)
        turn = torch.where(actions == TURN_LEFT, -1, torch.where(actions == TURN_RIGHT, 1, 0))
        self.heading = (self.heading + turn) % 4
        old_head, old_food = self.head, self.food
        new_head = old_head + self._dirs[self.heading]

        hit_wall = (new_head <= self._arena_low).any(dim=1) | (new_head >= self._arena_high).any(
            dim=1
        )
        clamped = new_head.clamp(0, self.board_size - 1)
        hit_body = self.body[self._env_ids, clamped[:, 0], clamped[:, 1]] & ~hit_wall
        terminated = hit_wall | hit_body
        alive = ~terminated

        ate = (new_head == self.food).all(dim=1) & alive
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
        starved = alive & (self.steps_since_food >= self.starvation_steps)
        reached_goal = torch.zeros_like(ate)
        if self.goal_length is not None:
            reached_goal = ate & (self.snake_len >= self.goal_length)
        truncated = starved | reached_goal
        done = terminated | truncated
        reward = self.reward(
            StepContext(
                old_head=old_head,
                new_head=new_head,
                head=self.head,
                old_food=old_food,
                food=self.food,
                ate=ate,
                terminated=terminated,
                starved=starved,
                steps_since_food=self.steps_since_food,
                starvation_steps=self.starvation_steps,
                body=self.body,
                playable=self._playable,
            )
        )
        self.episode_length += 1
        self.episode_return += reward

        end_reason = torch.zeros_like(self.snake_len)
        for code, mask in (
            (END_WALL, hit_wall),
            (END_BODY, hit_body),
            (END_STARVED, starved),
            (END_GOAL, reached_goal),
        ):
            end_reason = torch.where(mask, code, end_reason)
        obs = self.observe()
        info = {
            "done": done,
            "final_obs": obs,
            "episode_score": self.score.clone(),
            "episode_length": self.episode_length.clone(),
            "episode_return": self.episode_return.clone(),
            "end_reason": end_reason,
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
    def _build_arena(self) -> None:
        """Wall channel, playable mask and collision bounds of the current arena."""
        low = (self.board_size - self.arena_size) // 2
        self._arena_low, self._arena_high = low, low + self.arena_size - 1
        self._playable = torch.zeros(
            (self.board_size, self.board_size), dtype=torch.bool, device=self.device
        )
        self._playable[low + 1 : self._arena_high, low + 1 : self._arena_high] = True
        self._walls = (~self._playable).float()
        self._food_cells = self._playable[1:-1, 1:-1].reshape(-1)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        center = self._arena_low + self.arena_size // 2
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
        # draw over the whole board interior and mask the arena, so the draw never changes
        free = ~self.body[:, 1:-1, 1:-1].reshape(self.num_envs, -1) & self._food_cells
        scores = torch.rand(free.shape, device=self.device, generator=self.generator)
        scores = scores.masked_fill(~free, -1.0)
        index = scores.argmax(dim=1)
        food = torch.stack([index // inner + 1, index % inner + 1], dim=1)
        self.food = torch.where(mask.view(-1, 1), food, self.food)
