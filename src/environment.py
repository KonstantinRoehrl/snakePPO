import enum

import torch

import config as cfg


class Reward(enum.Enum):
    EAT_APPLE = 1.
    DIE = -3.
    PROXIMITY_REWARD = 0.01
    PROXIMITY_DISCOUNT = 0.015


class TorchSnakeEnv:
    def __init__(self, size=cfg.GAME_SIZE, num_envs=cfg.NUM_ENVS, device=cfg.DEVICE):
        self.size = size
        self.num_envs = num_envs
        self.device = device
        cfg.DEVICE = device
        
        # Maximum possible snake length (only playable area)
        playable_size = size - 2
        self.max_len = playable_size * playable_size
        
        # Pre-allocate constant tensors to avoid repeated creation
        self.dirs = torch.tensor(
            [[-1, 0], [0, 1], [1, 0], [0, -1]],
            device=self.device,
            dtype=torch.long
        )
        self.env_indices = torch.arange(self.num_envs, device=self.device)
        self.max_len_range = torch.arange(self.max_len, device=self.device)
        
        # Pre-compute reward values as tensors
        self.reward_eat = torch.tensor(Reward.EAT_APPLE.value, device=device)
        self.reward_die = torch.tensor(Reward.DIE.value, device=device)
        self.reward_prox = torch.tensor(Reward.PROXIMITY_REWARD.value, device=device)
        self.reward_disc = torch.tensor(Reward.PROXIMITY_DISCOUNT.value, device=device)
        
        # Pre-compute walls channel (constant across all envs and steps)
        self.walls_channel = torch.zeros((self.num_envs, self.size, self.size), device=self.device)
        self.walls_channel[:, 0, :] = 1
        self.walls_channel[:, -1, :] = 1
        self.walls_channel[:, :, 0] = 1
        self.walls_channel[:, :, -1] = 1
        
        self.reset()

    def reset(self):
        """Initialize all environments to starting state"""
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.score = torch.zeros(self.num_envs, device=self.device)
        self.snake_len = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        center = self.size // 2

        # Head position (inside walls)
        self.head = torch.full((self.num_envs, 2), center, device=self.device, dtype=torch.long)
        
        # Initial direction: UP (-1, 0)
        self.direction = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.long)
        self.direction[:, 0] = -1

        # Body grid: (num_envs, size, size) - each env has its own grid
        self.body = torch.zeros((self.num_envs, self.size, self.size), device=self.device, dtype=torch.bool)
        self.body[:, center, center] = True

        # Body segment history for tail tracking
        self.body_positions = torch.full(
            (self.num_envs, self.max_len, 2), 
            -1, 
            device=self.device, 
            dtype=torch.long
        )
        self.body_positions[:, 0] = self.head.clone()

        # Initialize food positions
        self.food = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self.spawn_food()
        
        return self.get_obs()

    def spawn_food(self, env_mask=None):
        """Spawn food in empty cells for specified environments (batched, no loops)"""
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Early exit if no envs need food
        if not env_mask.any():
            return

        # Playable area mask (exclude walls)
        playable = self.body[:, 1:self.size-1, 1:self.size-1]
        free_mask = ~playable

        E, H, W = free_mask.shape
        HW = H * W

        # Flatten spatial dims
        free_flat = free_mask.view(E, HW)

        # Assign random scores to free cells only
        rand = torch.rand((E, HW), device=self.device)
        rand.masked_fill_(~free_flat, -1.0)

        # Select best free index per env
        best_idx = rand.argmax(dim=1)

        # Convert flat index → (x, y) using floor division
        y = best_idx % W
        x = best_idx // W

        # Offset because playable area starts at index 1
        new_food = torch.stack([x + 1, y + 1], dim=1)

        # Detect envs with NO free space (fallback case)
        has_free = free_flat.any(dim=1)

        # Only generate fallback for envs that need it
        fallback = torch.randint(1, self.size - 1, (self.num_envs, 2), device=self.device)

        # Choose free-cell spawn if possible, else fallback random
        chosen_food = torch.where(has_free.unsqueeze(1), new_food, fallback)

        # Apply only to requested envs
        self.food = torch.where(env_mask.unsqueeze(1), chosen_food, self.food)

    def step(self, actions):
        rewards = torch.zeros(self.num_envs, device=self.device)

        alive = ~self.done

        # Early exit if all envs are done
        if not alive.any():
            obs, rotated_obs = self.get_obs()
            return obs, rotated_obs, rewards, self.done

        # -------- TURN LOGIC (VECTORIZED) --------
        # Use pre-computed dirs tensor
        matches = (self.direction.unsqueeze(1) == self.dirs).all(dim=2)
        dir_idx = matches.long().argmax(dim=1)

        turn_left = (actions == 1) & alive
        turn_right = (actions == 2) & alive

        dir_idx = torch.where(turn_left, (dir_idx - 1) % 4, dir_idx)
        dir_idx = torch.where(turn_right, (dir_idx + 1) % 4, dir_idx)

        self.direction = self.dirs[dir_idx]

        # -------- MOVE HEAD --------
        new_head = self.head + self.direction
        new_head = torch.where(alive.unsqueeze(1), new_head, self.head)

        # -------- DISTANCE REWARD --------
        # Compute distance more efficiently
        head_diff = (self.head - self.food).float()
        new_head_diff = (new_head - self.food).float()
        
        old_dist_sq = (head_diff * head_diff).sum(dim=1)
        new_dist_sq = (new_head_diff * new_head_diff).sum(dim=1)
        
        # Use squared distance to avoid sqrt
        dist_delta_sq = old_dist_sq - new_dist_sq
        
        # Scale rewards appropriately for squared distance
        pos_reward = torch.clamp(dist_delta_sq, min=0.0) * self.reward_prox
        neg_reward = torch.clamp(dist_delta_sq, max=0.0) * self.reward_disc
        rewards += (pos_reward + neg_reward) * alive.float()

        # -------- WALL COLLISION --------
        collision_wall = (
            (new_head[:, 0] <= 0) |
            (new_head[:, 0] >= self.size - 1) |
            (new_head[:, 1] <= 0) |
            (new_head[:, 1] >= self.size - 1)
        ) & alive

        # -------- BODY COLLISION (FULLY VECTORIZED) --------
        nh_x = new_head[:, 0].clamp(0, self.size - 1)
        nh_y = new_head[:, 1].clamp(0, self.size - 1)

        collision_body = self.body[self.env_indices, nh_x, nh_y] & alive & ~collision_wall

        died = (collision_wall | collision_body) & alive

        rewards[died] = self.reward_die
        self.done |= died

        still_alive = ~self.done

        # -------- FOOD EAT --------
        ate = (new_head == self.food).all(dim=1) & still_alive

        rewards[ate] = self.reward_eat
        self.score[ate] += 1

        # -------- UPDATE HEAD --------
        self.head = torch.where(still_alive.unsqueeze(1), new_head, self.head)

        # -------- UPDATE LENGTH --------
        self.snake_len += ate.long()

        # -------- SHIFT BODY POSITIONS (NO LOOPS) --------
        # Shift positions down by 1
        shifted = self.body_positions.roll(1, dims=1)
        shifted[:, 0] = self.head

        # Compute keep mask
        keep_mask = self.max_len_range.unsqueeze(0) < self.snake_len.unsqueeze(1)

        self.body_positions = torch.where(
            still_alive.unsqueeze(1).unsqueeze(2),
            shifted,
            self.body_positions
        )

        # -------- REBUILD BODY GRID VIA SCATTER --------
        self.body.zero_()

        flat_env = self.env_indices.unsqueeze(1).expand_as(self.body_positions[..., 0])
        coords = self.body_positions

        valid_mask = (
            keep_mask &
            still_alive.unsqueeze(1) &
            (coords[..., 0] > 0) &
            (coords[..., 0] < self.size - 1) &
            (coords[..., 1] > 0) &
            (coords[..., 1] < self.size - 1)
        )

        env_ids = flat_env[valid_mask]
        xs = coords[..., 0][valid_mask]
        ys = coords[..., 1][valid_mask]

        self.body[env_ids, xs, ys] = True

        # -------- SPAWN FOOD --------
        if ate.any():
            self.spawn_food(env_mask=ate)

        obs, rotated_obs = self.get_obs()
        return obs, rotated_obs, rewards, self.done

    def get_obs(self):
        """Generate multi-channel observation grid"""
        obs = torch.zeros((self.num_envs, 4, self.size, self.size), device=self.device)
        
        # Channel 0: Snake body
        obs[:, 0] = self.body.float()
        
        # Channel 1: Snake head
        valid_head = (
            (self.head[:, 0] > 0) & 
            (self.head[:, 0] < self.size - 1) & 
            (self.head[:, 1] > 0) & 
            (self.head[:, 1] < self.size - 1) &
            ~self.done
        )
        if valid_head.any():
            valid_indices = valid_head.nonzero(as_tuple=True)[0]
            obs[valid_indices, 1, self.head[valid_head, 0], self.head[valid_head, 1]] = 1
        
        # Channel 2: Food
        obs[self.env_indices, 2, self.food[:, 0], self.food[:, 1]] = 1
        
        # Channel 3: Walls (pre-computed constant)
        obs[:, 3] = self.walls_channel
        
        # Create rotated observations where snake always faces UP
        rotated_obs = self._rotate_obs_to_up(obs)
        
        return obs, rotated_obs
    
    def _rotate_obs_to_up(self, obs):
        """Rotate observations so snake always faces UP (no loops)"""
        # Match each env direction to dirs
        matches = (self.direction.unsqueeze(1) == self.dirs).all(dim=2)
        dir_idx = matches.long().argmax(dim=1)

        # Rotation needed: RIGHT→1, DOWN→2, LEFT→3, UP→0
        k = dir_idx

        # Precompute all 4 rotations of the full batch
        all_rots = torch.stack([
            obs,
            torch.rot90(obs, k=1, dims=(2, 3)),
            torch.rot90(obs, k=2, dims=(2, 3)),
            torch.rot90(obs, k=3, dims=(2, 3))
        ], dim=1)

        # Select correct rotation per env
        rotated_obs = all_rots[self.env_indices, k]

        # Keep original obs if env is done
        rotated_obs = torch.where(
            self.done.view(-1, 1, 1, 1),
            obs,
            rotated_obs
        )

        return rotated_obs