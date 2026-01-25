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
        # Walls are at borders (index 0 and size-1)
        self.body = torch.zeros((self.num_envs, self.size, self.size), device=self.device, dtype=torch.bool)
        self.body[:, center, center] = True

        # Body segment history for tail tracking (stores positions from head to tail)
        # Position 0 is the head, position 1 is next segment, etc.
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
        """Spawn food in empty cells for specified environments (inside walls)
        
        Args:
            env_mask: Boolean mask of environments to spawn food for. 
                     If None, spawn for all environments.
        """
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        env_indices = torch.where(env_mask)[0]
        
        for env_idx in env_indices:
            # Find all free cells in playable area (not occupied by snake body or walls)
            # Playable area is from index 1 to size-2 (inclusive)
            playable_area = self.body[env_idx, 1:self.size-1, 1:self.size-1]
            free_mask = ~playable_area  # Shape: (size-2, size-2)
            free_coords = torch.nonzero(free_mask, as_tuple=False)  # Shape: (num_free, 2)
            
            if free_coords.shape[0] > 0:
                # Randomly select one free cell
                idx = torch.randint(0, free_coords.shape[0], (1,), device=self.device).item()
                # Add 1 to coordinates to account for wall offset
                self.food[env_idx] = free_coords[idx] + 1
            else:
                # Fallback: random position in playable area
                self.food[env_idx] = torch.randint(1, self.size - 1, (2,), device=self.device)

    def step(self, actions):
        """Execute one step for all environments"""
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Cardinal directions: up, right, down, left
        dirs = torch.tensor(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], 
            device=self.device, 
            dtype=self.direction.dtype
        )

        # Only update alive environments
        alive = ~self.done
        
        # Process turns for alive environments
        turn_left = (actions == 1) & alive
        turn_right = (actions == 2) & alive

        if turn_left.any():
            cur = self.direction[turn_left]
            matches = (cur.unsqueeze(1) == dirs.unsqueeze(0)).all(dim=2)
            idx = matches.long().argmax(dim=1)
            new_idx = (idx - 1) % 4
            self.direction[turn_left] = dirs[new_idx]

        if turn_right.any():
            cur = self.direction[turn_right]
            matches = (cur.unsqueeze(1) == dirs.unsqueeze(0)).all(dim=2)
            idx = matches.long().argmax(dim=1)
            new_idx = (idx + 1) % 4
            self.direction[turn_right] = dirs[new_idx]

        # Compute new head position (only for alive)
        new_head = self.head.clone()
        new_head[alive] = self.head[alive] + self.direction[alive]

        # Distance to food BEFORE move
        old_dist = torch.sqrt(
            (self.head[:, 0] - self.food[:, 0]).float() ** 2 +
            (self.head[:, 1] - self.food[:, 1]).float() ** 2
        )

        # Distance to food AFTER move
        new_dist = torch.sqrt(
            (new_head[:, 0] - self.food[:, 0]).float() ** 2 +
            (new_head[:, 1] - self.food[:, 1]).float() ** 2
        )

        # Positive reward when moving closer, slightly stronger penalty when moving away
        dist_delta = old_dist - new_dist


        pos_reward = torch.clamp(dist_delta, min=0.0) * Reward.PROXIMITY_REWARD.value
        neg_reward = torch.clamp(dist_delta, max=0.0) * Reward.PROXIMITY_DISCOUNT.value

        prox_reward = pos_reward + neg_reward
        prox_reward = prox_reward * alive.float()

        rewards += prox_reward

        # Check wall collision (hitting border at 0 or size-1)
        collision_wall = (
            (new_head[:, 0] <= 0) | 
            (new_head[:, 0] >= self.size - 1) | 
            (new_head[:, 1] <= 0) | 
            (new_head[:, 1] >= self.size - 1)
        ) & alive

        # Check body collision - vectorized using body grid
        collision_body = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        valid = alive & ~collision_wall
        if valid.any():
            valid_indices = torch.where(valid)[0]
            collision_body[valid] = self.body[
                valid_indices,
                new_head[valid, 0],
                new_head[valid, 1]
            ]

        # Determine which snakes died this step
        died = (collision_wall | collision_body) & alive

        # Apply death penalty
        rewards[died] = Reward.DIE.value
        self.done[died] = True

        # Update alive mask after deaths
        still_alive = ~self.done

        # Check if food was eaten (before updating position)
        ate = (new_head == self.food).all(dim=1) & still_alive
        rewards[ate] = Reward.EAT_APPLE.value
        self.score[ate] += 1

        # Update head position for alive snakes
        self.head[still_alive] = new_head[still_alive]

        # Update snake body for alive snakes
        for env_idx in torch.where(still_alive)[0]:
            if ate[env_idx]:
                # Snake grows - increment length
                self.snake_len[env_idx] += 1
                
                # Shift body positions and add new head
                old_positions = self.body_positions[env_idx].clone()
                self.body_positions[env_idx, 0] = self.head[env_idx]
                length = self.snake_len[env_idx] - 1  # Old length before growth
                self.body_positions[env_idx, 1:length+1] = old_positions[:length]
            else:
                # Snake moves - shift body positions
                old_positions = self.body_positions[env_idx].clone()
                self.body_positions[env_idx, 0] = self.head[env_idx]
                length = self.snake_len[env_idx]
                if length > 1:
                    self.body_positions[env_idx, 1:length] = old_positions[:length-1]
            
            # Rebuild body grid from positions
            self.body[env_idx] = False
            length = self.snake_len[env_idx]
            valid_positions = self.body_positions[env_idx, :length]
            # Ensure positions are within playable area (not on walls)
            valid_mask = (
                (valid_positions[:, 0] > 0) & 
                (valid_positions[:, 0] < self.size - 1) &
                (valid_positions[:, 1] > 0) & 
                (valid_positions[:, 1] < self.size - 1)
            )
            valid_positions = valid_positions[valid_mask]
            
            if valid_positions.shape[0] > 0:
                self.body[env_idx, valid_positions[:, 0], valid_positions[:, 1]] = True

        # Spawn new food only for environments that ate
        if ate.any():
            self.spawn_food(env_mask=ate)

        obs, rotated_obs = self.get_obs()
        return obs, rotated_obs, rewards, self.done

    def get_obs(self):
        """Generate multi-channel observation grid"""
        obs = torch.zeros((self.num_envs, 4, self.size, self.size), device=self.device)
        
        # Channel 0: Snake body (directly from body grid)
        obs[:, 0] = self.body.float()
        
        # Channel 1: Snake head (only for alive snakes with valid positions inside walls)
        valid_head = (
            (self.head[:, 0] > 0) & 
            (self.head[:, 0] < self.size - 1) & 
            (self.head[:, 1] > 0) & 
            (self.head[:, 1] < self.size - 1) &
            ~self.done
        )
        if valid_head.any():
            valid_indices = torch.where(valid_head)[0]
            obs[valid_indices, 1, self.head[valid_head, 0], self.head[valid_head, 1]] = 1
        
        # Channel 2: Food (each env has its own food position, guaranteed inside walls)
        env_indices = torch.arange(self.num_envs, device=self.device)
        obs[env_indices, 2, self.food[:, 0], self.food[:, 1]] = 1
        
        # Channel 3: Walls (border)
        obs[:, 3, 0, :] = 1
        obs[:, 3, -1, :] = 1
        obs[:, 3, :, 0] = 1
        obs[:, 3, :, -1] = 1
        
        # Create rotated observations where snake always faces UP
        rotated_obs = self._rotate_obs_to_up(obs)
        
        return obs, rotated_obs
    
    def _rotate_obs_to_up(self, obs):
        """Rotate observations so snake always faces UP
        
        Args:
            obs: Original observations (num_envs, 4, size, size)
            
        Returns:
            Rotated observations where snake direction is normalized to UP
        """
        # Cardinal directions: up, right, down, left
        dirs = torch.tensor(
            [[-1, 0], [0, 1], [1, 0], [0, -1]], 
            device=self.device, 
            dtype=self.direction.dtype
        )
        
        rotated_obs = obs.clone()
        
        for env_idx in range(self.num_envs):
            if self.done[env_idx]:
                continue
                
            # Find current direction index
            cur_dir = self.direction[env_idx]
            matches = (cur_dir.unsqueeze(0) == dirs).all(dim=1)
            dir_idx = matches.long().argmax().item()
            
            # Rotation needed: 0=UP (no rotation), 1=RIGHT (rotate 90° CCW), 
            # 2=DOWN (rotate 180°), 3=LEFT (rotate 90° CW)
            if dir_idx == 0:  # Already facing UP
                continue
            elif dir_idx == 1:  # Facing RIGHT, rotate 90° CCW (k=1)
                rotated_obs[env_idx] = torch.rot90(obs[env_idx], k=1, dims=(1, 2))
            elif dir_idx == 2:  # Facing DOWN, rotate 180° (k=2)
                rotated_obs[env_idx] = torch.rot90(obs[env_idx], k=2, dims=(1, 2))
            elif dir_idx == 3:  # Facing LEFT, rotate 90° CW (k=3 or -1)
                rotated_obs[env_idx] = torch.rot90(obs[env_idx], k=3, dims=(1, 2))
        
        return rotated_obs