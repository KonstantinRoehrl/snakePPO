import torch
import torch.nn as nn

from config import GAME_SIZE


class PPOAgent(nn.Module):
    def __init__(self, in_channels=4, actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),  # reduce spatial size
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # reduce spatial size
            nn.AdaptiveAvgPool2d((4, 4)),  # always output 4×4 regardless of input size
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.policy = nn.Linear(256, actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.net(x)
        return self.policy(x), self.value(x)
    
    def sample_action(self, obs):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)

        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values, dist
    
    def get_action_greedy(self, obs):
        """Get greedy (deterministic) action - useful for evaluation
        
        Args:
            obs: Observations (batch_size, channels, height, width)
            
        Returns:
            actions: Greedy actions (argmax)
            values: Value estimates
        """
        with torch.no_grad():
            logits, values = self.forward(obs)
            actions = logits.argmax(dim=-1)
        return actions, values