import torch
import torch.nn as nn

from config import GAME_SIZE


class BasePPOAgent(nn.Module):
    """Base class for PPO agents with shared sampling logic"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Must be implemented by subclasses.
        Returns: (logits, values) or (logits, values, hidden_state) for recurrent agents
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def sample_action(self, obs):
        """
        Sample actions from the policy distribution.
        Works for both feedforward and recurrent agents.
        """
        forward_output = self.forward(obs)
        
        # Handle both feedforward (logits, values) and recurrent (logits, values, hidden)
        if isinstance(forward_output, tuple) and len(forward_output) == 2:
            logits, values = forward_output
        else:
            # Recurrent case - extract first two elements
            logits, values = forward_output[0], forward_output[1]
        
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs, values, dist


class PPOAgent(BasePPOAgent):
    """Standard feedforward CNN-based PPO agent"""
    
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


class SpatialAwarePPOAgent(BasePPOAgent):
    """PPO agent with less spatial compression for better spatial reasoning"""
    
    def __init__(self, in_channels=4, actions=3, grid_size=GAME_SIZE):
        super().__init__()
        
        # Less aggressive downsampling - only one stride-2 conv
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            # Only reduce spatial size once, keep it at 8x8 for 16x16 input
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        
        spatial_size = grid_size // 2  # After one stride-2 conv
        
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial_size * spatial_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        
        self.policy = nn.Linear(256, actions)
        self.value = nn.Linear(256, 1)
    
    def forward(self, x):
        features = self.features(x)
        shared = self.shared(features)
        return self.policy(shared), self.value(shared)


class RecurrentPPOAgent(BasePPOAgent):
    """PPO agent with LSTM for temporal memory (returns only logits, values)"""
    
    def __init__(self, in_channels=4, actions=3, grid_size=GAME_SIZE):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        
        spatial_size = grid_size // 2
        conv_output_size = 256 * spatial_size * spatial_size
        
        # LSTM for temporal memory
        self.lstm = nn.LSTM(conv_output_size, 512, num_layers=2, batch_first=True)
        self.hidden = None  # internal hidden state
        
        self.policy = nn.Linear(512, actions)
        self.value = nn.Linear(512, 1)
    
    def forward(self, x):
        """
        Forward pass returns only (policy_logits, value) to match BasePPOAgent.
        Hidden state is stored internally.
        """
        batch_size = x.size(0)
        features = self.conv(x).flatten(1)  # (batch, features)
        
        # Ensure hidden state matches batch size
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.reset_hidden(batch_size, x.device)
        
        # Add sequence dim
        features_seq = features.unsqueeze(1)  # (batch, seq_len=1, features)
        
        lstm_out, new_hidden = self.lstm(features_seq, self.hidden)
        lstm_out = lstm_out.squeeze(1)  # (batch, hidden_size)
        
        # Update internal hidden state
        self.hidden = new_hidden
        
        # Return only policy and value
        return self.policy(lstm_out), self.value(lstm_out)
    
    def reset_hidden(self, batch_size=None, device=None):
        """Reset LSTM hidden state (call at episode boundaries)"""
        if batch_size is not None and device is not None:
            self.hidden = (
                torch.zeros(2, batch_size, 512, device=device),  # h_0
                torch.zeros(2, batch_size, 512, device=device)   # c_0
            )
        else:
            self.hidden = None
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        features = self.conv(x).flatten(1)
        
        # Use stored hidden state if not provided
        if hidden is None:
            hidden = self.hidden
        
        # Process through LSTM
        lstm_out, new_hidden = self.lstm(features.unsqueeze(1), hidden)
        lstm_out = lstm_out.squeeze(1)
        
        # Store hidden state for next forward pass
        self.hidden = new_hidden
        
        return self.policy(lstm_out), self.value(lstm_out)
    
    def reset_hidden(self, batch_size=None, device=None):
        """Reset LSTM hidden state (call at episode boundaries)"""
        if batch_size is not None and device is not None:
            self.hidden = (
                torch.zeros(2, batch_size, 512, device=device),
                torch.zeros(2, batch_size, 512, device=device)
            )
        else:
            self.hidden = None


class DeepPPOAgent(BasePPOAgent):
    """Deeper PPO agent with residual connections"""
    
    def __init__(self, in_channels=4, actions=3, grid_size=GAME_SIZE):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        self.downsample1 = nn.Conv2d(64, 128, 1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        self.downsample2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        spatial_size = grid_size // 2
        
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial_size * spatial_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.policy = nn.Linear(256, actions)
        self.value = nn.Linear(256, 1)
    
    def forward(self, x):
        # Residual block 1
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        
        # Residual block 2
        identity = out
        out = self.conv2(out)
        out += self.downsample1(identity)
        out = self.relu2(out)
        
        # Residual block 3
        identity = out
        out = self.conv3(out)
        out += self.downsample2(identity)
        out = self.relu3(out)
        
        shared = self.shared(out)
        return self.policy(shared), self.value(shared)