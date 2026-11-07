"""Feed-forward policy/value networks. Layer structure is unchanged from the pre-C1 agents."""

from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Maps an observation batch (B, C, S, S) to ``(logits (B, A), value (B,))``."""

    def act(
        self, obs: torch.Tensor, greedy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick actions; returns ``(actions, log_probs, values)``, each of shape (B,)."""
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if greedy else dist.sample()
        return actions, dist.log_prob(actions), value


def _pooling_bins(size: int, out: int) -> list[tuple[int, int]]:
    """Start/end of each pooling window, using the same boundaries as torch's adaptive pool."""
    return [(i * size // out, -(-(i + 1) * size // out)) for i in range(out)]


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """``nn.AdaptiveAvgPool2d`` built from slices and means.

    The built-in layer has no deterministic CUDA backward, so it breaks ``train.deterministic``.
    Rows are averaged first, then columns; every window is a rectangle, so the result is the same.
    """

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_h, out_w = self.output_size
        rows = torch.stack(
            [x[:, :, start:end, :].mean(dim=2) for start, end in _pooling_bins(x.shape[2], out_h)],
            dim=2,
        )
        return torch.stack(
            [rows[..., start:end].mean(dim=-1) for start, end in _pooling_bins(x.shape[3], out_w)],
            dim=-1,
        )


class CNNPolicy(PolicyNetwork):
    """Former ``PPOAgent``."""

    def __init__(self, board_size: int, in_channels: int = 4, num_actions: int = 3) -> None:
        super().__init__()
        del board_size  # the adaptive pool makes this network board-size independent
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            DeterministicAdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        return self.policy(x), self.value(x).squeeze(-1)


class SpatialCNNPolicy(PolicyNetwork):
    """Former ``SpatialAwarePPOAgent``: one stride-2 conv keeps half-resolution features."""

    def __init__(self, board_size: int, in_channels: int = 4, num_actions: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        spatial = board_size // 2
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial * spatial, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(self.features(x))
        return self.policy(shared), self.value(shared).squeeze(-1)


class ResNetPolicy(PolicyNetwork):
    """Former ``DeepPPOAgent``: residual conv blocks with projection shortcuts."""

    def __init__(self, board_size: int, in_channels: int = 4, num_actions: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1)
        )
        self.downsample1 = nn.Conv2d(64, 128, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        self.downsample2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        spatial = board_size // 2
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial * spatial, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out) + self.downsample1(out))
        out = self.relu(self.conv3(out) + self.downsample2(out))
        shared = self.shared(out)
        return self.policy(shared), self.value(shared).squeeze(-1)
