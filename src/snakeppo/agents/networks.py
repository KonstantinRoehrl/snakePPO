"""Feed-forward policy/value networks.

``CNNPolicy``, ``SpatialCNNPolicy`` and ``ResNetPolicy`` keep the pre-C1 agents' layer structure.
"""

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


def conv_features(in_channels: int, width: int) -> nn.Sequential:
    """Three 3x3 convs (``width``, ``2*width`` with stride 2, ``2*width``) pooled to 4x4, flattened.

    Maps (B, C, S, S) to (B, 2 * width * 16) for any S, so a network built on it runs on any board.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, padding=1),
        nn.ReLU(),
        DeterministicAdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
    )


class MLPPolicy(PolicyNetwork):
    """Flatten, two hidden layers of 256. Its first layer is sized for one board, so it is bound."""

    def __init__(self, board_size: int, in_channels: int = 4, num_actions: int = 3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        return self.policy(shared), self.value(shared).squeeze(-1)


class ScalableCNNPolicy(PolicyNetwork):
    """``conv_features`` of one ``width``, then a hidden layer of 256 (cnn-tiny/small/medium)."""

    def __init__(
        self, board_size: int, in_channels: int = 4, num_actions: int = 3, *, width: int
    ) -> None:
        super().__init__()
        del board_size  # the pool in conv_features makes this network board-size independent
        self.features = conv_features(in_channels, width)
        self.shared = nn.Sequential(nn.Linear(2 * width * 16, 256), nn.ReLU())
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(self.features(x))
        return self.policy(shared), self.value(shared).squeeze(-1)


def sinusoidal_positions_2d(
    size: int, dim: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Fixed 2D sine/cosine positions for the cells of a ``size`` x ``size`` grid: (size*size, dim).

    The first ``dim // 2`` dims encode the row and the rest the column, each as sines then cosines
    of ``dim // 4`` frequencies. Rows are in row-major cell order, matching ``Tensor.flatten(2)``.
    """
    if dim % 4:
        raise ValueError(f"the position dimension must be divisible by 4, got {dim}")
    quarter = dim // 4
    frequencies = 10000.0 ** (-torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
    angles = torch.arange(size, device=device, dtype=torch.float32)[:, None] * frequencies
    axis = torch.cat([angles.sin(), angles.cos()], dim=1)  # (size, dim // 2)
    rows = axis[:, None, :].expand(size, size, dim // 2)
    columns = axis[None, :, :].expand(size, size, dim // 2)
    return torch.cat([rows, columns], dim=-1).reshape(size * size, dim)


ATTENTION_WIDTH = 64


class AttentionPolicy(PolicyNetwork):
    """Conv stem, one token per cell plus its 2D position, 2 pre-norm transformer layers, mean pool.

    Positions are computed for the input's size on every call, so the network runs on any board.
    """

    def __init__(self, board_size: int, in_channels: int = 4, num_actions: int = 3) -> None:
        super().__init__()
        del board_size  # positions are computed per input, and the mean pool fixes the width
        self.stem = nn.Sequential(nn.Conv2d(in_channels, ATTENTION_WIDTH, 3, padding=1), nn.ReLU())
        # a ModuleList, not nn.TransformerEncoder, which warns about nested tensors with norm_first
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                ATTENTION_WIDTH,
                nhead=4,
                dim_feedforward=128,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        )
        self.norm = nn.LayerNorm(ATTENTION_WIDTH)
        self.shared = nn.Sequential(nn.Linear(ATTENTION_WIDTH, 256), nn.ReLU())
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.stem(x).flatten(2).transpose(1, 2)  # (B, S*S, D), row-major cells
        tokens = tokens + sinusoidal_positions_2d(x.shape[-1], ATTENTION_WIDTH, x.device)
        for layer in self.layers:
            tokens = layer(tokens)
        shared = self.shared(self.norm(tokens).mean(dim=1))
        return self.policy(shared), self.value(shared).squeeze(-1)
