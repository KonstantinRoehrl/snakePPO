"""Recurrent policies, the episode-start helper and the player that carries their state.

State layout: one tensor ``(B, K, H)`` per batch of envs, ``K = 1`` for a GRU (``h``) and ``K = 2``
for an LSTM (``h``, ``c``), so resets, env-subset indexing and checkpoints handle a single tensor.
A state row is zeroed where an episode starts (``starts``), before that step is processed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from snakeppo.agents.networks import PolicyNetwork, conv_features

CELLS = ("gru", "lstm")
RECURRENT_HIDDEN = 256


class RecurrentPolicy(PolicyNetwork, ABC):
    """A policy that carries a hidden state ``(B, K, H)`` from one step to the next."""

    state_slots: int  # K
    hidden_size: int  # H

    def initial_state(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """The zero state for ``batch_size`` fresh episodes: (B, K, H)."""
        return torch.zeros(batch_size, self.state_slots, self.hidden_size, device=device)

    @abstractmethod
    def unroll(
        self, obs: torch.Tensor, state: torch.Tensor, starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run T steps: obs (T, B, C, S, S), state (B, K, H), starts (T, B) bool.

        Returns ``(logits (T, B, A), values (T, B), final_state (B, K, H))``; the state of envs
        whose ``starts[t]`` is true is zeroed before step ``t``.
        """

    def forward(
        self, obs: torch.Tensor, state: torch.Tensor, starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: obs (B, C, S, S), starts (B,) -> ``(logits (B, A), value (B,), new_state)``."""
        logits, values, new_state = self.unroll(obs.unsqueeze(0), state, starts.unsqueeze(0))
        return logits[0], values[0], new_state

    def act(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        starts: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick actions; returns ``(actions, log_probs, values, new_state)``."""
        logits, value, new_state = self(obs, state, starts)
        dist = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if greedy else dist.sample()
        return actions, dist.log_prob(actions), value, new_state


class RecurrentCNNPolicy(RecurrentPolicy):
    """``conv_features(in_channels, 32)`` and a 256 feature, a GRU or LSTM of 256, then heads.

    The encoder runs batched over all T*B observations; the cell then steps through time.
    """

    def __init__(
        self, board_size: int, in_channels: int = 4, num_actions: int = 3, cell: str = "gru"
    ) -> None:
        super().__init__()
        del board_size  # the pool in conv_features makes this network board-size independent
        if cell not in CELLS:
            raise ValueError(
                f"agent 'recurrent' cell must be one of {', '.join(CELLS)}, got {cell!r}"
            )
        self.cell = cell
        self.state_slots = 1 if cell == "gru" else 2
        self.hidden_size = RECURRENT_HIDDEN
        self.encoder = nn.Sequential(
            conv_features(in_channels, 32), nn.Linear(2 * 32 * 16, RECURRENT_HIDDEN), nn.ReLU()
        )
        rnn_type = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_type(RECURRENT_HIDDEN, RECURRENT_HIDDEN)
        self.policy = nn.Linear(RECURRENT_HIDDEN, num_actions)
        self.value = nn.Linear(RECURRENT_HIDDEN, 1)

    def unroll(
        self, obs: torch.Tensor, state: torch.Tensor, starts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steps, batch = obs.shape[:2]
        features = self.encoder(obs.flatten(0, 1)).view(steps, batch, -1)
        outputs = []
        for t in range(steps):
            state = state.masked_fill(starts[t][:, None, None], 0.0)
            output, state = self._step(features[t], state)
            outputs.append(output)
        hidden = torch.stack(outputs)
        return self.policy(hidden), self.value(hidden).squeeze(-1), state

    def _step(
        self, features: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One cell step: features (B, H), state (B, K, H) -> (output (B, H), new state)."""
        layered = state.transpose(0, 1).contiguous()  # (K, B, H); each slot is (1 layer, B, H)
        if self.cell == "gru":
            output, h = self.rnn(features.unsqueeze(0), layered)
            return output[0], h.transpose(0, 1)
        output, (h, c) = self.rnn(features.unsqueeze(0), (layered[:1], layered[1:]))
        return output[0], torch.cat([h, c]).transpose(0, 1)


def episode_starts(done: torch.Tensor) -> torch.Tensor:
    """Episode starts of a rollout from its dones (T, B): row 0 false, row t = done[t - 1].

    Row 0 is false because the rollout's saved start state is already reset where needed.
    """
    starts = torch.zeros(done.shape, dtype=torch.bool, device=done.device)
    starts[1:] = done[:-1].bool()
    return starts


class PolicyPlayer:
    """Plays ``num_games`` parallel games with one model, carrying a recurrent model's state.

    A feed-forward model is called as ``model.act(obs, greedy=greedy)``, so any object with that
    method works (the scripted test policies included).
    """

    def __init__(self, model: PolicyNetwork, num_games: int, device: torch.device | str) -> None:
        self.model = model
        self.state: torch.Tensor | None = None
        if isinstance(model, RecurrentPolicy):
            self.state = model.initial_state(num_games, device)
            self._no_starts = torch.zeros(num_games, dtype=torch.bool, device=device)

    def act(self, obs: torch.Tensor, greedy: bool) -> torch.Tensor:
        """The actions (B,) for one step of every game."""
        if self.state is None:
            actions, _, _ = self.model.act(obs, greedy=greedy)
            return actions
        actions, _, _, self.state = self.model.act(obs, self.state, self._no_starts, greedy=greedy)
        return actions

    def end_episodes(self, done: torch.Tensor) -> None:
        """Reset the state of the games that just finished (done (B,) bool); no-op if stateless."""
        if self.state is not None:
            self.state = self.state.masked_fill(done[:, None, None], 0.0)
