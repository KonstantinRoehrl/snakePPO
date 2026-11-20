"""Server-side replay: up to four trained models play side by side, stepped on the CPU.

A :class:`ReplayManager` holds at most one :class:`ReplaySession`. A session loads each board's
checkpoint the way ``snakeppo eval`` does, plays one game per board in its own single-game play
env, and steps every board in lockstep on a daemon thread at the chosen speed. Each step
publishes one frame; streams read the latest frame with :meth:`ReplaySession.next_frame`, so a
slow reader skips frames instead of queueing them. A session closes itself once no stream has
been attached for ``idle_seconds``.

Replay runs on the CPU with :data:`REPLAY_TORCH_THREADS` torch threads, so it never takes the GPU
or the cores from a training run on the same machine.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from snakeppo.agents import PolicyPlayer
from snakeppo.agents.networks import PolicyNetwork
from snakeppo.builders import EVAL_SEED_OFFSET, load_policy, make_play_env, play_env_config
from snakeppo.config import config_from_dict
from snakeppo.env import END_REASONS, TorchSnakeEnv
from snakeppo.runs import CHECKPOINT_NAMES, load_checkpoint, resolve_checkpoint

MAX_BOARDS = 4
REPLAY_SPEEDS = (2, 5, 10, 25, 60, 0)  # steps per second; 0 = as fast as one CPU thread allows
DEFAULT_SPEED = 10
IDLE_SECONDS = 30.0
REPLAY_TORCH_THREADS = 1
# Frame cell codes; a cell shows the first that applies of wall, food, head, body, else empty.
CELL_EMPTY, CELL_WALL, CELL_FOOD, CELL_HEAD, CELL_BODY = range(5)
CONTROL_ACTIONS = ("pause", "resume", "step", "speed", "restart")

BOARD_FIELDS = ("run", "checkpoint", "greedy", "board_size")
POLL_SECONDS = 0.1  # longest the stepping thread waits before it re-reads the idle clock
JOIN_SECONDS = 5.0  # how long close() waits for the stepping thread to finish its step


class ReplayError(ValueError):
    """A replay request names a board, checkpoint, action or value that cannot be played."""


@dataclass(frozen=True)
class BoardSpec:
    """What one board plays: a run's checkpoint, greedy or sampled, on a board size."""

    run: str
    checkpoint: str = "best"  # one of runs.CHECKPOINT_NAMES
    greedy: bool = True
    board_size: int | None = None  # None = the size it trained on


def parse_board_specs(raw: Any) -> list[BoardSpec]:
    """A JSON list of 1 to ``MAX_BOARDS`` board objects; :class:`ReplayError` names a bad field."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_BOARDS:
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise ReplayError(f"boards must be a list of 1 to {MAX_BOARDS} board objects, got {got}")
    return [_parse_board_spec(item, f"boards[{index}]") for index, item in enumerate(raw)]


def _parse_board_spec(item: Any, where: str) -> BoardSpec:
    if not isinstance(item, dict):
        raise ReplayError(f"{where} must be an object, got {type(item).__name__}")
    unknown = sorted(set(item) - set(BOARD_FIELDS))
    if unknown:
        raise ReplayError(
            f"{where} has unknown field {unknown[0]!r}; expected {', '.join(BOARD_FIELDS)}"
        )
    run = item.get("run")
    if not isinstance(run, str) or not run:
        raise ReplayError(f"{where}.run must be a run id, got {run!r}")
    checkpoint = item.get("checkpoint", "best")
    if checkpoint not in CHECKPOINT_NAMES:
        raise ReplayError(
            f"{where}.checkpoint must be one of {', '.join(CHECKPOINT_NAMES)}, got {checkpoint!r}"
        )
    greedy = item.get("greedy", True)
    if not isinstance(greedy, bool):
        raise ReplayError(f"{where}.greedy must be true or false, got {greedy!r}")
    board_size = item.get("board_size")
    if board_size is not None and type(board_size) is not int:
        raise ReplayError(
            f"{where}.board_size must be a whole number or null (the trained size), "
            f"got {board_size!r}"
        )
    return BoardSpec(run, checkpoint, greedy, board_size)


def _check_speed(speed: Any) -> int:
    if type(speed) is not int or speed not in REPLAY_SPEEDS:
        choices = ", ".join(map(str, REPLAY_SPEEDS))
        raise ReplayError(
            f"speed must be one of {choices} (0 = as fast as possible), got {speed!r}"
        )
    return speed


class _Board:
    """One board's game: its model, single-game play env, player and finished-episode stats."""

    def __init__(self, spec: BoardSpec, run_dir: Path, index: int) -> None:
        cpu = torch.device("cpu")
        _, checkpoint_path = resolve_checkpoint(run_dir, spec.checkpoint)
        checkpoint = load_checkpoint(checkpoint_path, map_location=cpu)
        config = config_from_dict(checkpoint["config"])
        self.model: PolicyNetwork = load_policy(checkpoint, config, checkpoint_path).eval()
        play = play_env_config(config, spec.board_size)
        self.spec, self.agent, self.board_size = spec, config.agent.name, play.board_size
        self.env: TorchSnakeEnv = make_play_env(
            play,
            config.observation,
            num_envs=1,
            device=cpu,
            seed=config.env.seed + EVAL_SEED_OFFSET + index,
        )
        self.restart()

    def restart(self) -> None:
        """New episode, fresh player state, stats zeroed."""
        self.obs = self.env.reset()
        self.player = PolicyPlayer(self.model, 1, self.env.device)
        self.episodes, self.score_sum, self.last_end = 0, 0.0, None

    def step(self) -> None:
        obs, _, _, _, info = self.env.step(self.player.act(self.obs, self.spec.greedy))
        self.obs = obs
        self.player.end_episodes(info["done"])
        if bool(info["done"][0]):
            self.episodes += 1
            self.score_sum += float(info["episode_score"][0])
            self.last_end = END_REASONS[int(info["end_reason"][0]) - 1]

    def describe(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "run": spec.run,
            "checkpoint": spec.checkpoint,
            "greedy": spec.greedy,
            "board_size": self.board_size,
            "agent": self.agent,
        }

    def frame(self) -> dict[str, Any]:
        env, size = self.env, self.board_size
        grid = env.render_grid()[0]  # (4, S, S): body (head included), head, food, walls
        cells = torch.full((size, size), CELL_EMPTY, dtype=torch.uint8)
        for channel, code in ((0, CELL_BODY), (1, CELL_HEAD), (2, CELL_FOOD), (3, CELL_WALL)):
            cells[grid[channel] > 0] = code
        length = int(env.snake_len[0])
        body = env.body_positions[0, :length]
        return {
            "size": size,
            "cells": (cells.flatten() + ord("0")).numpy().tobytes().decode("ascii"),
            "body": (body[:, 0] * size + body[:, 1]).tolist(),
            "score": int(env.score[0]),
            "length": length,
            "since_apple": int(env.steps_since_food[0]),
            "last_end": self.last_end,
            "episodes": self.episodes,
            "mean_score": self.score_sum / self.episodes if self.episodes else None,
        }


class ReplaySession:
    """Up to ``MAX_BOARDS`` boards stepped in lockstep on one daemon thread."""

    def __init__(
        self,
        boards: Sequence[_Board],
        *,
        speed: int,
        idle_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self.id = uuid.uuid4().hex
        self._boards = list(boards)
        self._idle_seconds, self._clock = idle_seconds, clock
        self._cond = threading.Condition()
        self._paused, self._speed, self._closed = False, speed, False
        self._steps_requested, self._restart_requested = 0, False
        self._attached, self._idle_since = 0, clock()
        self._due = 0.0  # time.monotonic() at which the next paced step is due
        self._seq = 0
        self._frame: dict[str, Any] = {}
        with self._cond:
            self._publish_locked([board.frame() for board in self._boards])
        self._thread = threading.Thread(
            target=self._run, name=f"snakeppo-replay-{self.id[:8]}", daemon=True
        )
        self._thread.start()

    @property
    def closed(self) -> bool:
        return self._closed

    def describe(self) -> dict[str, Any]:
        with self._cond:
            return {
                "id": self.id,
                "paused": self._paused,
                "speed": self._speed,
                "boards": [board.describe() for board in self._boards],
            }

    def control(self, action: str, value: Any = None) -> dict[str, Any]:
        """Apply ``action`` (one of :data:`CONTROL_ACTIONS`); returns ``{"paused", "speed"}``."""
        if action not in CONTROL_ACTIONS:
            raise ReplayError(
                f"unknown replay action {action!r}; expected one of {', '.join(CONTROL_ACTIONS)}"
            )
        with self._cond:
            if action == "pause" or action == "resume":
                self._paused = action == "pause"
                self._republish_locked()
            elif action == "speed":
                self._speed = _check_speed(value)
                self._republish_locked()
            elif action == "step":
                if not self._paused:
                    raise ReplayError("step works only while paused; pause the replay first")
                self._steps_requested += 1
            else:
                self._restart_requested = True
            self._cond.notify_all()
            return {"paused": self._paused, "speed": self._speed}

    def next_frame(self, after: int, timeout: float) -> dict[str, Any] | None:
        """The latest frame once one newer than ``after`` exists; ``{"closed": True}`` once closed.

        ``None`` when ``timeout`` seconds pass with neither.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._seq > after, timeout)
            if self._closed:
                return {"closed": True}
            return self._frame if self._seq > after else None

    def attach(self) -> None:
        """A stream connected: the session stays open while one is attached."""
        with self._cond:
            self._attached += 1

    def detach(self) -> None:
        """A stream went away; the idle clock starts when none is left."""
        with self._cond:
            self._attached = max(0, self._attached - 1)
            if not self._attached:
                self._idle_since = self._clock()

    def close(self) -> None:
        """Stop stepping and wake every waiting stream; waits for the stepping thread."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(JOIN_SECONDS)

    # ---------------------------------------------------------------- stepping thread
    def _run(self) -> None:
        try:
            with torch.no_grad():  # grad mode is per thread
                while (action := self._next_action()) is not None:
                    started = time.monotonic()
                    for board in self._boards:
                        if action == "restart":
                            board.restart()
                        else:
                            board.step()
                    frames = [board.frame() for board in self._boards]
                    with self._cond:
                        if self._closed:
                            return
                        self._publish_locked(frames)
                        speed = self._speed
                    if action == "step" and speed:
                        self._due = started + 1.0 / speed
        finally:
            self.close()

    def _next_action(self) -> str | None:
        """Block until the thread should restart or step; None once the session should end."""
        with self._cond:
            while True:
                if self._closed:
                    return None
                if not self._attached and self._clock() - self._idle_since >= self._idle_seconds:
                    return None
                if self._restart_requested:
                    self._restart_requested = False
                    return "restart"
                if self._paused:
                    if self._steps_requested:
                        self._steps_requested -= 1
                        return "step"
                    self._cond.wait(POLL_SECONDS)
                    continue
                delay = self._due - time.monotonic() if self._speed else 0.0
                if delay <= 0:
                    return "step"
                self._cond.wait(min(delay, POLL_SECONDS))

    def _publish_locked(self, boards: list[dict[str, Any]]) -> None:
        self._seq += 1
        self._frame = {"seq": self._seq, "paused": self._paused, "speed": self._speed}
        self._frame["boards"] = boards
        self._cond.notify_all()

    def _republish_locked(self) -> None:
        """The last frame again under a new seq, so streams see a pause or speed change."""
        self._publish_locked(self._frame["boards"])


class ReplayManager:
    """Holds the one replay session; creating a session closes the previous one."""

    def __init__(
        self, *, idle_seconds: float = IDLE_SECONDS, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._idle_seconds, self._clock = idle_seconds, clock
        self._lock = threading.Lock()
        self._session: ReplaySession | None = None

    def create(
        self, boards: Sequence[tuple[BoardSpec, Path]], *, speed: int = DEFAULT_SPEED
    ) -> ReplaySession:
        """A new session playing ``boards`` (each spec with its run folder), started at once.

        Every board is loaded before the previous session closes, so a refused request leaves
        the current replay playing. Missing checkpoints, mismatched observations and board sizes
        the run cannot play raise :class:`ReplayError` naming the board.
        """
        _check_speed(speed)
        if not 1 <= len(boards) <= MAX_BOARDS:
            raise ReplayError(f"a replay plays 1 to {MAX_BOARDS} boards, got {len(boards)}")
        with self._lock:
            torch.set_num_threads(REPLAY_TORCH_THREADS)
            loaded = []
            for index, (spec, run_dir) in enumerate(boards):
                try:
                    loaded.append(_Board(spec, run_dir, index))
                except (FileNotFoundError, ValueError) as error:
                    raise ReplayError(f"board {index + 1} ({spec.run}): {error}") from error
            if self._session is not None:
                self._session.close()
            self._session = ReplaySession(
                loaded, speed=speed, idle_seconds=self._idle_seconds, clock=self._clock
            )
            return self._session

    def get(self, session_id: str) -> ReplaySession | None:
        """The session with ``session_id``; None when it is unknown or closed."""
        session = self._session
        if session is None or session.id != session_id or session.closed:
            return None
        return session

    def close(self) -> None:
        """Close the current session, if any (the server calls this when it shuts down)."""
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
