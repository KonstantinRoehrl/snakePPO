"""Run folders: layout, ``run.json``, the ``metrics.jsonl`` contract, status and checkpoints.

Layout (schema version 2)::

    <runs_dir>/<YYYY-MM-DD_HHMMSS>_<name>_seed<k>/
        run.json  config.yaml  metrics.jsonl  tb/  checkpoints/{best,last}.pt

Readers accept schema-1 folders too; their status is inferred (see :func:`derive_status`).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from snakeppo.config import Config, load_config, total_updates

SCHEMA_VERSION = 2
RUN_INFO_FILE = "run.json"
CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.jsonl"
TENSORBOARD_DIR = "tb"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAMES = ("best", "last")

# Stored in run.json by the trainer.
WRITTEN_STATUSES = ("running", "finished", "interrupted", "failed")
# What readers report: the stored ones plus the two derived ones.
RUN_STATUSES = ("running", "stalled", "finished", "interrupted", "failed", "unknown")
STALL_FLOOR_SECONDS = 60.0
STALL_GAP_FACTOR = 5.0
STALL_RECENT_UPDATES = 10

_REPLACE_ATTEMPTS = 20
_REPLACE_BACKOFF_SECONDS = 0.05


def create_run_dir(runs_dir: str | Path, name: str, seed: int, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    base = Path(runs_dir) / f"{stamp}_{name}_seed{seed}"
    run_dir, suffix = base, 2
    while run_dir.exists():
        run_dir, suffix = base.with_name(f"{base.name}_{suffix}"), suffix + 1
    (run_dir / CHECKPOINT_DIR).mkdir(parents=True)
    return run_dir


def git_sha(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


# ------------------------------------------------------------------ atomic writes
def replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Write via ``write(temp)`` next to ``path``, then ``os.replace`` it into place.

    On Windows ``os.replace`` fails while another process (the dashboard) has the target
    open, so a short retry loop rides out those reads.
    """
    temp = path.with_name(f".{path.name}.tmp")
    write(temp)
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_SECONDS)


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    text = json.dumps(data, indent=2) + "\n"
    replace_atomically(path, lambda temp: temp.write_text(text, encoding="utf-8"))


# ------------------------------------------------------------------ run.json
def write_run_info(
    run_dir: Path,
    *,
    name: str,
    agent: str,
    param_count: int,
    device: str,
    total_updates: int,
    group: str | None,
) -> dict[str, Any]:
    info = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "agent": agent,
        "param_count": param_count,
        "device": device,
        "status": "running",
        "total_updates": total_updates,
        "group": group,
        "resumed": [],
    }
    write_json_atomic(run_dir / RUN_INFO_FILE, info)
    return info


def read_run_info(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_INFO_FILE).read_text(encoding="utf-8"))


def update_run_info(run_dir: Path, **fields: Any) -> dict[str, Any]:
    """Rewrite ``run.json`` with ``fields`` changed; ``status`` must be a stored status."""
    status = fields.get("status")
    if status is not None and status not in WRITTEN_STATUSES:
        raise ValueError(f"status must be one of {', '.join(WRITTEN_STATUSES)}, got {status!r}")
    info = read_run_info(run_dir) | fields
    write_json_atomic(run_dir / RUN_INFO_FILE, info)
    return info


# ------------------------------------------------------------------ metrics.jsonl
class MetricsLogger:
    """Appends one JSON object per PPO update and mirrors every key to TensorBoard.

    TensorBoard's x-axis is ``env_steps`` so runs with different rollout sizes overlay.
    ``wall_time_offset`` continues a resumed run's clock; ``purge_step`` drops TensorBoard
    events at or after that step left behind by the interrupted session.
    """

    def __init__(
        self,
        run_dir: Path,
        tensorboard: bool = True,
        *,
        wall_time_offset: float = 0.0,
        purge_step: int | None = None,
    ) -> None:
        self._file = (run_dir / METRICS_FILE).open("a", encoding="utf-8")
        self._writer = (
            SummaryWriter(str(run_dir / TENSORBOARD_DIR), purge_step=purge_step)
            if tensorboard
            else None
        )
        self._offset = wall_time_offset
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Seconds of training so far, including the time before a resume."""
        return self._offset + time.perf_counter() - self._start

    def log(self, update: int, env_steps: int, metrics: Mapping[str, float]) -> None:
        record = {"update": update, "env_steps": env_steps, "wall_time": round(self.elapsed(), 3)}
        record.update({key: float(value) for key, value in metrics.items()})
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, float(value), env_steps)

    def close(self) -> None:
        self._file.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Every complete line of ``metrics.jsonl``; a partial last line is left out."""
    return read_metrics_since(run_dir, 0).lines


@dataclass(frozen=True)
class MetricsChunk:
    lines: list[dict[str, Any]]
    offset: int  # byte offset just after the last complete line returned
    reset: bool  # the caller's offset no longer fits the file; ``lines`` start from byte 0


def read_metrics_since(run_dir: Path, offset: int) -> MetricsChunk:
    """Complete lines after byte ``offset``; restarts from 0 when the file was rewritten.

    A valid offset always sits just after a newline, so an offset past the end of the file or
    one not preceded by a newline means a resume truncated the file: the chunk then starts
    again from byte 0 with ``reset=True``.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    path = run_dir / METRICS_FILE
    if not path.is_file():
        return MetricsChunk([], 0, offset > 0)
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        reset = offset > size
        if not reset and offset > 0:
            handle.seek(offset - 1)
            reset = handle.read(1) != b"\n"
        start = 0 if reset else offset
        handle.seek(start)
        data = handle.read(size - start)
    complete = data[: data.rfind(b"\n") + 1]
    lines = [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]
    return MetricsChunk(lines, start + len(complete), reset)


def truncate_metrics(run_dir: Path, after_update: int) -> int:
    """Drop every line with ``update > after_update`` (atomic rewrite); returns lines dropped."""
    lines = read_metrics(run_dir)
    kept = [line for line in lines if line["update"] <= after_update]
    text = "".join(json.dumps(line) + "\n" for line in kept)
    replace_atomically(run_dir / METRICS_FILE, lambda temp: temp.write_text(text, encoding="utf-8"))
    return len(lines) - len(kept)


# ------------------------------------------------------------------ status
def seconds_per_update(metrics: Sequence[Mapping[str, Any]]) -> float | None:
    """Median ``wall_time`` gap over the last updates; ``None`` before two updates."""
    times = [line["wall_time"] for line in metrics[-STALL_RECENT_UPDATES:]]
    gaps = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
    return statistics.median(gaps) if gaps else None


def stall_window(metrics: Sequence[Mapping[str, Any]]) -> float:
    """Seconds without a metrics write after which a running run counts as stalled."""
    typical = seconds_per_update(metrics) or 0.0
    return max(STALL_FLOOR_SECONDS, STALL_GAP_FACTOR * typical)


def last_write_time(run_dir: Path) -> float:
    """mtime of ``metrics.jsonl``, or of ``run.json`` before the first update is logged."""
    metrics = run_dir / METRICS_FILE
    return (metrics if metrics.is_file() else run_dir / RUN_INFO_FILE).stat().st_mtime


def derive_status(
    info: Mapping[str, Any],
    config: Config,
    metrics: Sequence[Mapping[str, Any]],
    last_write: float,
    now: float,
) -> str:
    """One of :data:`RUN_STATUSES`.

    Schema 2: the stored status, reported as ``stalled`` when it says ``running`` but nothing
    was written for longer than :func:`stall_window`. Schema 1 stored no status: ``finished``
    when the last update reached the run's total, otherwise ``unknown``.
    """
    if "status" not in info:
        last = metrics[-1]["update"] if metrics else 0
        return "finished" if last == total_updates(config) else "unknown"
    status = info["status"]
    if status == "running" and now - last_write > stall_window(metrics):
        return "stalled"
    return status


@dataclass(frozen=True)
class RunRecord:
    """Everything readers need about one run folder, loaded at one moment."""

    path: Path
    info: dict[str, Any]
    config: Config
    metrics: list[dict[str, Any]]
    status: str

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def group(self) -> str | None:
        return self.info.get("group")

    @property
    def seed(self) -> int:
        return self.config.env.seed

    @property
    def total_updates(self) -> int:
        return self.info.get("total_updates") or total_updates(self.config)

    @property
    def update(self) -> int:
        return self.metrics[-1]["update"] if self.metrics else 0

    @property
    def env_steps(self) -> int:
        return self.metrics[-1]["env_steps"] if self.metrics else 0


def load_run(run_dir: Path, now: float | None = None) -> RunRecord:
    """Read one run folder; a key missing from an older ``config.yaml`` takes today's default."""
    info = read_run_info(run_dir)
    config = load_config(run_dir / CONFIG_FILE)
    metrics = read_metrics(run_dir)
    status = derive_status(
        info, config, metrics, last_write_time(run_dir), time.time() if now is None else now
    )
    return RunRecord(path=run_dir, info=info, config=config, metrics=metrics, status=status)


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    env_steps: int,
    config: Mapping[str, Any],
    best_eval: float,
    wall_time: float,
    rng: Mapping[str, Any],
    env: Mapping[str, Any],
) -> None:
    """Write a schema-2 checkpoint atomically, so a Ctrl+C mid-save never leaves half a file."""
    state = {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "env_steps": env_steps,
        "config": dict(config),
        "best_eval": best_eval,
        "wall_time": wall_time,
        "rng": dict(rng),
        "env": dict(env),
    }
    replace_atomically(path, lambda temp: torch.save(state, temp))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Returns the dict ``save_checkpoint`` wrote (schema-1 files lack the schema-2 keys)."""
    return torch.load(path, map_location=map_location, weights_only=True)


def resolve_checkpoint(target: str | Path, which: str = "best") -> tuple[Path, Path]:
    """Accept a run folder or a ``.pt`` file; return ``(run_dir, checkpoint_path)``."""
    if which not in CHECKPOINT_NAMES:
        raise ValueError(f"checkpoint must be one of {', '.join(CHECKPOINT_NAMES)}, got {which!r}")
    target = Path(target)
    if target.is_file():
        return target.parent.parent, target
    checkpoint = target / CHECKPOINT_DIR / f"{which}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no {which} checkpoint in {target} (looked for {checkpoint})")
    return target, checkpoint


# ------------------------------------------------------------------ listing
def find_runs(paths: list[Path]) -> list[Path]:
    """Each path is a run folder or a folder of run folders; returns run folders, sorted."""
    runs: set[Path] = set()
    for path in paths:
        if (path / RUN_INFO_FILE).is_file():
            runs.add(path)
        elif path.is_dir():
            runs.update(child for child in path.iterdir() if (child / RUN_INFO_FILE).is_file())
    return sorted(runs)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """One comparison row: identity, progress and eval scores of a finished or running run."""
    info = read_run_info(run_dir)
    config = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    metrics = read_metrics(run_dir)
    evals = [m["eval/score_mean"] for m in metrics if "eval/score_mean" in m]
    return {
        "run": run_dir.name,
        "agent": info["agent"],
        "seed": config["env"]["seed"],
        "updates": metrics[-1]["update"] if metrics else 0,
        "env_steps": metrics[-1]["env_steps"] if metrics else 0,
        "best_eval": max(evals) if evals else None,
        "last_eval": evals[-1] if evals else None,
    }
