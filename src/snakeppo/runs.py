"""Run folders: layout, ``run.json``, the ``metrics.jsonl`` contract and checkpoints.

Layout (schema version 1)::

    <runs_dir>/<YYYY-MM-DD_HHMMSS>_<name>_seed<k>/
        run.json  config.yaml  metrics.jsonl  tb/  checkpoints/{best,last}.pt
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

SCHEMA_VERSION = 1
RUN_INFO_FILE = "run.json"
CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.jsonl"
TENSORBOARD_DIR = "tb"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAMES = ("best", "last")


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


def write_run_info(
    run_dir: Path, *, name: str, agent: str, param_count: int, device: str
) -> dict[str, Any]:
    info = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "agent": agent,
        "param_count": param_count,
        "device": device,
    }
    (run_dir / RUN_INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


class MetricsLogger:
    """Appends one JSON object per PPO update and mirrors every key to TensorBoard.

    TensorBoard's x-axis is ``env_steps`` so runs with different rollout sizes overlay.
    """

    def __init__(self, run_dir: Path, tensorboard: bool = True) -> None:
        self._file = (run_dir / METRICS_FILE).open("a", encoding="utf-8")
        self._writer = SummaryWriter(str(run_dir / TENSORBOARD_DIR)) if tensorboard else None
        self._start = time.perf_counter()

    def log(self, update: int, env_steps: int, metrics: Mapping[str, float]) -> None:
        record = {
            "update": update,
            "env_steps": env_steps,
            "wall_time": round(time.perf_counter() - self._start, 3),
        }
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


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    env_steps: int,
    config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "env_steps": env_steps,
            "config": dict(config),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Returns the dict ``save_checkpoint`` wrote: model, optimizer, update, env_steps, config."""
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


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / METRICS_FILE
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


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
    info = json.loads((run_dir / RUN_INFO_FILE).read_text(encoding="utf-8"))
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
