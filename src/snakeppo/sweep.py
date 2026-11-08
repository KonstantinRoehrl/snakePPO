"""``snakeppo sweep``: one ``snakeppo train`` child process per seed, all in one seed group.

Every seed runs in its own process, even one at a time, so each behaves exactly like a manual
``train`` and a crash in one cannot take the others down. With ``jobs > 1`` the children's
output goes to temporary files instead of the terminal, so progress bars never interleave;
a failed seed's last lines are printed when it ends.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO

from snakeppo.runs import find_runs, load_run

GROUP_TIME_FORMAT = "%Y%m%d-%H%M%S"
FAILURE_TAIL_LINES = 20
POLL_SECONDS = 0.2
INTERRUPT_GRACE_SECONDS = 30.0

Command = Callable[[int, str, Sequence[str]], list[str]]


def group_id(name: str, now: datetime | None = None) -> str:
    """``<train.name>@<YYYYmmdd-HHMMSS>``: a rerun of the same sweep never joins an old group."""
    return f"{name}@{(now or datetime.now()).strftime(GROUP_TIME_FORMAT)}"


def child_command(seed: int, group: str, train_args: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "snakeppo.cli",
        "train",
        *train_args,
        "--seed",
        str(seed),
        "--group",
        group,
    ]


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    returncode: int
    run_dir: Path | None  # None when the child failed before creating its folder
    status: str  # the run's status, or "no run folder"


@dataclass
class _Child:
    seed: int
    process: subprocess.Popen[bytes]
    log: IO[bytes] | None


def run_sweep(
    seeds: Sequence[int],
    train_args: Sequence[str],
    *,
    runs_dir: Path,
    group: str,
    jobs: int = 1,
    command: Command | None = None,
) -> list[SeedOutcome]:
    """Train every seed (at most ``jobs`` at once); returns one outcome per seed, in order."""
    if jobs < 1:
        raise ValueError(f"--jobs must be at least 1, got {jobs}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds lists a seed twice: {', '.join(map(str, seeds))}")
    build = command or child_command
    capture = jobs > 1
    pending, running, codes = list(seeds), [], {}
    try:
        with contextlib.ExitStack() as logs:
            while pending or running:
                while pending and len(running) < jobs:
                    seed = pending.pop(0)
                    log = logs.enter_context(tempfile.TemporaryFile()) if capture else None
                    process = subprocess.Popen(
                        build(seed, group, train_args),
                        stdout=log,
                        stderr=subprocess.STDOUT if capture else None,
                    )
                    running.append(_Child(seed, process, log))
                    if capture:
                        print(f"seed {seed}: started")
                for child in [c for c in running if c.process.poll() is not None]:
                    running.remove(child)
                    codes[child.seed] = child.process.returncode
                    if child.log is not None:
                        _report(child, child.log)
                if running:
                    time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        # the console delivered Ctrl+C to the children too; let them record "interrupted"
        for child in running:
            try:
                child.process.wait(timeout=INTERRUPT_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                child.process.kill()
        raise
    folders = _folders_by_seed(runs_dir, group)
    return [_outcome(seed, codes[seed], folders.get(seed)) for seed in seeds]


def _report(child: _Child, log: IO[bytes]) -> None:
    """One line per finished seed; a failed seed also shows the tail of its captured output."""
    code = child.process.returncode
    if code == 0:
        print(f"seed {child.seed}: done")
    else:
        log.seek(0)
        tail = log.read().decode("utf-8", "replace").splitlines()[-FAILURE_TAIL_LINES:]
        print(f"seed {child.seed}: failed with exit code {code}; last output:")
        for line in tail:
            print(f"    {line}")


def _folders_by_seed(runs_dir: Path, group: str) -> dict[int, Path]:
    folders = {}
    for run_dir in find_runs([runs_dir]) if runs_dir.is_dir() else []:
        record = load_run(run_dir)
        if record.group == group:
            folders[record.seed] = run_dir
    return folders


def _outcome(seed: int, returncode: int, run_dir: Path | None) -> SeedOutcome:
    status = load_run(run_dir).status if run_dir is not None else "no run folder"
    return SeedOutcome(seed, returncode, run_dir, status)


def format_outcomes(group: str, outcomes: Sequence[SeedOutcome]) -> str:
    ok = sum(1 for o in outcomes if o.returncode == 0)
    lines = [f"Sweep {group}: {ok} of {len(outcomes)} seeds trained successfully."]
    for o in outcomes:
        result = o.status if o.returncode == 0 else f"{o.status}, exit code {o.returncode}"
        lines.append(f"  seed {o.seed}: {result}  {o.run_dir or '-'}")
    return "\n".join(lines)
