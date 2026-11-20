"""``snakeppo sweep``: one ``snakeppo train`` child process per seed, all in one seed group.

Every seed runs in its own process, even one at a time, so each behaves exactly like a manual
``train`` and a crash in one cannot take the others down. With ``jobs > 1`` the children's
output goes to temporary files instead of the terminal, so progress bars never interleave;
a failed seed's last lines are printed when it ends. A detached sweep (one the dashboard
started) has no terminal: its seeds write straight into the sweep's own output instead.

A sweep may join a group that already has run folders: their interrupted or stalled seeds are
continued with ``train --resume``, and finished, failed or still-running seeds are left alone.
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

from snakeppo.processes import child_creationflags
from snakeppo.runs import find_runs, load_run

GROUP_TIME_FORMAT = "%Y%m%d-%H%M%S"
FAILURE_TAIL_LINES = 20
POLL_SECONDS = 0.2
INTERRUPT_GRACE_SECONDS = 30.0
RESUMED_STATUSES = ("interrupted", "stalled")
SKIPPED = "skipped"

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


def resume_command(run_dir: Path) -> list[str]:
    return [sys.executable, "-m", "snakeppo.cli", "train", "--resume", str(run_dir)]


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    returncode: int | None  # None when this sweep did not run it (skipped, or already in the group)
    run_dir: Path | None  # None when no folder exists for the seed
    status: str  # the run's status, "no run folder", or "skipped"


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
    stop_file: Path | None = None,
    detached: bool = False,
) -> list[SeedOutcome]:
    """Train every seed (at most ``jobs`` at once); returns one outcome per seed, in order.

    A seed that already has a folder in ``group`` is resumed when it is interrupted or stalled
    and otherwise left alone. Once ``stop_file`` exists no further seed starts; the seeds never
    started are ``skipped``. ``detached`` is for a sweep with no console (a dashboard launch).
    """
    if jobs < 1:
        raise ValueError(f"--jobs must be at least 1, got {jobs}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds lists a seed twice: {', '.join(map(str, seeds))}")
    build = command or child_command
    existing, commands = _folders_by_seed(runs_dir, group), {}
    for seed in seeds:
        folder = existing.get(seed)
        if folder is None:
            commands[seed] = build(seed, group, train_args)
        elif load_run(folder).status in RESUMED_STATUSES:
            commands[seed] = resume_command(folder)
    capture = jobs > 1 and not detached
    pending, running, codes = [s for s in seeds if s in commands], [], {}
    stopped = False
    try:
        with contextlib.ExitStack() as logs:
            while (pending and not stopped) or running:
                stopped = stopped or (stop_file is not None and stop_file.exists())
                while pending and not stopped and len(running) < jobs:
                    seed = pending.pop(0)
                    log = logs.enter_context(tempfile.TemporaryFile()) if capture else None
                    running.append(_Child(seed, _start(commands[seed], log, detached), log))
                    if jobs > 1:
                        print(f"seed {seed}: started")
                for child in [c for c in running if c.process.poll() is not None]:
                    running.remove(child)
                    codes[child.seed] = child.process.returncode
                    if jobs > 1:
                        _report(child)
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
    return [
        SeedOutcome(seed, None, None, SKIPPED)
        if seed in pending
        else _outcome(seed, codes.get(seed), folders.get(seed))
        for seed in seeds
    ]


def _start(argv: list[str], log: IO[bytes] | None, detached: bool) -> subprocess.Popen[bytes]:
    if detached:
        # no console to inherit: CREATE_NO_WINDOW keeps each seed from opening a window, and
        # the explicit stdout sends its output into the sweep's own (the launch's output.log)
        sys.stdout.flush()
        return subprocess.Popen(
            argv,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            creationflags=child_creationflags(),
        )
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT if log is not None else None)


def _report(child: _Child) -> None:
    """One line per finished seed; a failed seed also shows the tail of its captured output."""
    code = child.process.returncode
    if code == 0:
        print(f"seed {child.seed}: done")
        return
    if child.log is None:  # its output already went to the sweep's own output
        print(f"seed {child.seed}: failed with exit code {code}")
        return
    child.log.seek(0)
    tail = child.log.read().decode("utf-8", "replace").splitlines()[-FAILURE_TAIL_LINES:]
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


def _outcome(seed: int, returncode: int | None, run_dir: Path | None) -> SeedOutcome:
    status = load_run(run_dir).status if run_dir is not None else "no run folder"
    return SeedOutcome(seed, returncode, run_dir, status)


def format_outcomes(group: str, outcomes: Sequence[SeedOutcome]) -> str:
    ok = sum(1 for o in outcomes if o.status == "finished")
    lines = [f"Sweep {group}: {ok} of {len(outcomes)} seeds trained successfully."]
    for o in outcomes:
        if o.status == SKIPPED:
            result = SKIPPED
        elif o.returncode is None:
            result = f"{o.status}, already in the group"
        elif o.returncode == 0:
            result = o.status
        else:
            result = f"{o.status}, exit code {o.returncode}"
        lines.append(f"  seed {o.seed}: {result}  {o.run_dir or '-'}")
    return "\n".join(lines)
