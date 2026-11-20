"""Process records, stop requests and detached spawning, shared by the trainer, sweeps and the
dashboard.

A process record is ``{"pid", "started", "host"}``: the pid plus its creation time as ``psutil``
reports it, so a recycled pid is never mistaken for the original process, and the host, so a
record written on another machine is never judged here. Liveness goes through ``psutil`` because
on Windows ``os.kill(pid, 0)`` sends a Ctrl-C event (``signal.CTRL_C_EVENT == 0``) and would
interrupt the process it probes.

A stop request is a small JSON file in a run folder (or a launch folder) that the process
owning it polls and answers by ending gracefully. This module imports only the standard library
and ``psutil``; :mod:`snakeppo.runs` imports it.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil

LAUNCHES_DIR = ".launches"
LAUNCH_ENV = "SNAKEPPO_LAUNCH_ID"
STOP_REQUEST_FILE = "stop-request"
CREATE_TIME_TOLERANCE = 0.5  # seconds; psutil create_time() float compare
TERMINATE_WAIT_SECONDS = 5.0


def process_record(pid: int | None = None) -> dict[str, Any]:
    """``{"pid", "started", "host"}`` of ``pid`` (default this process).

    ``started`` is the creation time ``psutil`` reports; ``psutil.NoSuchProcess`` is raised when
    no process has that pid.
    """
    pid = os.getpid() if pid is None else pid
    return {"pid": pid, "started": psutil.Process(pid).create_time(), "host": socket.gethostname()}


def _record_pid(record: Mapping[str, Any] | None) -> tuple[int, float] | None:
    """``(pid, started)`` of a well-formed record written on this host, else ``None``."""
    if not isinstance(record, Mapping):
        return None
    pid, started, host = record.get("pid"), record.get("started"), record.get("host")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if isinstance(started, bool) or not isinstance(started, int | float):
        return None
    if not isinstance(host, str) or host != socket.gethostname():
        return None
    return pid, float(started)


def _probe(record: Mapping[str, Any] | None) -> tuple[bool | None, psutil.Process | None]:
    """The :func:`is_alive` verdict plus the matching live process when there is one."""
    fields = _record_pid(record)
    if fields is None:
        return None, None
    pid, started = fields
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - started) > CREATE_TIME_TOLERANCE:
            return False, None  # the pid was recycled
        if process.status() == psutil.STATUS_ZOMBIE:
            return False, None  # it exited and only waits to be reaped
    except psutil.NoSuchProcess:  # also covers psutil.ZombieProcess
        return False, None
    except psutil.AccessDenied:
        return None, None  # cannot tell; callers fall back to their own rule
    return True, process


def is_alive(record: Mapping[str, Any] | None) -> bool | None:
    """True: same host and a live process with that pid and creation time. False: same host, no
    such process (gone or pid recycled). None: no record, malformed record, or another host.

    ``None`` also covers a process ``psutil`` may not inspect (access denied).
    """
    return _probe(record)[0]


# ------------------------------------------------------------------ stop requests
def request_stop(folder: Path, now: float | None = None) -> Path:
    """Write ``folder/STOP_REQUEST_FILE`` holding the request time; returns its path."""
    path = folder / STOP_REQUEST_FILE
    requested = time.time() if now is None else now
    path.write_text(json.dumps({"requested": requested}) + "\n", encoding="utf-8")
    return path


def stop_requested(folder: Path) -> bool:
    return (folder / STOP_REQUEST_FILE).is_file()


def clear_stop_request(folder: Path) -> None:
    """Remove the stop request; a folder without one is fine."""
    (folder / STOP_REQUEST_FILE).unlink(missing_ok=True)


def launch_dir(runs_dir: Path, launch_id: str) -> Path:
    """The launch record folder ``runs_dir/LAUNCHES_DIR/launch_id`` (the id is not validated)."""
    return runs_dir / LAUNCHES_DIR / launch_id


# ------------------------------------------------------------------ spawning
def spawn_detached(
    argv: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Start ``argv`` detached from this process, its output appended to ``log_path``.

    stdin is the null device and stderr goes to the log too. On Windows the child gets its own
    process group and no console; on POSIX its own session, so a Ctrl-C in the terminal that
    started this process never reaches it. The log handle is closed here once the child holds it.
    """
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
        platform_options: dict[str, Any] = {"creationflags": flags}
    else:
        platform_options = {"start_new_session": True}
    with log_path.open("ab") as log:
        return subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=None if env is None else dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **platform_options,
        )


def child_creationflags() -> int:
    """Creation flags for a detached parent's own children: no console window on Windows."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def terminate(record: Mapping[str, Any], *, wait: float = TERMINATE_WAIT_SECONDS) -> bool:
    """When is_alive(record): terminate the process and its recursive children, wait, kill
    survivors. True when nothing of it is left running.

    The whole tree goes because a venv's ``python.exe`` on Windows may be a launcher whose child
    is the interpreter. A record that is already gone gives True; one this host cannot judge
    (another host, malformed, access denied) is left alone and gives False.
    """
    alive, process = _probe(record)
    if alive is False:
        return True
    if process is None:
        return False
    try:
        family = [*process.children(recursive=True), process]
    except psutil.NoSuchProcess:
        return True
    for member in family:
        with contextlib.suppress(psutil.NoSuchProcess):
            member.terminate()
    _, survivors = psutil.wait_procs(family, timeout=wait)
    for member in survivors:
        with contextlib.suppress(psutil.NoSuchProcess):
            member.kill()
    _, survivors = psutil.wait_procs(survivors, timeout=wait)
    return not survivors
