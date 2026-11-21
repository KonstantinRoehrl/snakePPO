"""Browser launches: the ``snakeppo`` command a launch request stands for, spawned detached.

Every launch is a folder ``<runs_dir>/.launches/<id>/`` holding ``launch.json`` (what was
started and its process record) and ``output.log`` (the child's stdout and stderr). The server
keeps no state of its own: everything here is read back from those folders and the run folders,
so a restarted server finds every launch again. Only ``python -m snakeppo.cli train|sweep`` is
ever spawned, as an argument list, so a browser launch is exactly the command a person types.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

from snakeppo import processes
from snakeppo.builders import make_policy
from snakeppo.config import load_config
from snakeppo.runs import RunRecord, find_runs, load_run, write_json_atomic
from snakeppo.sweep import group_id

LAUNCH_FILE = "launch.json"
OUTPUT_FILE = "output.log"
STOP_GRACE_SECONDS = 60.0
LOG_TAIL_LINES = 20
LAUNCH_KINDS = ("train", "sweep", "resume")

_REQUEST_KINDS = ("train", "sweep")
_RESUMABLE_STATUSES = ("interrupted", "failed", "stalled")
_TAIL_BYTES = 64 * 1024
_UNREADABLE = (OSError, ValueError, KeyError, yaml.YAMLError)


class LaunchError(ValueError):
    """A launch request the server cannot carry out as asked (HTTP 400)."""


class LaunchConflict(LaunchError):
    """The run or launch is not in a state that allows the action (HTTP 409)."""


# called as spawn(argv, cwd=cwd, log_path=log, env=env); tests pass a fake
Spawn = Callable[..., subprocess.Popen[bytes]]


@dataclass(frozen=True)
class LaunchRequest:
    """What the Launch view asked for; ``seeds`` and ``jobs`` belong to a sweep only."""

    kind: str  # "train" or "sweep"
    preset: str  # posix path relative to the configs folder
    overrides: tuple[str, ...] = ()
    description: str | None = None
    seeds: tuple[int, ...] = ()  # sweep only, non-empty, unique
    jobs: int = 1  # sweep only, >= 1


def parse_launch_request(body: Mapping[str, Any]) -> LaunchRequest:
    """The request in a ``POST /api/launch`` body; LaunchError names the first wrong field.

    Override lines are stripped and blank ones dropped, a blank description means none (the CLI
    then derives one from the preset), and a single run ignores ``seeds`` and ``jobs``.
    """
    if not isinstance(body, Mapping):
        raise LaunchError("the launch request must be a JSON object")
    known = [field.name for field in fields(LaunchRequest)]
    unknown = sorted(set(body) - set(known))
    if unknown:
        raise LaunchError(f"unknown field '{unknown[0]}'; known fields: {', '.join(known)}")
    kind = body.get("kind")
    if kind not in _REQUEST_KINDS:
        raise LaunchError(f"kind must be 'train' or 'sweep', got {kind!r}")
    preset = body.get("preset")
    if not isinstance(preset, str) or not preset.strip():
        raise LaunchError("preset must be a config path such as 'baseline.yaml'")
    overrides = body.get("overrides", [])
    if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
        raise LaunchError("overrides must be a list of 'section.key=value' strings")
    description = body.get("description")
    if description is not None and not isinstance(description, str):
        raise LaunchError("description must be text")
    request = LaunchRequest(
        kind=kind,
        preset=preset.strip(),
        overrides=tuple(o.strip() for o in overrides if o.strip()),
        description=(description or "").strip() or None,
    )
    if kind == "train":
        return request
    seeds = body.get("seeds", [])
    if not isinstance(seeds, list) or not all(_is_whole(seed) for seed in seeds):
        raise LaunchError(f"seeds must be a list of whole numbers, got {seeds!r}")
    if not seeds:
        raise LaunchError("a sweep needs at least one seed, e.g. [1, 2, 3]")
    repeated = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if repeated:
        raise LaunchError(f"seeds lists seed {repeated[0]} twice")
    jobs = body.get("jobs", 1)
    if not _is_whole(jobs) or jobs < 1:
        raise LaunchError(f"jobs must be a whole number of at least 1, got {jobs!r}")
    return replace(request, seeds=tuple(seeds), jobs=jobs)


def launch_argv(
    request: LaunchRequest,
    *,
    configs_dir: Path,
    runs_dir: Path,
    cwd: Path,
    group: str | None = None,
) -> list[str]:
    """The exact ``snakeppo`` command line for ``request``, validated as the CLI would.

    The preset must be a ``.yaml`` file inside ``configs_dir``. The config is loaded with the
    overrides and its network built, as ``snakeppo sweep`` does up front, so nothing is spawned
    for a request that would fail. A config whose ``train.runs_dir`` (resolved from ``cwd``) is
    not ``runs_dir`` gets ``--set train.runs_dir=<runs_dir>``, so the run lands where the
    dashboard looks. A sweep joins ``group``, by default a new group named after the config.
    """
    argv, _ = _command(request, configs_dir=configs_dir, runs_dir=runs_dir, cwd=cwd, group=group)
    return argv


def start_launch(
    request: LaunchRequest,
    *,
    configs_dir: Path,
    runs_dir: Path,
    cwd: Path,
    spawn: Spawn | None = None,
) -> dict[str, Any]:
    """Validate ``request``, spawn its command detached and record it; returns its LaunchEntry."""
    argv, group = _command(request, configs_dir=configs_dir, runs_dir=runs_dir, cwd=cwd, group=None)
    sweep = request.kind == "sweep"
    return _spawn(
        argv,
        kind=request.kind,
        runs_dir=runs_dir,
        cwd=cwd,
        spawn=spawn,
        preset=request.preset,
        overrides=list(request.overrides),
        description=request.description,
        seeds=list(request.seeds) if sweep else None,
        jobs=request.jobs if sweep else None,
        group=group,
        run=None,
        resumes=None,
    )


def resume_run(
    run_dir: Path, *, runs_dir: Path, cwd: Path, spawn: Spawn | None = None
) -> dict[str, Any]:
    """Continue an interrupted, failed or stalled run with ``snakeppo train --resume``."""
    run = load_run(run_dir)
    if run.status not in _RESUMABLE_STATUSES:
        raise LaunchConflict(
            f"run '{run.id}' is {run.status}; only an interrupted, failed or stalled run "
            "can be resumed"
        )
    busy = _live_launch(runs_dir, lambda entry: entry["run"] == run.id)
    if busy is not None:
        raise LaunchConflict(f"launch '{busy}' is still working on run '{run.id}'; wait for it")
    argv = [sys.executable, "-m", "snakeppo.cli", "train", "--resume", str(run_dir.resolve())]
    return _spawn(
        argv,
        kind="resume",
        runs_dir=runs_dir,
        cwd=cwd,
        spawn=spawn,
        preset=None,
        overrides=[],
        description=None,
        seeds=None,
        jobs=None,
        group=run.group,
        run=run.id,
        resumes=None,
    )


def resume_sweep(
    launch_id: str,
    *,
    runs_dir: Path,
    configs_dir: Path,
    cwd: Path,
    spawn: Spawn | None = None,
) -> dict[str, Any]:
    """Continue an ended sweep launch as the same seed group; returns the new launch's entry.

    The new launch runs ``snakeppo sweep --group <group>`` with the original preset, overrides,
    description, seeds and jobs: the sweep leaves finished seeds alone, resumes interrupted and
    stalled ones and starts the missing ones.
    """
    record, folder = _load_launch(runs_dir, launch_id)
    old = _entry(record, folder, _load_runs(runs_dir))
    if old["kind"] != "sweep":
        raise LaunchConflict(
            f"launch '{launch_id}' is a {old['kind']} launch, not a sweep; resume its run instead"
        )
    if old["alive"]:
        raise LaunchConflict(f"sweep launch '{launch_id}' is still running; stop it first")
    if old["state"] == "finished":
        raise LaunchConflict(f"sweep launch '{launch_id}' finished every seed; nothing to resume")
    group = old["group"]
    busy = _live_launch(
        runs_dir, lambda entry: entry["kind"] == "sweep" and entry["group"] == group
    )
    if busy is not None:
        raise LaunchConflict(f"group '{group}' is already being trained by launch '{busy}'")
    request = LaunchRequest(
        kind="sweep",
        preset=record["preset"],
        overrides=tuple(record["overrides"]),
        description=record["description"],
        seeds=tuple(record["seeds"]),
        jobs=record["jobs"],
    )
    argv, group = _command(
        request, configs_dir=configs_dir, runs_dir=runs_dir, cwd=cwd, group=group
    )
    return _spawn(
        argv,
        kind="sweep",
        runs_dir=runs_dir,
        cwd=cwd,
        spawn=spawn,
        preset=request.preset,
        overrides=list(request.overrides),
        description=request.description,
        seeds=list(request.seeds),
        jobs=request.jobs,
        group=group,
        run=None,
        resumes=launch_id,
    )


def list_launches(runs_dir: Path) -> list[dict[str, Any]]:
    """Every readable launch under ``runs_dir`` as a LaunchEntry, newest first."""
    base = runs_dir / processes.LAUNCHES_DIR
    if not base.is_dir():
        return []
    runs = _load_runs(runs_dir)
    entries = []
    for folder in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        try:
            record = _read_launch(folder)
        except _UNREADABLE:
            continue  # being created right now, or damaged
        entries.append(_entry(record, folder, runs))
    return entries


def stop_run(run_dir: Path, *, grace: float = STOP_GRACE_SECONDS) -> None:
    """Ask a run's trainer to stop at its next update boundary, as Ctrl+C would.

    Writes the run's ``stop-request``; if the training process still runs ``grace`` seconds
    later, it is terminated with its children (the run is then reported stalled and resumable).
    """
    run = load_run(run_dir)
    if run.status != "running":
        raise LaunchConflict(f"run '{run.id}' is {run.status}; only a running run can be stopped")
    process = run.info.get("process")
    if processes.is_alive(process) is not True:
        raise LaunchConflict(
            f"run '{run.id}' has no training process on this machine; stop it where it runs"
        )
    processes.request_stop(run_dir)
    _escalate(process, grace)


def stop_sweep(launch_id: str, *, runs_dir: Path, grace: float = STOP_GRACE_SECONDS) -> None:
    """Stop a browser-launched sweep: it starts no further seed and every live seed is stopped.

    Writes the launch's ``stop-request`` (polled by ``run_sweep``), stops each running seed of
    its group with :func:`stop_run`, and terminates the sweep process if it outlives ``grace``.
    """
    record, folder = _load_launch(runs_dir, launch_id)
    if record["kind"] != "sweep":
        raise LaunchConflict(
            f"launch '{launch_id}' is a {record['kind']} launch, not a sweep; stop its run instead"
        )
    if processes.is_alive(record.get("process")) is not True:
        raise LaunchConflict(f"sweep launch '{launch_id}' is not running")
    processes.request_stop(folder)
    for run in _load_runs(runs_dir).values():
        if run.group == record.get("group") and run.status == "running":
            with contextlib.suppress(LaunchConflict):  # it ended meanwhile, or runs elsewhere
                stop_run(run.path, grace=grace)
    _escalate(record["process"], grace)


def launch_folder(runs_dir: Path, launch_id: str) -> Path:
    """The folder of launch ``launch_id``; LaunchError unless it is a launch of ``runs_dir``."""
    base = os.path.realpath(runs_dir / processes.LAUNCHES_DIR)
    folder = os.path.realpath(os.path.join(base, launch_id))
    if not folder.startswith(base + os.sep) or os.path.dirname(folder) != base:
        raise LaunchError(f"no launch '{launch_id}'")
    if not os.path.isfile(os.path.join(folder, LAUNCH_FILE)):
        raise LaunchError(f"no launch '{launch_id}'")
    return Path(folder)


# ------------------------------------------------------------------ helpers
def _is_whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _command(
    request: LaunchRequest,
    *,
    configs_dir: Path,
    runs_dir: Path,
    cwd: Path,
    group: str | None,
) -> tuple[list[str], str | None]:
    """:func:`launch_argv` plus the group the run joins (``train.group`` for a single run)."""
    preset = _preset_path(request.preset, configs_dir)
    try:
        config = load_config(preset, request.overrides)
        make_policy(config)
    except yaml.YAMLError as error:
        raise LaunchError(f"preset '{request.preset}' is not valid YAML: {error}") from error
    except ValueError as error:  # ConfigError is a ValueError
        raise LaunchError(str(error)) from error
    sets = [item for override in request.overrides for item in ("--set", override)]
    if (cwd / config.train.runs_dir).resolve() != runs_dir.resolve():
        sets += ["--set", f"train.runs_dir={runs_dir.resolve().as_posix()}"]
    described = [] if request.description is None else ["--description", request.description]
    config_args = ["--config", _command_path(preset, cwd)]
    cli = [sys.executable, "-m", "snakeppo.cli"]
    if request.kind == "train":
        return [*cli, "train", *config_args, *sets, *described], config.train.group
    group = group or group_id(config.train.name)
    seeds = ",".join(map(str, request.seeds))
    jobs = ["--jobs", str(request.jobs)]
    argv = [*cli, "sweep", "--seeds", seeds, *config_args, *sets, *jobs, "--group", group]
    return [*argv, *described], group


def _preset_path(preset: str, configs_dir: Path) -> Path:
    root = configs_dir.resolve()
    path = (root / preset).resolve()
    if path.suffix != ".yaml" or not path.is_relative_to(root) or not path.is_file():
        raise LaunchError(
            f"no preset '{preset}' in {configs_dir.as_posix()}; "
            "choose a .yaml file from that folder"
        )
    return path


def _command_path(path: Path, cwd: Path) -> str:
    """``path`` as a person in ``cwd`` would type it: relative when it lies inside ``cwd``."""
    with contextlib.suppress(ValueError):
        return path.relative_to(cwd.resolve()).as_posix()
    return path.as_posix()


def _spawn(
    argv: Sequence[str],
    *,
    kind: str,
    runs_dir: Path,
    cwd: Path,
    spawn: Spawn | None,
    **details: Any,
) -> dict[str, Any]:
    """Start ``argv`` detached in a new launch folder and write its ``launch.json``."""
    launch_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    folder = processes.launch_dir(runs_dir, launch_id)
    folder.mkdir(parents=True)
    env = os.environ | {processes.LAUNCH_ENV: launch_id, "PYTHONUNBUFFERED": "1"}
    child = (spawn or processes.spawn_detached)(
        list(argv), cwd=cwd, log_path=folder / OUTPUT_FILE, env=env
    )
    record = {
        "id": launch_id,
        "kind": kind,
        "argv": list(argv),
        "process": _process_record(child.pid),
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        **details,
    }
    write_json_atomic(folder / LAUNCH_FILE, record)
    return _entry(record, folder, _load_runs(runs_dir))


def _process_record(pid: int) -> dict[str, Any] | None:
    """The child's process record, or None when it already ended (nothing is left to stop)."""
    try:
        return processes.process_record(pid)
    except psutil.Error:
        return None


def _escalate(process: Mapping[str, Any], grace: float) -> None:
    """After ``grace`` seconds, terminate ``process`` and its children if it still runs."""
    timer = threading.Timer(grace, processes.terminate, args=(dict(process),))
    timer.daemon = True
    timer.start()


def _read_launch(folder: Path) -> dict[str, Any]:
    record = json.loads((folder / LAUNCH_FILE).read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("kind") not in LAUNCH_KINDS:
        raise ValueError(f"{folder / LAUNCH_FILE} is not a launch record")
    return {**record, "id": folder.name}


def _load_launch(runs_dir: Path, launch_id: str) -> tuple[dict[str, Any], Path]:
    folder = launch_folder(runs_dir, launch_id)
    try:
        return _read_launch(folder), folder
    except _UNREADABLE as error:
        raise LaunchError(
            f"launch '{launch_id}' has an unreadable {LAUNCH_FILE}: {error}"
        ) from error


def _load_runs(runs_dir: Path) -> dict[str, RunRecord]:
    """Every readable run folder under ``runs_dir`` by id; unreadable ones are left out."""
    records = {}
    for run_dir in find_runs([runs_dir]):
        with contextlib.suppress(*_UNREADABLE):
            records[run_dir.name] = load_run(run_dir)
    return records


def _live_launch(runs_dir: Path, matches: Callable[[dict[str, Any]], bool]) -> str | None:
    """The id of a launch whose process still runs and that ``matches``, if any."""
    return next((e["id"] for e in list_launches(runs_dir) if e["alive"] and matches(e)), None)


def _entry(
    record: Mapping[str, Any], folder: Path, runs: Mapping[str, RunRecord]
) -> dict[str, Any]:
    """The LaunchEntry of a launch record, read against the run folders ``runs``."""
    kind, launch_id = record["kind"], record["id"]
    alive = processes.is_alive(record.get("process")) is True
    run_id = record.get("run")
    if kind == "train" and run_id is None:
        run_id = next((r.id for r in runs.values() if r.info.get("launch") == launch_id), None)
    run = None if run_id is None else runs.get(run_id)
    if kind == "sweep":
        stop = processes.stop_requested(folder)
        seed_runs = _seed_runs(record, runs, alive=alive, stop=stop)
    else:
        stop = run is not None and processes.stop_requested(run.path)
        seed_runs = []
    return {
        "id": launch_id,
        "kind": kind,
        "created": record.get("created"),
        "preset": record.get("preset"),
        "overrides": list(record.get("overrides") or []),
        "description": record.get("description"),
        "seeds": record.get("seeds"),
        "jobs": record.get("jobs"),
        "group": record.get("group"),
        "run": run_id,
        "resumes": record.get("resumes"),
        "alive": alive,
        "stop_requested": stop,
        "state": _state(kind, alive=alive, stop=stop, run=run, seed_runs=seed_runs),
        "seed_runs": seed_runs,
        "log_tail": _log_tail(folder / OUTPUT_FILE),
    }


def _seed_runs(
    record: Mapping[str, Any], runs: Mapping[str, RunRecord], *, alive: bool, stop: bool
) -> list[dict[str, Any]]:
    """One row per seed of a sweep launch.

    A seed without a run folder is waiting while the sweep runs, skipped once the sweep was
    told to stop, and failed otherwise (its child ended before it created a folder).
    """
    by_seed = {run.seed: run for run in runs.values() if run.group == record.get("group")}
    missing = "skipped" if stop else "waiting" if alive else "failed"
    rows = []
    for seed in record.get("seeds") or []:
        run = by_seed.get(seed)
        if run is None:
            rows.append({"seed": seed, "run": None, "status": missing})
        else:
            rows.append({"seed": seed, "run": run.id, "status": run.status})
    return rows


def _state(
    kind: str,
    *,
    alive: bool,
    stop: bool,
    run: RunRecord | None,
    seed_runs: Sequence[Mapping[str, Any]],
) -> str:
    """``running``/``stopping`` while the process lives; afterwards what the runs ended as."""
    if alive:
        return "stopping" if stop else "running"
    if kind != "sweep":
        status = None if run is None else run.status
        return {"finished": "finished", "interrupted": "stopped"}.get(status or "", "failed")
    statuses = {row["status"] for row in seed_runs}
    if statuses == {"finished"}:
        return "finished"
    if statuses & {"interrupted", "skipped"}:
        return "stopped"
    return "failed"


def _log_tail(path: Path) -> list[str]:
    """The last :data:`LOG_TAIL_LINES` lines of a launch's output; only its end is read."""
    try:
        with path.open("rb") as handle:
            start = max(0, handle.seek(0, os.SEEK_END) - _TAIL_BYTES)
            handle.seek(start)
            lines = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    if start > 0:
        lines = lines[1:]  # the first line was cut by the seek
    return lines[-LOG_TAIL_LINES:]
