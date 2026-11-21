import json
import os
import re
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from snakeppo import processes
from snakeppo.builders import observation_record
from snakeppo.config import Config, apply_overrides, save_config
from snakeppo.dashboard.launches import (
    LAUNCH_FILE,
    LOG_TAIL_LINES,
    OUTPUT_FILE,
    LaunchConflict,
    LaunchError,
    LaunchRequest,
    launch_argv,
    launch_folder,
    list_launches,
    parse_launch_request,
    resume_run,
    resume_sweep,
    start_launch,
    stop_run,
    stop_sweep,
)
from snakeppo.runs import create_run_dir, update_run_info, write_run_info

CLI = [sys.executable, "-m", "snakeppo.cli"]
LAUNCH = "20261119-090000-aaaaaa"
GROUP = "tiny@20261119-090000"


@pytest.fixture
def bench(tmp_path):
    """A repo-like folder: two presets under configs/ and an empty runs/ folder beside them."""
    cwd = tmp_path / "repo"
    configs = cwd / "configs"
    (configs / "rewards").mkdir(parents=True)
    (configs / "tiny.yaml").write_text("train:\n  name: tiny\n", encoding="utf-8")
    (configs / "rewards" / "far.yaml").write_text(
        "train:\n  name: far\n  runs_dir: elsewhere\n", encoding="utf-8"
    )
    runs = cwd / "runs"
    runs.mkdir()
    return SimpleNamespace(cwd=cwd, configs=configs, runs=runs)


class FakeSpawn:
    """Records every spawn and stands in this test process as the child (alive, same host)."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, *, cwd, log_path, env):
        self.calls.append(SimpleNamespace(argv=list(argv), cwd=cwd, log_path=log_path, env=env))
        return SimpleNamespace(pid=os.getpid())


def _gone():
    """A process record whose process ended (this pid, but another creation time)."""
    return {**processes.process_record(), "started": 1.0}


def _launch(runs, launch_id=LAUNCH, *, kind="train", alive=True, **fields):
    folder = processes.launch_dir(runs, launch_id)
    folder.mkdir(parents=True)
    record = {
        "id": launch_id,
        "kind": kind,
        "argv": [],
        "process": processes.process_record() if alive else _gone(),
        "created": "2026-11-19T09:00:00+01:00",
        "preset": "tiny.yaml",
        "overrides": [],
        "description": None,
        "seeds": None,
        "jobs": None,
        "group": None,
        "run": None,
        "resumes": None,
    } | fields
    (folder / LAUNCH_FILE).write_text(json.dumps(record), encoding="utf-8")
    return folder


def _run(runs, seed, *, status, group=None, launch=None, process=None, name="tiny"):
    config = apply_overrides(Config(), [f"env.seed={seed}"])
    run_dir = create_run_dir(runs, name, seed, now=datetime(2026, 11, 19, 9, 0))
    save_config(config, run_dir / "config.yaml")
    write_run_info(
        run_dir,
        name=name,
        agent="cnn",
        param_count=1,
        device="cpu",
        total_updates=3,
        group=group,
        observation=observation_record(config),
        process=process,
        launch=launch,
    )
    update_run_info(run_dir, status=status)
    return run_dir


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


# ------------------------------------------------------------------ requests
def test_parse_launch_request_reads_a_single_run():
    body = {
        "kind": "train",
        "preset": "rewards/far.yaml",
        "overrides": [" ppo.lr=0.001 ", ""],
        "description": "  Far food.  ",
        "seeds": [1, 2],
        "jobs": 3,
    }
    assert parse_launch_request(body) == LaunchRequest(
        kind="train",
        preset="rewards/far.yaml",
        overrides=("ppo.lr=0.001",),
        description="Far food.",
    )


def test_parse_launch_request_reads_a_sweep():
    body = {"kind": "sweep", "preset": "tiny.yaml", "seeds": [3, 1], "jobs": 2, "description": " "}
    assert parse_launch_request(body) == LaunchRequest(
        kind="sweep", preset="tiny.yaml", seeds=(3, 1), jobs=2
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (["train"], "the launch request must be a JSON object"),
        ({"kind": "train", "preset": "tiny.yaml", "name": "x"}, "unknown field 'name'"),
        ({"kind": "eval", "preset": "tiny.yaml"}, "kind must be 'train' or 'sweep', got 'eval'"),
        ({"kind": "train"}, "preset must be a config path"),
        ({"kind": "train", "preset": "  "}, "preset must be a config path"),
        ({"kind": "train", "preset": "tiny.yaml", "overrides": "ppo.lr=1"}, "overrides must be"),
        ({"kind": "train", "preset": "tiny.yaml", "overrides": [1]}, "overrides must be"),
        ({"kind": "train", "preset": "tiny.yaml", "description": 5}, "description must be text"),
        ({"kind": "sweep", "preset": "tiny.yaml"}, "a sweep needs at least one seed"),
        ({"kind": "sweep", "preset": "tiny.yaml", "seeds": [1, True]}, "seeds must be a list"),
        ({"kind": "sweep", "preset": "tiny.yaml", "seeds": [1, 2, 1]}, "lists seed 1 twice"),
        (
            {"kind": "sweep", "preset": "tiny.yaml", "seeds": [1], "jobs": 0},
            "jobs must be a whole number of at least 1, got 0",
        ),
    ],
)
def test_parse_launch_request_names_the_bad_field(body, message):
    with pytest.raises(LaunchError, match=re.escape(message)):
        parse_launch_request(body)


# ------------------------------------------------------------------ command lines
def test_launch_argv_is_the_cli_train_command(bench):
    request = LaunchRequest(
        kind="train", preset="tiny.yaml", overrides=("ppo.lr=0.001",), description="Tiny run."
    )
    argv = launch_argv(request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd)
    assert argv == [
        *CLI,
        "train",
        "--config",
        "configs/tiny.yaml",
        "--set",
        "ppo.lr=0.001",
        "--description",
        "Tiny run.",
    ]


def test_launch_argv_sends_a_run_with_another_runs_folder_to_the_served_one(bench):
    request = LaunchRequest(kind="train", preset="rewards/far.yaml")
    argv = launch_argv(request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd)
    served = f"train.runs_dir={bench.runs.resolve().as_posix()}"
    assert argv == [*CLI, "train", "--config", "configs/rewards/far.yaml", "--set", served]
    request = LaunchRequest(
        kind="train", preset="rewards/far.yaml", overrides=("train.runs_dir=runs",)
    )
    argv = launch_argv(request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd)
    assert argv[-2:] == ["--set", "train.runs_dir=runs"]


def test_launch_argv_is_the_cli_sweep_command(bench):
    request = LaunchRequest(
        kind="sweep", preset="tiny.yaml", seeds=(1, 2, 3), jobs=2, description="d"
    )
    options = {"configs_dir": bench.configs, "runs_dir": bench.runs, "cwd": bench.cwd}
    assert launch_argv(request, group=GROUP, **options) == [
        *CLI,
        "sweep",
        "--seeds",
        "1,2,3",
        "--config",
        "configs/tiny.yaml",
        "--jobs",
        "2",
        "--group",
        GROUP,
        "--description",
        "d",
    ]
    argv = launch_argv(request, **options)
    assert re.fullmatch(r"tiny@\d{8}-\d{6}", argv[argv.index("--group") + 1])


@pytest.mark.parametrize(
    "preset", ["missing.yaml", "../outside.yaml", "rewards", "notes.txt", "ABS"]
)
def test_launch_argv_takes_only_a_yaml_preset_inside_the_configs_folder(bench, preset):
    (bench.cwd / "outside.yaml").write_text("train:\n  name: out\n", encoding="utf-8")
    (bench.configs / "notes.txt").write_text("train:\n  name: notes\n", encoding="utf-8")
    preset = str((bench.cwd / "outside.yaml").resolve()) if preset == "ABS" else preset
    request = LaunchRequest(kind="train", preset=preset)
    with pytest.raises(LaunchError, match=re.escape(f"no preset '{preset}' in ")):
        launch_argv(request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd)


@pytest.mark.parametrize(
    ("preset", "overrides", "message"),
    [
        ("tiny.yaml", ("ppo.nope=1",), "unknown key 'ppo.nope'"),
        ("tiny.yaml", ("ppo.lr",), "must look like section.key=value"),
        ("tiny.yaml", ("agent.name=nope",), "unknown agent 'nope'"),
        ("broken.yaml", (), "preset 'broken.yaml' is not valid YAML"),
    ],
)
def test_an_invalid_config_is_refused_and_nothing_is_spawned(bench, preset, overrides, message):
    (bench.configs / "broken.yaml").write_text("train: [", encoding="utf-8")
    spawn = FakeSpawn()
    request = LaunchRequest(kind="train", preset=preset, overrides=overrides)
    with pytest.raises(LaunchError, match=re.escape(message)):
        start_launch(
            request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd, spawn=spawn
        )
    assert spawn.calls == []
    assert not (bench.runs / processes.LAUNCHES_DIR).exists()


# ------------------------------------------------------------------ starting
def test_start_launch_spawns_the_command_and_records_the_launch(bench):
    spawn = FakeSpawn()
    request = LaunchRequest(
        kind="train", preset="tiny.yaml", overrides=("ppo.lr=0.001",), description="Tiny."
    )
    options = {"configs_dir": bench.configs, "runs_dir": bench.runs, "cwd": bench.cwd}
    entry = start_launch(request, spawn=spawn, **options)
    [call] = spawn.calls
    folder = bench.runs / processes.LAUNCHES_DIR / entry["id"]
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", entry["id"])
    assert call.argv == launch_argv(request, **options)
    assert call.cwd == bench.cwd
    assert call.log_path == folder / OUTPUT_FILE
    assert call.env[processes.LAUNCH_ENV] == entry["id"]
    assert call.env["PYTHONUNBUFFERED"] == "1"
    assert call.env["PATH"] == os.environ["PATH"]
    record = json.loads((folder / LAUNCH_FILE).read_text(encoding="utf-8"))
    assert datetime.fromisoformat(record["created"]).tzinfo is not None
    assert record == {
        "id": entry["id"],
        "kind": "train",
        "argv": call.argv,
        "process": processes.process_record(),
        "created": record["created"],
        "preset": "tiny.yaml",
        "overrides": ["ppo.lr=0.001"],
        "description": "Tiny.",
        "seeds": None,
        "jobs": None,
        "group": None,
        "run": None,
        "resumes": None,
    }
    assert entry == {
        "id": entry["id"],
        "kind": "train",
        "created": record["created"],
        "preset": "tiny.yaml",
        "overrides": ["ppo.lr=0.001"],
        "description": "Tiny.",
        "seeds": None,
        "jobs": None,
        "group": None,
        "run": None,
        "resumes": None,
        "alive": True,
        "stop_requested": False,
        "state": "running",
        "seed_runs": [],
        "log_tail": [],
    }


def test_start_launch_records_a_sweep_and_its_group(bench):
    spawn = FakeSpawn()
    request = LaunchRequest(kind="sweep", preset="tiny.yaml", seeds=(1, 2), jobs=2)
    entry = start_launch(
        request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd, spawn=spawn
    )
    argv = spawn.calls[0].argv
    assert argv[:4] == [*CLI, "sweep"]
    assert entry["group"] == argv[argv.index("--group") + 1]
    assert (entry["seeds"], entry["jobs"], entry["state"]) == ([1, 2], 2, "running")
    assert entry["seed_runs"] == [
        {"seed": 1, "run": None, "status": "waiting"},
        {"seed": 2, "run": None, "status": "waiting"},
    ]


def test_the_child_runs_detached_and_its_output_lands_in_the_launch_log(bench):
    children = []

    def spawn(argv, **kwargs):
        code = "import os; print('launch', os.environ['SNAKEPPO_LAUNCH_ID'])"
        children.append(processes.spawn_detached([sys.executable, "-c", code], **kwargs))
        return children[-1]

    request = LaunchRequest(kind="train", preset="tiny.yaml")
    entry = start_launch(
        request, configs_dir=bench.configs, runs_dir=bench.runs, cwd=bench.cwd, spawn=spawn
    )
    assert children[0].wait(timeout=60) == 0
    [listed] = list_launches(bench.runs)
    assert listed["log_tail"] == [f"launch {entry['id']}"]
    assert (listed["alive"], listed["state"]) == (False, "failed")


# ------------------------------------------------------------------ listing
@pytest.mark.parametrize(
    ("alive", "run_status", "state"),
    [
        (True, "running", "running"),
        (False, "finished", "finished"),
        (False, "interrupted", "stopped"),
        (False, "failed", "failed"),
        (False, None, "failed"),
    ],
)
def test_a_train_launch_takes_its_state_from_the_run_it_made(bench, alive, run_status, state):
    _launch(bench.runs, alive=alive)
    _run(bench.runs, 7, status="finished", launch="20261119-080000-ffffff")
    run_dir = None
    if run_status is not None:
        process = processes.process_record() if run_status == "running" else None
        run_dir = _run(bench.runs, 1, status=run_status, launch=LAUNCH, process=process)
    [entry] = list_launches(bench.runs)
    assert entry["run"] == (None if run_dir is None else run_dir.name)
    assert (entry["alive"], entry["stop_requested"], entry["state"]) == (alive, False, state)


def test_a_stop_request_on_its_run_shows_a_train_launch_as_stopping(bench):
    _launch(bench.runs)
    run_dir = _run(
        bench.runs, 1, status="running", launch=LAUNCH, process=processes.process_record()
    )
    processes.request_stop(run_dir)
    [entry] = list_launches(bench.runs)
    assert (entry["stop_requested"], entry["state"]) == (True, "stopping")


def test_a_resume_launch_follows_the_run_it_resumes(bench):
    run_dir = _run(bench.runs, 1, status="interrupted", launch="20261119-080000-ffffff")
    _launch(bench.runs, kind="resume", alive=False, preset=None, run=run_dir.name)
    [entry] = list_launches(bench.runs)
    assert (entry["kind"], entry["run"], entry["state"]) == ("resume", run_dir.name, "stopped")


def test_a_running_sweep_lists_every_seed(bench):
    _launch(bench.runs, kind="sweep", seeds=[1, 2, 3], jobs=2, group=GROUP)
    one = _run(bench.runs, 1, status="finished", group=GROUP)
    two = _run(bench.runs, 2, status="running", group=GROUP, process=processes.process_record())
    _run(bench.runs, 3, status="finished", group="other@20261119-090000", name="other")
    [entry] = list_launches(bench.runs)
    assert entry["seed_runs"] == [
        {"seed": 1, "run": one.name, "status": "finished"},
        {"seed": 2, "run": two.name, "status": "running"},
        {"seed": 3, "run": None, "status": "waiting"},
    ]
    assert entry["state"] == "running"


def test_a_stopping_sweep_will_skip_its_unstarted_seeds(bench):
    folder = _launch(bench.runs, kind="sweep", seeds=[1, 2], jobs=1, group=GROUP)
    _run(bench.runs, 1, status="running", group=GROUP, process=processes.process_record())
    processes.request_stop(folder)
    [entry] = list_launches(bench.runs)
    assert [row["status"] for row in entry["seed_runs"]] == ["running", "skipped"]
    assert (entry["stop_requested"], entry["state"]) == (True, "stopping")


@pytest.mark.parametrize(
    ("statuses", "stopped", "reported", "state"),
    [
        (
            ("finished", "finished", "finished"),
            False,
            ("finished", "finished", "finished"),
            "finished",
        ),
        (
            ("finished", "interrupted", None),
            True,
            ("finished", "interrupted", "skipped"),
            "stopped",
        ),
        (("finished", "failed", "finished"), False, ("finished", "failed", "finished"), "failed"),
        (("finished", "finished", None), False, ("finished", "finished", "failed"), "failed"),
    ],
)
def test_an_ended_sweep_takes_its_state_from_its_seeds(bench, statuses, stopped, reported, state):
    folder = _launch(bench.runs, kind="sweep", alive=False, seeds=[1, 2, 3], jobs=2, group=GROUP)
    if stopped:
        processes.request_stop(folder)
    for seed, status in enumerate(statuses, start=1):
        if status is not None:
            _run(bench.runs, seed, status=status, group=GROUP)
    [entry] = list_launches(bench.runs)
    assert tuple(row["status"] for row in entry["seed_runs"]) == reported
    assert (entry["alive"], entry["state"]) == (False, state)


def test_the_log_tail_is_the_last_lines_of_the_output(bench):
    folder = _launch(bench.runs, alive=False)
    lines = b"".join(f"line {i}\r\n".encode() for i in range(50))
    (folder / OUTPUT_FILE).write_bytes(lines + b"caf\xc3\xa9 \xff")
    [entry] = list_launches(bench.runs)
    kept = [f"line {i}" for i in range(50 - LOG_TAIL_LINES + 1, 50)]
    assert entry["log_tail"] == [*kept, "café �"]


def test_list_launches_is_newest_first_and_skips_unreadable_folders(bench):
    _launch(bench.runs, "20261119-090000-aaaaaa", alive=False)
    _launch(bench.runs, "20261119-100000-bbbbbb", alive=False)
    launches_dir = bench.runs / processes.LAUNCHES_DIR
    (launches_dir / "20261119-110000-cccccc").mkdir()
    (launches_dir / "20261119-120000-dddddd").mkdir()
    (launches_dir / "20261119-120000-dddddd" / LAUNCH_FILE).write_text("{", encoding="utf-8")
    (launches_dir / "notes.txt").write_text("not a launch", encoding="utf-8")
    ids = [entry["id"] for entry in list_launches(bench.runs)]
    assert ids == ["20261119-100000-bbbbbb", "20261119-090000-aaaaaa"]


def test_list_launches_of_a_folder_without_launches_is_empty(tmp_path):
    assert list_launches(tmp_path / "runs") == []


@pytest.mark.parametrize("launch_id", ["", ".", "..", "a/b", "a\\b", "20261119-090000-ffffff"])
def test_launch_folder_takes_only_a_launch_of_the_runs_folder(bench, launch_id):
    _launch(bench.runs)
    with pytest.raises(LaunchError, match=re.escape(f"no launch '{launch_id}'")):
        launch_folder(bench.runs, launch_id)


def test_launch_folder_finds_a_launch(bench):
    folder = _launch(bench.runs)
    assert launch_folder(bench.runs, LAUNCH) == folder


# ------------------------------------------------------------------ resuming
@pytest.mark.parametrize("status", ["interrupted", "failed", "stalled"])
def test_resume_run_continues_a_stopped_run(bench, status):
    stored, process = ("running", _gone()) if status == "stalled" else (status, None)
    run_dir = _run(bench.runs, 1, status=stored, group=GROUP, process=process)
    spawn = FakeSpawn()
    entry = resume_run(run_dir, runs_dir=bench.runs, cwd=bench.cwd, spawn=spawn)
    assert spawn.calls[0].argv == [*CLI, "train", "--resume", str(run_dir.resolve())]
    assert spawn.calls[0].env[processes.LAUNCH_ENV] == entry["id"]
    assert (entry["kind"], entry["run"], entry["group"]) == ("resume", run_dir.name, GROUP)
    assert (entry["preset"], entry["seeds"], entry["state"]) == (None, None, "running")


@pytest.mark.parametrize("status", ["finished", "running"])
def test_resume_run_refuses_a_finished_or_live_run(bench, status):
    process = processes.process_record() if status == "running" else None
    run_dir = _run(bench.runs, 1, status=status, process=process)
    spawn = FakeSpawn()
    with pytest.raises(LaunchConflict, match=f"run '{run_dir.name}' is {status}"):
        resume_run(run_dir, runs_dir=bench.runs, cwd=bench.cwd, spawn=spawn)
    assert spawn.calls == []


def test_resume_run_refuses_a_run_a_live_launch_is_working_on(bench):
    run_dir = _run(bench.runs, 1, status="interrupted")
    _launch(bench.runs, kind="resume", preset=None, run=run_dir.name)
    with pytest.raises(LaunchConflict, match=f"launch '{LAUNCH}'"):
        resume_run(run_dir, runs_dir=bench.runs, cwd=bench.cwd, spawn=FakeSpawn())


def test_resume_sweep_starts_the_same_sweep_in_the_same_group(bench):
    folder = _launch(
        bench.runs,
        kind="sweep",
        alive=False,
        overrides=["ppo.lr=0.001"],
        description="Tiny.",
        seeds=[1, 2],
        jobs=2,
        group=GROUP,
    )
    processes.request_stop(folder)
    _run(bench.runs, 1, status="finished", group=GROUP)
    _run(bench.runs, 2, status="interrupted", group=GROUP)
    spawn = FakeSpawn()
    entry = resume_sweep(
        LAUNCH, runs_dir=bench.runs, configs_dir=bench.configs, cwd=bench.cwd, spawn=spawn
    )
    assert spawn.calls[0].argv == [
        *CLI,
        "sweep",
        "--seeds",
        "1,2",
        "--config",
        "configs/tiny.yaml",
        "--set",
        "ppo.lr=0.001",
        "--jobs",
        "2",
        "--group",
        GROUP,
        "--description",
        "Tiny.",
    ]
    assert entry["id"] != LAUNCH
    assert (entry["kind"], entry["group"], entry["resumes"]) == ("sweep", GROUP, LAUNCH)
    assert [row["status"] for row in entry["seed_runs"]] == ["finished", "interrupted"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("alive", "is still running"),
        ("train", "is a train launch, not a sweep"),
        ("finished", "finished every seed"),
        ("busy", "is already being trained by launch '20261119-100000-bbbbbb'"),
    ],
)
def test_resume_sweep_refuses_what_cannot_be_resumed(bench, case, message):
    fields = {"seeds": [1, 2], "jobs": 1, "group": GROUP}
    _launch(
        bench.runs, kind="train" if case == "train" else "sweep", alive=case == "alive", **fields
    )
    if case == "finished":
        _run(bench.runs, 1, status="finished", group=GROUP)
        _run(bench.runs, 2, status="finished", group=GROUP)
    if case == "busy":
        _launch(bench.runs, "20261119-100000-bbbbbb", kind="sweep", resumes=LAUNCH, **fields)
    spawn = FakeSpawn()
    with pytest.raises(LaunchConflict, match=re.escape(message)):
        resume_sweep(
            LAUNCH, runs_dir=bench.runs, configs_dir=bench.configs, cwd=bench.cwd, spawn=spawn
        )
    assert spawn.calls == []


def test_resume_sweep_of_an_unknown_launch_is_refused(bench):
    with pytest.raises(LaunchError, match="no launch 'nope'"):
        resume_sweep("nope", runs_dir=bench.runs, configs_dir=bench.configs, cwd=bench.cwd)


# ------------------------------------------------------------------ stopping
def test_stop_run_asks_the_trainer_to_stop_then_ends_it_after_the_grace_period(bench):
    child = _sleeper()
    try:
        run_dir = _run(bench.runs, 1, status="running", process=processes.process_record(child.pid))
        stop_run(run_dir, grace=1.0)
        assert processes.stop_requested(run_dir)
        assert child.poll() is None
        child.wait(timeout=60)
    finally:
        child.kill()


@pytest.mark.parametrize(
    ("status", "process", "message"),
    [
        ("finished", None, "is finished; only a running run can be stopped"),
        ("interrupted", None, "is interrupted; only a running run can be stopped"),
        ("running", "gone", "is stalled; only a running run can be stopped"),
        ("running", None, "has no training process on this machine"),
    ],
)
def test_stop_run_refuses_a_run_that_is_not_training_here(bench, status, process, message):
    run_dir = _run(bench.runs, 1, status=status, process=_gone() if process else None)
    with pytest.raises(LaunchConflict, match=re.escape(f"run '{run_dir.name}' {message}")):
        stop_run(run_dir)
    assert not processes.stop_requested(run_dir)


def test_stop_sweep_stops_the_sweep_and_every_live_seed(bench):
    parent, seed = _sleeper(), _sleeper()
    try:
        folder = _launch(
            bench.runs,
            kind="sweep",
            process=processes.process_record(parent.pid),
            seeds=[1, 2, 3],
            jobs=2,
            group=GROUP,
        )
        live = _run(
            bench.runs, 1, status="running", group=GROUP, process=processes.process_record(seed.pid)
        )
        done = _run(bench.runs, 2, status="finished", group=GROUP)
        stop_sweep(LAUNCH, runs_dir=bench.runs, grace=1.0)
        assert processes.stop_requested(folder)
        assert processes.stop_requested(live)
        assert not processes.stop_requested(done)
        assert parent.poll() is None
        parent.wait(timeout=60)
        seed.wait(timeout=60)
    finally:
        parent.kill()
        seed.kill()


@pytest.mark.parametrize(
    ("kind", "alive", "message"),
    [
        ("train", True, "is a train launch, not a sweep"),
        ("sweep", False, "is not running"),
    ],
)
def test_stop_sweep_refuses_a_launch_that_is_not_a_live_sweep(bench, kind, alive, message):
    folder = _launch(bench.runs, kind=kind, alive=alive, seeds=[1], jobs=1, group=GROUP)
    with pytest.raises(LaunchConflict, match=re.escape(message)):
        stop_sweep(LAUNCH, runs_dir=bench.runs)
    assert not processes.stop_requested(folder)


def test_stop_sweep_of_an_unknown_launch_is_refused(bench):
    with pytest.raises(LaunchError, match="no launch 'nope'"):
        stop_sweep("nope", runs_dir=bench.runs)
