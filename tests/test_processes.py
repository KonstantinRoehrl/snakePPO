import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from snakeppo import processes
from snakeppo.processes import (
    CREATE_TIME_TOLERANCE,
    LAUNCH_ENV,
    LAUNCHES_DIR,
    STOP_REQUEST_FILE,
    child_creationflags,
    clear_stop_request,
    is_alive,
    launch_dir,
    process_record,
    request_stop,
    spawn_detached,
    stop_requested,
    terminate,
)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def _wait_for(condition, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.05)


def test_constants_are_the_shared_names():
    assert LAUNCHES_DIR == ".launches"
    assert LAUNCH_ENV == "SNAKEPPO_LAUNCH_ID"
    assert STOP_REQUEST_FILE == "stop-request"
    assert processes.TERMINATE_WAIT_SECONDS == 5.0


def test_process_record_names_this_process_on_this_host():
    record = process_record()
    assert record == {
        "pid": os.getpid(),
        "started": pytest.approx(psutil.Process().create_time()),
        "host": socket.gethostname(),
    }
    assert json.loads(json.dumps(record)) == record  # it is stored in run.json
    assert is_alive(record) is True


def test_a_recycled_pid_is_not_the_recorded_process():
    record = process_record()
    assert is_alive({**record, "started": record["started"] - 10.0}) is False
    shifted = record["started"] + CREATE_TIME_TOLERANCE / 2
    assert is_alive({**record, "started": shifted}) is True


def test_a_process_that_exited_is_not_alive():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    record = process_record(child.pid)
    child.wait()
    assert is_alive(record) is False


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"pid": "12", "started": 1.0, "host": "x"},
        {"pid": True, "started": 1.0, "host": "x"},
        {"pid": 12, "started": None, "host": "x"},
        {"pid": 12, "started": 1.0},
        [1, 2, 3],
    ],
)
def test_no_or_malformed_record_cannot_be_judged(record):
    if isinstance(record, dict) and "host" in record:
        record = {**record, "host": socket.gethostname()}
    assert is_alive(record) is None


def test_a_record_from_another_host_cannot_be_judged():
    record = {**process_record(), "host": socket.gethostname() + "-elsewhere"}
    assert is_alive(record) is None
    assert terminate(record) is False  # left alone


def test_stop_request_round_trip(tmp_path):
    assert not stop_requested(tmp_path)
    clear_stop_request(tmp_path)  # missing is fine
    path = request_stop(tmp_path, now=123.5)
    assert path == tmp_path / STOP_REQUEST_FILE
    assert json.loads(path.read_text("utf-8")) == {"requested": 123.5}
    assert stop_requested(tmp_path)
    clear_stop_request(tmp_path)
    assert not stop_requested(tmp_path) and not path.exists()
    before = time.time()
    assert json.loads(request_stop(tmp_path).read_text("utf-8"))["requested"] >= before


def test_launch_dir_is_under_the_launches_folder(tmp_path):
    assert launch_dir(tmp_path, "20261119-101500-a1b2c3") == (
        tmp_path / ".launches" / "20261119-101500-a1b2c3"
    )


def test_spawn_detached_appends_output_and_stderr_to_the_log(tmp_path):
    log = tmp_path / "output.log"
    log.write_bytes(b"earlier\n")
    script = (
        "import os, sys; data = sys.stdin.read(); "
        "print('out', os.environ['SNAKEPPO_PROBE'], repr(data)); print(os.getcwd(), flush=True); "
        "print('err', file=sys.stderr)"
    )
    env = {**os.environ, "SNAKEPPO_PROBE": "probe-value"}
    child = spawn_detached([sys.executable, "-c", script], cwd=tmp_path, log_path=log, env=env)
    assert child.wait(timeout=30) == 0
    text = log.read_text("utf-8")
    assert text.startswith("earlier\n")
    lines = text.splitlines()
    assert lines[1] == "out probe-value ''"
    assert Path(lines[2]).samefile(tmp_path)
    assert "err" in lines[3:]
    log.unlink()  # the parent holds no handle on the log


def test_spawn_detached_uses_the_platform_detach_options(tmp_path, monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            seen.update(kwargs, argv=argv)

    monkeypatch.setattr(processes.subprocess, "Popen", FakePopen)
    spawn_detached(("a", "b"), cwd=tmp_path, log_path=tmp_path / "x.log")
    assert seen["argv"] == ["a", "b"] and seen["cwd"] == tmp_path and seen["env"] is None
    assert seen["stdin"] is subprocess.DEVNULL and seen["stderr"] is subprocess.STDOUT
    assert seen["stdout"].closed  # closed in the parent right after Popen
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        assert seen["creationflags"] == flags and "start_new_session" not in seen
    else:
        assert seen["start_new_session"] is True and "creationflags" not in seen


def test_child_creationflags_hide_the_console_only_on_windows():
    expected = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    assert child_creationflags() == expected


def test_terminate_ends_the_whole_process_tree(tmp_path):
    ready = tmp_path / "ready"
    grandchild = f"import pathlib, time; pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    )
    top = subprocess.Popen([sys.executable, "-c", parent])
    try:
        record = process_record(top.pid)
        _wait_for(ready.exists)
        family = [psutil.Process(top.pid), *psutil.Process(top.pid).children(recursive=True)]
        assert len(family) >= 2
        assert terminate(record, wait=5.0) is True
        assert not any(member.is_running() for member in family)
        assert is_alive(record) is False
    finally:
        if top.poll() is None:
            top.kill()
        top.wait(timeout=10)


def test_terminate_of_a_gone_process_is_done_at_once():
    child = subprocess.Popen(SLEEPER)
    record = process_record(child.pid)
    child.kill()
    child.wait(timeout=10)
    assert terminate(record, wait=0.1) is True
