"""A smoke run launched, stopped and resumed through the dashboard, as the browser does it.

Real server, real detached ``snakeppo train`` processes: the run the dashboard starts is the
CLI's own command, and it stops and resumes the same way a terminal run does.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from snakeppo.dashboard.server import make_server
from snakeppo.processes import LAUNCHES_DIR, STOP_REQUEST_FILE, terminate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# smoke.yaml at 32 updates instead of 4, so the run is still training when the stop arrives
LONG_SMOKE = ["train.total_env_steps=4096", "train.eval_interval=16"]
UPDATES = 32
WAIT_SECONDS = 180.0


@pytest.fixture
def live_dashboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # launched processes start here
    runs = tmp_path / "runs"
    server = make_server(runs, 0, configs_dir=CONFIGS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, runs
    server.shutdown()
    server.server_close()
    # a failed assertion must not leave a detached trainer behind
    for record in [*runs.glob(f"{LAUNCHES_DIR}/*/launch.json"), *runs.glob("*/run.json")]:
        process = json.loads(record.read_text("utf-8")).get("process")
        if process:
            terminate(process)


def _get(server, path):
    with urllib.request.urlopen(server.url.rstrip("/") + path, timeout=10) as response:
        return json.loads(response.read())


def _post(server, path, body):
    request = urllib.request.Request(
        server.url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Origin": server.url.rstrip("/"), "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _launch(server, launch_id):
    return next(
        entry for entry in _get(server, "/api/launches")["launches"] if entry["id"] == launch_id
    )


def _wait_for(what, check):
    """Poll ``check()`` until it returns something truthy; fail naming ``what`` on timeout."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.1)
    pytest.fail(f"timed out after {WAIT_SECONDS:.0f} s waiting for {what}")


def _run_of(server, launch_id):
    runs = _get(server, "/api/runs")["runs"]
    return next((run for run in runs if run["launch"] == launch_id), None)


def test_a_launched_run_stops_as_interrupted_and_resumes_to_the_end(live_dashboard):
    server, runs = live_dashboard
    body = {"kind": "train", "preset": "smoke.yaml", "overrides": LONG_SMOKE}
    status, reply = _post(server, "/api/launch", body)
    assert status == 202, reply
    launch_id = reply["launch"]["id"]

    running = _wait_for(
        "the launched run to start training",
        lambda: (run := _run_of(server, launch_id)) and run["status"] == "running" and run,
    )
    assert running["alive"] is True and isinstance(running["pid"], int)
    assert running["total_updates"] == UPDATES
    assert _launch(server, launch_id)["run"] == running["id"]

    assert _post(server, f"/api/runs/{running['id']}/stop", {}) == (202, {"ok": True})
    stopped = _wait_for(
        "the run to stop",
        lambda: (run := _run_of(server, launch_id)) and run["status"] != "running" and run,
    )
    assert stopped["status"] == "interrupted" and stopped["alive"] is None
    assert stopped["update"] < UPDATES
    run_dir = runs / stopped["id"]
    assert not (run_dir / STOP_REQUEST_FILE).exists()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    first = _wait_for(
        "the launch to end", lambda: (e := _launch(server, launch_id)) and not e["alive"] and e
    )
    assert first["state"] == "stopped"
    assert any("Interrupted." in line for line in first["log_tail"])

    status, reply = _post(server, f"/api/runs/{stopped['id']}/resume", {})
    assert status == 202, reply
    resume_id = reply["launch"]["id"]
    assert reply["launch"]["kind"] == "resume" and reply["launch"]["run"] == stopped["id"]
    finished = _wait_for(
        "the resumed run to finish",
        lambda: (run := _run_of(server, launch_id)) and run["status"] == "finished" and run,
    )
    assert finished["update"] == UPDATES and finished["alive"] is None
    resumed = _wait_for(
        "the resume launch to end",
        lambda: (e := _launch(server, resume_id)) and not e["alive"] and e,
    )
    assert resumed["state"] == "finished"
