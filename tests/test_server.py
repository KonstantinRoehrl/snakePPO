import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pytest

from snakeppo.builders import observation_record
from snakeppo.config import Config, apply_overrides, save_config
from snakeppo.dashboard import server as server_module
from snakeppo.dashboard.api import json_safe
from snakeppo.dashboard.launches import LaunchRequest, launch_argv
from snakeppo.dashboard.server import MAX_BODY_BYTES, make_server, serve, served_runs_dir
from snakeppo.processes import (
    LAUNCH_ENV,
    LAUNCHES_DIR,
    STOP_REQUEST_FILE,
    process_record,
    request_stop,
    terminate,
)
from snakeppo.results import BOOKKEEPING_KEYS
from snakeppo.runs import (
    MetricsLogger,
    create_run_dir,
    truncate_metrics,
    update_run_info,
    write_run_info,
)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


def _run(
    runs_dir,
    group,
    seed,
    *overrides,
    status="finished",
    evals=(1.0, 2.0, 3.0),
    process=None,
    launch=None,
):
    config = apply_overrides(Config(), [f"env.seed={seed}", *overrides])
    run_dir = create_run_dir(runs_dir, group or "solo", seed, now=datetime(2026, 11, 7, 9, 0))
    save_config(config, run_dir / "config.yaml")
    write_run_info(
        run_dir,
        name="x",
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
    with MetricsLogger(run_dir, tensorboard=False) as logger:
        for update, score in enumerate(evals, start=1):
            logger.log(update, update * 10, {"eval/score_mean": score, "loss/value": 0.5})
    return run_dir


class _FakeSpawn:
    """Stands in for ``processes.spawn_detached``: records each call and starts a sleeper.

    The sleeper is a real process, so launch records hold a live, stoppable pid that is never
    the test process itself (an escalation timer may terminate it).
    """

    def __init__(self):
        self.calls = []
        self.children = []

    def __call__(self, argv, *, cwd, log_path, env):
        self.calls.append({"argv": list(argv), "cwd": cwd, "log_path": log_path, "env": env})
        child = subprocess.Popen(SLEEPER)
        self.children.append(child)
        return child

    def close(self):
        for child in self.children:
            if child.poll() is None:
                terminate(process_record(child.pid))
            child.wait(timeout=30)


@pytest.fixture
def sleeper():
    """The process record of a live process that is not this one."""
    child = subprocess.Popen(SLEEPER)
    try:
        yield process_record(child.pid)
    finally:
        if child.poll() is None:
            terminate(process_record(child.pid))
        child.wait(timeout=30)


@pytest.fixture
def dashboard(tmp_path):
    """A server on a free port over ``tmp_path/runs``, with its own static and configs folders."""
    runs = tmp_path / "runs"
    runs.mkdir()
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>dash</title>", "utf-8")
    (static / "app.js").write_text("export {};", "utf-8")
    configs = tmp_path / "configs"
    (configs / "rewards").mkdir(parents=True)
    (configs / "demo.yaml").write_text(
        "train:\n  name: demo\n  description: Demo preset.\n", "utf-8"
    )
    (configs / "rewards" / "near.yaml").write_text(
        "reward:\n  strategy: distance\ntrain:\n  name: near\n  description: Near food.\n",
        "utf-8",
    )
    spawn = _FakeSpawn()
    server = make_server(runs, 0, static=static, configs_dir=configs, spawn=spawn)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, runs
    server.shutdown()
    server.server_close()
    spawn.close()


def _get(server, path):
    with urllib.request.urlopen(server.url.rstrip("/") + path, timeout=10) as response:
        return response.status, response.headers, response.read()


def _json(server, path):
    status, _, body = _get(server, path)
    assert status == 200
    return json.loads(body)


def _status_of(server, path, method="GET"):
    request = urllib.request.Request(server.url.rstrip("/") + path, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def _post(server, path, body=None, *, headers=None, data=None):
    """POST as the dashboard page does (own Origin, JSON body); returns (status, JSON reply)."""
    sent = {"Origin": server.url.rstrip("/"), "Content-Type": "application/json"}
    if headers is not None:
        sent = {key: value for key, value in (sent | headers).items() if value is not None}
    payload = data if data is not None else json.dumps({} if body is None else body).encode()
    request = urllib.request.Request(
        server.url.rstrip("/") + path, data=payload, headers=sent, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        reply = error.read()
        return error.code, json.loads(reply) if reply else None


def test_serves_the_static_frontend_and_bundled_plotly(dashboard):
    server, _ = dashboard
    status, headers, body = _get(server, "/")
    assert status == 200 and b"<title>dash</title>" in body
    assert headers["Content-Type"].startswith("text/html")
    status, headers, _ = _get(server, "/static/app.js")
    assert headers["Content-Type"].startswith("text/javascript")
    status, headers, body = _get(server, "/vendor/plotly.min.js")
    assert status == 200 and b"plotly.js" in body[:200] and len(body) > 1_000_000
    assert server.server_address[0] == "127.0.0.1"


def test_run_list_and_detail(dashboard):
    server, runs = dashboard
    finished = _run(runs, "g@1", 1, "ppo.target_kl=0.02")
    (runs / "not-a-run").mkdir()
    payload = _json(server, "/api/runs")
    assert payload["runs_dir"] == runs.as_posix()
    assert payload["runs_dir_resolved"] == runs.resolve().as_posix()
    (listed,) = payload["runs"]
    assert listed["id"] == finished.name and listed["group"] == "g@1" and listed["seed"] == 1
    assert listed["status"] == "finished" and listed["update"] == 3 and listed["total_updates"] == 3
    assert listed["score"] == 2.0 and listed["switches"] == {"ppo.target_kl": 0.02}
    assert 0 <= listed["quiet_seconds"] < 60 and listed["stall_seconds"] >= 60
    assert listed["alive"] is None and listed["pid"] is None and listed["launch"] is None
    assert listed["stop_requested"] is False
    assert listed["size_agnostic"] is True and listed["board_size"] == Config().env.board_size
    detail = _json(server, f"/api/runs/{finished.name}")
    assert detail["run"]["id"] == finished.name
    assert detail["config"]["ppo"]["target_kl"] == 0.02 and detail["info"]["agent"] == "cnn"


def test_run_summary_reports_the_training_process_and_its_stop_request(dashboard, sleeper):
    server, runs = dashboard
    run_dir = _run(runs, None, 1, status="running", process=sleeper, launch="20261119-090000-abc")
    (listed,) = _json(server, "/api/runs")["runs"]
    assert listed["status"] == "running" and listed["alive"] is True
    assert listed["pid"] == sleeper["pid"] and listed["launch"] == "20261119-090000-abc"
    assert listed["stop_requested"] is False
    request_stop(run_dir)
    assert _json(server, "/api/runs")["runs"][0]["stop_requested"] is True


def test_a_run_whose_process_is_gone_is_stalled_at_once(dashboard):
    server, runs = dashboard
    child = subprocess.Popen(SLEEPER)
    record = process_record(child.pid)
    terminate(record)
    child.wait(timeout=30)
    _run(runs, None, 1, status="running", process=record)
    (listed,) = _json(server, "/api/runs")["runs"]
    assert listed["status"] == "stalled" and listed["alive"] is False and listed["pid"] is None


def test_board_bound_agents_are_reported_as_such(dashboard):
    server, runs = dashboard
    _run(runs, None, 1, "agent.name=mlp", "env.board_size=10")
    (listed,) = _json(server, "/api/runs")["runs"]
    assert listed["size_agnostic"] is False and listed["board_size"] == 10


def test_old_runs_take_the_description_of_the_preset_with_their_name(dashboard):
    server, runs = dashboard
    for seed in (1, 2, 3):
        _run(runs, "demo@1", seed, "train.name=demo")
    _run(runs, None, 4, "train.name=demo", "train.description=Own text.")
    listed = {run["seed"]: run for run in _json(server, "/api/runs")["runs"]}
    assert listed[1]["description"] == "Demo preset." and listed[4]["description"] == "Own text."
    detail = _json(server, f"/api/runs/{listed[1]['id']}")
    assert detail["run"]["description"] == "Demo preset."
    (row,) = _json(server, "/api/results")["rows"]
    assert row["group"] == "demo@1" and row["description"] == "Demo preset."


def test_a_broken_preset_only_loses_the_description_fallback(dashboard, tmp_path):
    server, runs = dashboard
    _run(runs, None, 1, "train.name=demo")
    (tmp_path / "configs" / "broken.yaml").write_text("ppo: {nope: 1}\n", "utf-8")
    (listed,) = _json(server, "/api/runs")["runs"]
    assert listed["description"] == ""
    status, reply = 0, {}
    try:
        _get(server, "/api/presets")
    except urllib.error.HTTPError as error:
        status, reply = error.code, json.loads(error.read())
    assert status == 400 and "broken.yaml" in reply["error"] and "nope" in reply["error"]


def test_presets_lists_the_configs_folder_with_the_bookkeeping_keys(dashboard, tmp_path):
    server, _ = dashboard
    payload = _json(server, "/api/presets")
    assert payload["presets"] == [
        {"path": "demo.yaml", "name": "demo", "description": "Demo preset."},
        {"path": "rewards/near.yaml", "name": "near", "description": "Near food."},
    ]
    assert payload["bookkeeping_keys"] == sorted(BOOKKEEPING_KEYS)
    assert "train.description" in payload["bookkeeping_keys"]
    assert payload["configs_dir"] == (tmp_path / "configs").as_posix()


def test_the_configs_folder_defaults_to_configs_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = make_server(tmp_path / "runs", 0)
    try:
        assert server.cwd == tmp_path and server.configs_dir == tmp_path / "configs"
        assert server.spawn is None
    finally:
        server.server_close()


def test_launches_is_empty_before_anything_was_launched(dashboard):
    server, _ = dashboard
    assert _json(server, "/api/launches") == {"launches": []}


def test_launch_spawns_the_cli_command_and_lists_the_launch(dashboard):
    server, runs = dashboard
    spawn = server.spawn
    body = {
        "kind": "train",
        "preset": "demo.yaml",
        "overrides": ["ppo.lr=0.001"],
        "description": "Demo preset. Changed: ppo.lr=0.001",
    }
    status, reply = _post(server, "/api/launch", body)
    assert status == 202
    entry = reply["launch"]
    assert entry["kind"] == "train" and entry["preset"] == "demo.yaml"
    assert entry["overrides"] == ["ppo.lr=0.001"] and entry["alive"] is True
    assert entry["state"] == "running"
    (call,) = spawn.calls
    request = LaunchRequest(
        "train", "demo.yaml", ("ppo.lr=0.001",), "Demo preset. Changed: ppo.lr=0.001"
    )
    expected = launch_argv(request, configs_dir=server.configs_dir, runs_dir=runs, cwd=server.cwd)
    assert call["argv"] == expected
    assert call["argv"][1:4] == ["-m", "snakeppo.cli", "train"]
    assert call["argv"][call["argv"].index("--description") + 1] == body["description"]
    assert call["cwd"] == server.cwd and call["env"][LAUNCH_ENV] == entry["id"]
    assert call["log_path"].parent == runs / LAUNCHES_DIR / entry["id"]
    (listed,) = _json(server, "/api/launches")["launches"]
    assert listed["id"] == entry["id"] and listed["state"] == "running"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"kind": "train", "preset": "demo.yaml", "overrides": ["ppo.nope=1"]}, "ppo.nope"),
        ({"kind": "train", "preset": "../secret.yaml"}, "secret.yaml"),
        ({"kind": "train", "preset": "missing.yaml"}, "missing.yaml"),
        ({"kind": "dance", "preset": "demo.yaml"}, "kind"),
    ],
)
def test_an_invalid_launch_is_refused_without_spawning(dashboard, tmp_path, body, message):
    server, _ = dashboard
    (tmp_path / "secret.yaml").write_text("train:\n  name: secret\n", "utf-8")
    status, reply = _post(server, "/api/launch", body)
    assert status == 400 and message in reply["error"]
    assert server.spawn.calls == []
    assert _json(server, "/api/launches") == {"launches": []}


def test_stop_asks_a_running_run_to_stop(dashboard, sleeper):
    server, runs = dashboard
    run_dir = _run(runs, None, 1, status="running", process=sleeper)
    assert _post(server, f"/api/runs/{run_dir.name}/stop") == (202, {"ok": True})
    assert (run_dir / STOP_REQUEST_FILE).is_file()
    assert _json(server, "/api/runs")["runs"][0]["stop_requested"] is True


def test_stop_and_resume_refuse_a_run_in_the_wrong_state(dashboard, sleeper):
    server, runs = dashboard
    finished = _run(runs, None, 1)
    status, reply = _post(server, f"/api/runs/{finished.name}/stop")
    assert status == 409 and "finished" in reply["error"]
    assert _post(server, f"/api/runs/{finished.name}/resume")[0] == 409
    running = _run(runs, None, 2, status="running", process=sleeper)
    assert _post(server, f"/api/runs/{running.name}/resume")[0] == 409
    assert server.spawn.calls == []
    assert _post(server, "/api/runs/nope/stop")[0] == 404
    assert _post(server, "/api/runs/..%2F..%2Fsecret/resume")[0] == 404


def test_resume_continues_an_interrupted_run_through_the_cli(dashboard):
    server, runs = dashboard
    run_dir = _run(runs, None, 1, status="interrupted")
    status, reply = _post(server, f"/api/runs/{run_dir.name}/resume")
    assert status == 202
    entry = reply["launch"]
    assert entry["kind"] == "resume" and entry["run"] == run_dir.name
    (call,) = server.spawn.calls
    argv = call["argv"]
    assert argv[0] == sys.executable
    assert argv[1:5] == ["-m", "snakeppo.cli", "train", "--resume"]
    assert len(argv) == 6 and os.path.samefile(argv[5], run_dir)


def test_a_sweep_launch_stops_and_resumes_as_one_group(dashboard):
    server, runs = dashboard
    body = {"kind": "sweep", "preset": "demo.yaml", "seeds": [1, 2], "jobs": 2}
    status, reply = _post(server, "/api/launch", body)
    assert status == 202
    sweep = reply["launch"]
    assert sweep["kind"] == "sweep" and sweep["seeds"] == [1, 2] and sweep["jobs"] == 2
    assert server.spawn.calls[0]["argv"][1:4] == ["-m", "snakeppo.cli", "sweep"]
    assert _post(server, f"/api/launches/{sweep['id']}/resume")[0] == 409  # still running
    assert _post(server, f"/api/launches/{sweep['id']}/stop") == (202, {"ok": True})
    assert (runs / LAUNCHES_DIR / sweep["id"] / STOP_REQUEST_FILE).is_file()
    (listed,) = _json(server, "/api/launches")["launches"]
    assert listed["state"] == "stopping"
    server.spawn.close()  # the sweep process is gone
    status, reply = _post(server, f"/api/launches/{sweep['id']}/resume")
    assert status == 202
    resumed = reply["launch"]
    assert resumed["resumes"] == sweep["id"] and resumed["kind"] == "sweep"
    assert resumed["group"] == sweep["group"] and resumed["seeds"] == [1, 2]


def test_launch_routes_refuse_unknown_ids_and_train_launches(dashboard):
    server, _ = dashboard
    for launch_id in ("nope", "..", "..%2F..%2Fsecret"):
        assert _post(server, f"/api/launches/{launch_id}/stop")[0] == 404
        assert _post(server, f"/api/launches/{launch_id}/resume")[0] == 404
    _, reply = _post(server, "/api/launch", {"kind": "train", "preset": "demo.yaml"})
    train_id = reply["launch"]["id"]
    assert _post(server, f"/api/launches/{train_id}/stop")[0] == 409
    assert _post(server, f"/api/launches/{train_id}/resume")[0] == 409
    assert _post(server, "/api/nowhere")[0] == 404


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": None},
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "https://127.0.0.1"},
        {"Origin": None, "Referer": "http://evil.example/page"},
    ],
)
def test_a_post_from_another_origin_is_forbidden_before_any_work(dashboard, headers):
    server, _ = dashboard
    status, reply = _post(
        server, "/api/launch", {"kind": "train", "preset": "demo.yaml"}, headers=headers
    )
    assert status == 403 and "Origin" in reply["error"]
    assert server.spawn.calls == []


def test_the_own_origin_is_accepted_as_localhost_or_by_referer(dashboard):
    server, _ = dashboard
    port = server.server_address[1]
    for headers in (
        {"Origin": f"http://localhost:{port}"},
        {"Origin": None, "Referer": f"http://127.0.0.1:{port}/#tab=launch"},
    ):
        assert _post(server, "/api/runs/nope/stop", headers=headers)[0] == 404


def test_another_port_on_the_same_host_is_another_origin(dashboard):
    server, _ = dashboard
    other = server.server_address[1] + 1
    headers = {"Origin": f"http://127.0.0.1:{other}"}
    assert _post(server, "/api/runs/nope/stop", headers=headers)[0] == 403


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"],
)
def test_a_post_that_is_not_json_is_forbidden(dashboard, content_type):
    server, _ = dashboard
    status, reply = _post(server, "/api/runs/nope/stop", headers={"Content-Type": content_type})
    assert status == 403 and "application/json" in reply["error"]


def test_the_guards_run_in_order_origin_type_size_json(dashboard):
    server, _ = dashboard
    big = b" " * (MAX_BODY_BYTES + 1)
    assert (
        _post(server, "/api/launch", data=big, headers={"Origin": "http://evil.example"})[0] == 403
    )
    assert _post(server, "/api/launch", data=big, headers={"Content-Type": "text/plain"})[0] == 403
    status, reply = _post(server, "/api/launch", data=big)
    assert status == 413 and str(MAX_BODY_BYTES) in reply["error"]
    status, reply = _post(server, "/api/launch", data=b"{not json")
    assert status == 400 and "JSON" in reply["error"]
    status, reply = _post(server, "/api/launch", data=b"[1, 2]")
    assert status == 400 and "JSON object" in reply["error"]
    content_type = {"Content-Type": "application/json; charset=utf-8"}
    assert _post(server, "/api/runs/nope/stop", headers=content_type)[0] == 404
    assert server.spawn.calls == []


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_only_get_and_post_are_allowed(dashboard, method):
    server, _ = dashboard
    request = urllib.request.Request(server.url.rstrip("/") + "/api/runs", method=method)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 405 and error.value.headers["Allow"] == "GET, POST"


def test_get_routes_never_write_into_run_folders(dashboard):
    server, runs = dashboard
    run_dir = _run(runs, "g", 1)
    before = {p: p.stat().st_mtime_ns for p in run_dir.rglob("*")}
    for path in ("/api/runs", f"/api/runs/{run_dir.name}", "/api/results", "/api/launches"):
        _json(server, path)
    _json(server, f"/api/runs/{run_dir.name}/metrics?since=0")
    assert {p: p.stat().st_mtime_ns for p in run_dir.rglob("*")} == before


def test_an_unreadable_run_is_reported_not_fatal(dashboard):
    server, runs = dashboard
    good = _run(runs, None, 1)
    broken = _run(runs, None, 2)
    (broken / "config.yaml").write_text("ppo: {nope: 1}\n", "utf-8")
    payload = _json(server, "/api/runs")
    assert [run["id"] for run in payload["runs"]] == [good.name]
    assert payload["skipped"][0]["id"] == broken.name and "nope" in payload["skipped"][0]["error"]


def test_metrics_are_served_incrementally(dashboard):
    server, runs = dashboard
    run_dir = _run(runs, None, 1)
    first = _json(server, f"/api/runs/{run_dir.name}/metrics?since=0")
    assert [line["update"] for line in first["lines"]] == [1, 2, 3] and not first["reset"]
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"update": 4, "env_steps": 40, "wall_time": 4.0}\n{"update": 5')
    second = _json(server, f"/api/runs/{run_dir.name}/metrics?since={first['offset']}")
    assert [line["update"] for line in second["lines"]] == [4]  # partial line 5 held back
    truncate_metrics(run_dir, after_update=1)
    reset = _json(server, f"/api/runs/{run_dir.name}/metrics?since={second['offset']}")
    assert reset["reset"] and [line["update"] for line in reset["lines"]] == [1]
    assert _status_of(server, f"/api/runs/{run_dir.name}/metrics?since=-1") == 400
    assert _status_of(server, f"/api/runs/{run_dir.name}/metrics?since=x") == 400


def test_results_and_diff_use_the_shared_protocol(dashboard):
    server, runs = dashboard
    for seed in (1, 2, 3):
        _run(runs, "base", seed, evals=(1.0, 1.0, 1.0))
        _run(runs, "kl", seed, "ppo.target_kl=0.02", evals=(2.0, 2.0, 2.0))
    payload = _json(server, "/api/results")
    rows = {row["group"]: row for row in payload["rows"]}
    assert rows["base"]["verdict"] == "reference"
    assert rows["kl"]["verdict"] == "better" and rows["kl"]["budget"] == 10_000_000
    assert rows["kl"]["differences"] == {"ppo.target_kl": [None, 0.02]}
    assert _json(server, "/api/results?baseline=kl")["baseline"] == "kl"
    assert _status_of(server, "/api/results?baseline=nope") == 400
    names = sorted(p.name for p in runs.iterdir())
    diff = _json(server, f"/api/diff?base={names[0]}&runs={names[3]},{names[0]}")
    assert diff["differences"][names[3]] == {"ppo.target_kl": [None, 0.02]}
    assert diff["differences"][names[0]] == {}


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs/nope",
        "/api/runs/..%2F..%2Fsecret",
        "/api/runs/%2E%2E",
        "/api/runs/C:%5Cwindows",
        "/static/..%2Fsecret.html",
        "/static/missing.js",
        "/static/notes.txt",
        "/nowhere",
    ],
)
def test_unknown_and_traversal_paths_are_404(dashboard, path):
    server, runs = dashboard
    (runs.parent / "secret").mkdir()
    (runs.parent / "secret" / "run.json").write_text("{}", "utf-8")
    assert _status_of(server, path) == 404


def test_json_safe_replaces_non_finite_numbers():
    assert json_safe({"a": float("nan"), "b": (1.0, float("inf"))}) == {"a": None, "b": [1.0, None]}


def test_a_busy_port_is_reported_with_the_port_flag(dashboard):
    server, runs = dashboard
    with pytest.raises(ValueError, match=f"port {server.server_address[1]} is already in use"):
        serve(runs, server.server_address[1])


def test_serve_opens_the_given_page_and_names_a_missing_configs_folder(
    tmp_path, monkeypatch, capsys
):
    opened = []
    monkeypatch.setattr(server_module.webbrowser, "open", opened.append)

    def interrupted(self, poll_interval=0.5):
        raise KeyboardInterrupt

    monkeypatch.setattr(server_module.DashboardServer, "serve_forever", interrupted)
    missing = tmp_path / "no-configs"
    assert serve(tmp_path, 0, open_browser=True, configs_dir=missing, open_path="#tab=replay") == 0
    (url,) = opened
    assert url.startswith("http://127.0.0.1:") and url.endswith("/#tab=replay")
    out = capsys.readouterr().out
    assert f"{missing} does not exist" in out and "--configs-dir" in out


def test_served_runs_dir_finds_the_folder_a_dashboard_serves(dashboard):
    server, runs = dashboard
    assert served_runs_dir(server.server_address[1]) == runs.resolve()


def test_served_runs_dir_is_none_when_no_dashboard_answers(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert served_runs_dir(free_port, timeout=0.5) is None

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass

    other = HTTPServer(("127.0.0.1", 0), lambda *a: Quiet(*a, directory=str(tmp_path)))
    thread = threading.Thread(target=other.serve_forever, daemon=True)
    thread.start()
    try:
        assert served_runs_dir(other.server_address[1]) is None
    finally:
        other.shutdown()
        other.server_close()
