import json
import threading
import urllib.error
import urllib.request
from datetime import datetime

import pytest

from snakeppo.config import Config, apply_overrides, save_config
from snakeppo.dashboard.api import json_safe
from snakeppo.dashboard.server import make_server, serve
from snakeppo.runs import (
    MetricsLogger,
    create_run_dir,
    truncate_metrics,
    update_run_info,
    write_run_info,
)


def _run(runs_dir, group, seed, *overrides, status="finished", evals=(1.0, 2.0, 3.0)):
    config = apply_overrides(Config(), [f"env.seed={seed}", *overrides])
    run_dir = create_run_dir(runs_dir, group or "solo", seed, now=datetime(2026, 11, 7, 9, 0))
    save_config(config, run_dir / "config.yaml")
    write_run_info(
        run_dir, name="x", agent="cnn", param_count=1, device="cpu", total_updates=3, group=group
    )
    update_run_info(run_dir, status=status)
    with MetricsLogger(run_dir, tensorboard=False) as logger:
        for update, score in enumerate(evals, start=1):
            logger.log(update, update * 10, {"eval/score_mean": score, "loss/value": 0.5})
    return run_dir


@pytest.fixture
def dashboard(tmp_path):
    """A server on a free port over ``tmp_path/runs``, with its own static folder."""
    runs = tmp_path / "runs"
    runs.mkdir()
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>dash</title>", "utf-8")
    (static / "app.js").write_text("export {};", "utf-8")
    server = make_server(runs, 0, static=static)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, runs
    server.shutdown()
    server.server_close()


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
    (listed,) = _json(server, "/api/runs")["runs"]
    assert listed["id"] == finished.name and listed["group"] == "g@1" and listed["seed"] == 1
    assert listed["status"] == "finished" and listed["update"] == 3 and listed["total_updates"] == 3
    assert listed["score"] == 2.0 and listed["switches"] == {"ppo.target_kl": 0.02}
    detail = _json(server, f"/api/runs/{finished.name}")
    assert detail["run"]["id"] == finished.name
    assert detail["config"]["ppo"]["target_kl"] == 0.02 and detail["info"]["agent"] == "cnn"


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


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD"])
def test_only_get_is_allowed(dashboard, method):
    server, _ = dashboard
    assert _status_of(server, "/api/runs", method) == 405


def test_the_server_never_writes_into_run_folders(dashboard):
    server, runs = dashboard
    run_dir = _run(runs, "g", 1)
    before = {p: p.stat().st_mtime_ns for p in run_dir.rglob("*")}
    for path in ("/api/runs", f"/api/runs/{run_dir.name}", "/api/results"):
        _json(server, path)
    _json(server, f"/api/runs/{run_dir.name}/metrics?since=0")
    assert {p: p.stat().st_mtime_ns for p in run_dir.rglob("*")} == before


def test_json_safe_replaces_non_finite_numbers():
    assert json_safe({"a": float("nan"), "b": (1.0, float("inf"))}) == {"a": None, "b": [1.0, None]}


def test_a_busy_port_is_reported_with_the_port_flag(dashboard):
    server, runs = dashboard
    with pytest.raises(ValueError, match=f"port {server.server_address[1]} is already in use"):
        serve(runs, server.server_address[1])
