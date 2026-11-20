"""Replay over HTTP: create a session, steer it, and read its frames as Server-Sent Events."""

import http.client
import json
import threading
from pathlib import Path

import pytest
import torch

from snakeppo.config import load_config
from snakeppo.dashboard.replay import BoardSpec
from snakeppo.dashboard.server import make_server
from snakeppo.ppo.trainer import train

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One finished smoke run (an 8x8 ``cnn`` with best.pt and last.pt) alone in a runs folder."""
    runs = tmp_path_factory.mktemp("replay") / "runs"
    return train(load_config(SMOKE, [f"train.runs_dir={runs.as_posix()}"]), progress=False)


@pytest.fixture(autouse=True)
def _keep_torch_threads():
    """Creating a session lowers torch's thread count for the whole process; put it back."""
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def dashboard(smoke_run):
    server = make_server(smoke_run.parent, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _connect(server):
    return http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=30)


def _post(server, path, body, *, origin=True):
    """POST JSON the way the page does; returns ``(status, decoded reply)``."""
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = server.url.rstrip("/")
    connection = _connect(server)
    try:
        connection.request("POST", path, json.dumps(body), headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _get_status(server, path):
    connection = _connect(server)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def _create(server, run_dir, **options):
    status, reply = _post(server, "/api/replay", {"boards": [{"run": run_dir.name}], **options})
    assert status == 201, reply
    return reply["session"]


def _open_stream(server, session_id):
    connection = _connect(server)
    connection.request("GET", f"/api/replay/{session_id}/stream")
    return connection, connection.getresponse()


def _next_event(response):
    """The next SSE event as ``{field: value}``, skipping comments; None once the stream ends."""
    event = {}
    while True:
        line = response.readline().decode("utf-8")
        if not line:
            return event or None
        line = line.rstrip("\r\n")
        if not line:
            if event:
                return event
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            event[field] = value.removeprefix(" ")


def _next_frame(response):
    event = _next_event(response)
    assert event is not None and "event" not in event, event
    return json.loads(event["data"])


def test_a_session_streams_one_frame_per_step_for_every_board(dashboard, smoke_run):
    boards = [
        {"run": smoke_run.name},
        {"run": smoke_run.name, "checkpoint": "last", "greedy": False, "board_size": 12},
    ]
    status, reply = _post(dashboard, "/api/replay", {"boards": boards, "speed": 60})
    assert status == 201
    session = reply["session"]
    assert session["speed"] == 60 and session["paused"] is False
    assert [
        (b["run"], b["checkpoint"], b["greedy"], b["board_size"], b["agent"])
        for b in session["boards"]
    ] == [(smoke_run.name, "best", True, 8, "cnn"), (smoke_run.name, "last", False, 12, "cnn")]

    connection, response = _open_stream(dashboard, session["id"])
    try:
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/event-stream")
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("Content-Length") is None
        frames = [_next_frame(response) for _ in range(3)]
    finally:
        response.close()
        connection.close()
    seqs = [frame["seq"] for frame in frames]
    assert seqs == sorted(set(seqs))  # strictly increasing
    assert [board["size"] for board in frames[-1]["boards"]] == [8, 12]
    for board in frames[-1]["boards"]:
        assert len(board["cells"]) == board["size"] ** 2 and set(board["cells"]) <= set("01234")
        assert board["cells"][board["body"][0]] == "3" and len(board["body"]) == board["length"]


def test_controls_steer_the_session_and_a_paused_stream_shows_each_step(dashboard, smoke_run):
    session = _create(dashboard, smoke_run, speed=60)
    control = f"/api/replay/{session['id']}/control"
    assert _post(dashboard, control, {"action": "pause"}) == (200, {"paused": True, "speed": 60})
    assert _post(dashboard, control, {"action": "speed", "value": 5}) == (
        200,
        {"paused": True, "speed": 5},
    )
    connection, response = _open_stream(dashboard, session["id"])
    try:
        shown = _next_frame(response)
        assert shown["paused"] is True and shown["speed"] == 5
        assert _post(dashboard, control, {"action": "step"})[0] == 200
        assert _next_frame(response)["seq"] > shown["seq"]
    finally:
        response.close()
        connection.close()
    status, reply = _post(dashboard, control, {"action": "fly"})
    assert status == 400 and "fly" in reply["error"]
    assert _post(dashboard, control, {"action": "speed", "value": 7})[0] == 400


def test_unknown_sessions_are_404(dashboard):
    assert _post(dashboard, "/api/replay/nope/control", {"action": "pause"})[0] == 404
    assert _get_status(dashboard, "/api/replay/nope/stream") == 404


def test_a_new_session_ends_the_old_stream(dashboard, smoke_run):
    first = _create(dashboard, smoke_run)
    connection, response = _open_stream(dashboard, first["id"])
    try:
        _next_frame(response)
        _create(dashboard, smoke_run)
        events = iter(lambda: _next_event(response), None)
        ending = next(event for event in events if event.get("event") == "end")
        assert ending == {"event": "end", "data": "{}"}
        assert _next_event(response) is None  # the server closed the stream
    finally:
        response.close()
        connection.close()
    assert _get_status(dashboard, f"/api/replay/{first['id']}/stream") == 404
    assert _post(dashboard, f"/api/replay/{first['id']}/control", {"action": "pause"})[0] == 404


@pytest.mark.parametrize(
    ("body", "status", "named"),
    [
        (lambda run: [{"run": run}], 400, "object"),
        (lambda run: {"boards": []}, 400, "boards"),
        (lambda run: {"boards": [{"run": "nope"}]}, 404, "nope"),
        (lambda run: {"boards": [{"run": "../runs"}]}, 404, "../runs"),
        (lambda run: {"boards": [{"run": run, "checkpoint": "middle"}]}, 400, "middle"),
        (lambda run: {"boards": [{"run": run, "board_size": 4}]}, 400, "at least 5"),
        (lambda run: {"boards": [{"run": run}], "speed": 7}, 400, "speed"),
        (lambda run: {"boards": [{"run": run}], "speed": True}, 400, "speed"),
    ],
    ids=[
        "not-an-object",
        "no-boards",
        "unknown-run",
        "outside-run",
        "bad-checkpoint",
        "tiny-board",
        "bad-speed",
        "boolean-speed",
    ],
)
def test_a_bad_replay_request_names_the_problem(dashboard, smoke_run, body, status, named):
    got, reply = _post(dashboard, "/api/replay", body(smoke_run.name))
    assert got == status and named in reply["error"]


def test_replay_routes_take_the_same_origin_check_as_every_post(dashboard, smoke_run):
    body = {"boards": [{"run": smoke_run.name}]}
    assert _post(dashboard, "/api/replay", body, origin=False)[0] == 403


def test_a_stream_that_goes_away_detaches_without_a_traceback(
    dashboard, smoke_run, monkeypatch, capsys
):
    session = _create(dashboard, smoke_run, speed=60)
    live = dashboard.replays.get(session["id"])
    detached = threading.Event()
    detach = live.detach

    def recording_detach():
        detach()
        detached.set()

    monkeypatch.setattr(live, "detach", recording_detach)
    connection, response = _open_stream(dashboard, session["id"])
    _next_frame(response)
    response.close()
    connection.close()
    assert detached.wait(10)
    assert not live.closed  # the idle clock closes it later, not the disconnect
    assert "Traceback" not in capsys.readouterr().err


def test_closing_the_server_closes_its_replay_session(smoke_run):
    server = make_server(smoke_run.parent, 0)
    session = server.replays.create([(BoardSpec(smoke_run.name), smoke_run)])
    server.server_close()
    assert session.closed and server.replays.get(session.id) is None
