"""The dashboard's HTTP server: static frontend, bundled plotly.js and the JSON API.

GET routes::

    /                               static/index.html
    /static/<file>                  a file directly inside static/
    /vendor/plotly.min.js           the plotly.js bundled in the ``plotly`` Python package
    /api/runs                       {"runs": [summary], "skipped": [{"id", "error"}],
                                     "runs_dir": str, "runs_dir_resolved": str}
    /api/runs/<id>                  {"run": summary, "config": {...}, "info": {...}}
    /api/runs/<id>/metrics?since=N  {"lines": [...], "offset": int, "reset": bool}
    /api/results[?baseline=G]       {"rows": [...], "left_out": int, "baseline": str | null}
    /api/diff?base=<id>&runs=<id>,<id>
                                    {"base": id, "differences": {id: {key: [base, run]}}}
    /api/presets                    {"presets": [{"path", "name", "description"}],
                                     "bookkeeping_keys": [str], "configs_dir": str}
    /api/launches                   {"launches": [launch entry]}, newest first
    /api/replay/<id>/stream         text/event-stream: "data: <frame JSON>" per step,
                                     ": keepalive" while nothing moves, "event: end" at the close

POST routes (launch, stop and resume answer 202 Accepted: the work goes on in a detached
process; replay answers at once)::

    /api/launch                     body: a launch request; {"launch": entry}
    /api/runs/<id>/stop             {"ok": true}
    /api/runs/<id>/resume           {"launch": entry}
    /api/launches/<id>/stop         {"ok": true} (a sweep launch)
    /api/launches/<id>/resume       {"launch": entry} (a stopped sweep launch)
    /api/replay                     body: {"boards": [{"run", "checkpoint"?, "greedy"?,
                                     "board_size"?}], "speed"?: n}; 201 {"session": {...}},
                                     ending any older session
    /api/replay/<id>/control        body: {"action", "value"?}; 200 {"paused": bool, "speed": int}

Every POST must come from the dashboard page itself: an ``Origin`` (or, without one, a
``Referer``) of this server and ``Content-Type: application/json``, or it gets 403 before any
work, so another web page open in the same browser cannot start or stop processes. Other
methods get 405. Run ids are folder names of direct children of the runs folder that hold a
``run.json``; launch ids are folder names under ``<runs>/.launches``. Errors are
``{"error": text}``: 400 bad request, 403 refused origin or content type, 404 unknown id,
409 the run or launch is not in a state that allows the action, 413 body too large.
"""

from __future__ import annotations

import errno
import http.client
import importlib.resources
import json
import re
import sys
import traceback
import webbrowser
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

from snakeppo.config import DEFAULT_CONFIGS_DIR, ConfigError, Preset, list_presets
from snakeppo.dashboard.api import (
    diff_payload,
    json_safe,
    metrics_payload,
    presets_payload,
    results_payload,
    run_detail,
    run_summary,
)
from snakeppo.dashboard.launches import (
    LAUNCH_FILE,
    LaunchConflict,
    LaunchError,
    Spawn,
    launch_folder,
    list_launches,
    parse_launch_request,
    resume_run,
    resume_sweep,
    start_launch,
    stop_run,
    stop_sweep,
)
from snakeppo.dashboard.replay import (
    DEFAULT_SPEED,
    REPLAY_SPEEDS,
    ReplayError,
    ReplayManager,
    ReplaySession,
    parse_board_specs,
)
from snakeppo.runs import RUN_INFO_FILE, RunRecord, find_runs, load_run, read_metrics_since

DEFAULT_PORT = 8765
HOST = "127.0.0.1"
MAX_BODY_BYTES = 64 * 1024
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}
_STATIC_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")
_ADDRESS_IN_USE = {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", errno.EADDRINUSE)}
# Errors that make one run folder unreadable without making the whole listing fail.
_UNREADABLE = (OSError, ValueError, KeyError, yaml.YAMLError)
# A stream sends an SSE comment this often while no frame comes, so a page that went away is
# noticed (the write fails) even while the replay is paused.
STREAM_KEEPALIVE_SECONDS = 15.0
# What writing to a page that closed its stream raises (Windows reports an aborted connection).
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
# A refused body is read up to this size so the client gets the error reply, not a reset.
_DISCARD_LIMIT = 16 * MAX_BODY_BYTES


def static_dir() -> Traversable:
    return importlib.resources.files("snakeppo.dashboard") / "static"


def plotly_js() -> Traversable:
    return importlib.resources.files("plotly") / "package_data" / "plotly.min.js"


class DashboardServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already in use.
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True

    def __init__(
        self,
        runs_dir: Path,
        port: int,
        static: Traversable,
        *,
        configs_dir: Path | None = None,
        spawn: Spawn | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.static = static
        # launched processes start here, so a relative runs folder means the same folder to them
        self.cwd = Path.cwd()
        self.configs_dir = self.cwd / (DEFAULT_CONFIGS_DIR if configs_dir is None else configs_dir)
        self.spawn = spawn
        self.replays = ReplayManager()
        super().__init__((HOST, port), _Handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"

    @property
    def origins(self) -> tuple[str, str]:
        """The origins the dashboard page has when opened from this server."""
        port = self.server_address[1]
        return f"http://{HOST}:{port}", f"http://localhost:{port}"

    def server_close(self) -> None:
        self.replays.close()
        super().server_close()


def make_server(
    runs_dir: Path,
    port: int = DEFAULT_PORT,
    *,
    static: Traversable | None = None,
    configs_dir: Path | None = None,
    spawn: Spawn | None = None,
) -> DashboardServer:
    """Bind the server (``port=0`` picks a free one); raises ``OSError`` if the port is busy.

    ``configs_dir`` (default ``configs`` in the working directory) holds the presets the Launch
    view offers; ``spawn`` replaces the detached process start, for tests.
    """
    return DashboardServer(
        runs_dir,
        port,
        static if static is not None else static_dir(),
        configs_dir=configs_dir,
        spawn=spawn,
    )


def serve(
    runs_dir: Path,
    port: int = DEFAULT_PORT,
    *,
    open_browser: bool = False,
    configs_dir: Path | None = None,
    open_path: str = "",
) -> int:
    """Serve until Ctrl+C; a busy port is reported as a ``ValueError`` naming ``--port``.

    With ``open_browser`` the browser opens ``server.url + open_path`` (e.g. ``"#tab=replay"``).
    """
    try:
        server = make_server(runs_dir, port, configs_dir=configs_dir)
    except OSError as error:
        if error.errno in _ADDRESS_IN_USE:
            raise ValueError(
                f"port {port} is already in use; pick another one with --port"
            ) from error
        raise
    if not Path(runs_dir).is_dir():
        print(f"Note: {runs_dir} does not exist yet; runs appear once training writes them.")
    if not server.configs_dir.is_dir():
        print(
            f"Note: {server.configs_dir} does not exist, so the Launch view lists no presets; "
            "pick the preset folder with --configs-dir."
        )
    print(f"Dashboard for {runs_dir} at {server.url} (Ctrl+C stops it)")
    if open_browser:
        webbrowser.open(server.url + open_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped the dashboard.")
    finally:
        server.server_close()
    return 0


def served_runs_dir(port: int, timeout: float = 1.0) -> Path | None:
    """The runs folder the dashboard on ``port`` serves, resolved.

    ``None`` when nothing answers there or the answer is not a dashboard's run list.
    """
    # http.client rather than urllib: a configured HTTP proxy must not see a local request
    connection = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        connection.request("GET", "/api/runs")
        reply = connection.getresponse()
        if reply.status != HTTPStatus.OK:
            return None
        payload = json.loads(reply.read())
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        return None
    runs_dir = payload.get("runs_dir_resolved", payload.get("runs_dir"))
    return Path(runs_dir).resolve() if isinstance(runs_dir, str) else None


class _HttpError(Exception):
    """A request the server refuses; the message goes to the page as ``{"error": ...}``."""

    status = HTTPStatus.BAD_REQUEST


class _BadRequest(_HttpError):
    status = HTTPStatus.BAD_REQUEST


class _Forbidden(_HttpError):
    status = HTTPStatus.FORBIDDEN


class _NotFound(_HttpError):
    status = HTTPStatus.NOT_FOUND


class _TooLarge(_HttpError):
    status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE


_Route = Callable[[list[str], dict[str, str]], None]


class _Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    # ------------------------------------------------------------ dispatch
    def do_GET(self) -> None:
        self._dispatch(self._route)

    def do_POST(self) -> None:
        self._dispatch(self._route_post)

    def _not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _not_allowed

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # the page polls every two seconds; per-request lines would flood the terminal

    def _dispatch(self, route: _Route) -> None:
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        parts = [unquote(part) for part in url.path.split("/") if part]
        try:
            route(parts, query)
        except _HttpError as error:
            self._send_json({"error": str(error) or error.status.phrase}, error.status)
        except LaunchConflict as error:
            self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)
        except LaunchError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # report it to the page instead of dropping the connection
            traceback.print_exc()
            self._send_json({"error": repr(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route(self, parts: list[str], query: dict[str, str]) -> None:
        match parts:
            case []:
                self._send_static("index.html")
            case ["static", name]:
                self._send_static(name)
            case ["vendor", "plotly.min.js"]:
                self._send_file(plotly_js(), CONTENT_TYPES[".js"])
            case ["api", "runs"]:
                self._send_json(self._run_list())
            case ["api", "runs", run_id]:
                self._send_json(run_detail(self._load(run_id), self._description_presets()))
            case ["api", "runs", run_id, "metrics"]:
                chunk = read_metrics_since(self._run_dir(run_id), self._offset(query))
                self._send_json(metrics_payload(chunk))
            case ["api", "results"]:
                baseline = query.get("baseline") or None
                try:
                    payload = results_payload(
                        self._records(), baseline, self._description_presets()
                    )
                except ValueError as error:
                    raise _BadRequest(str(error)) from error
                self._send_json(payload)
            case ["api", "diff"]:
                base = self._load(query.get("base", ""))
                others = [self._load(rid) for rid in query.get("runs", "").split(",") if rid]
                self._send_json(diff_payload(base, others))
            case ["api", "presets"]:
                self._send_json(presets_payload(self._presets(), self.server.configs_dir))
            case ["api", "launches"]:
                self._send_json({"launches": list_launches(self.server.runs_dir)})
            case ["api", "replay", session_id, "stream"]:
                self._stream_replay(self._replay(session_id))
            case _:
                raise _NotFound(f"no route {'/' + '/'.join(parts)}")

    def _route_post(self, parts: list[str], query: dict[str, str]) -> None:
        body = self._json_body()
        server = self.server
        context = {"runs_dir": server.runs_dir, "cwd": server.cwd, "spawn": server.spawn}
        match parts:
            case ["api", "launch"]:
                request = parse_launch_request(body)
                entry = start_launch(request, configs_dir=server.configs_dir, **context)
                self._send_json({"launch": entry}, HTTPStatus.ACCEPTED)
            case ["api", "runs", run_id, "stop"]:
                stop_run(self._run_dir(run_id))
                self._send_json({"ok": True}, HTTPStatus.ACCEPTED)
            case ["api", "runs", run_id, "resume"]:
                entry = resume_run(self._run_dir(run_id), **context)
                self._send_json({"launch": entry}, HTTPStatus.ACCEPTED)
            case ["api", "launches", launch_id, "stop"]:
                stop_sweep(self._launch_id(launch_id), runs_dir=server.runs_dir)
                self._send_json({"ok": True}, HTTPStatus.ACCEPTED)
            case ["api", "launches", launch_id, "resume"]:
                launch_id = self._launch_id(launch_id)
                entry = resume_sweep(launch_id, configs_dir=server.configs_dir, **context)
                self._send_json({"launch": entry}, HTTPStatus.ACCEPTED)
            case ["api", "replay"]:
                session = self._create_replay(body)
                self._send_json({"session": session.describe()}, HTTPStatus.CREATED)
            case ["api", "replay", session_id, "control"]:
                session = self._replay(session_id)
                try:
                    result = session.control(body.get("action"), body.get("value"))
                except ReplayError as error:
                    raise _BadRequest(str(error)) from error
                self._send_json(result)
            case _:
                raise _NotFound(f"no route {'/' + '/'.join(parts)}")

    # ------------------------------------------------------------ request bodies
    def _json_body(self) -> dict[str, Any]:
        """The request's JSON object, read only after the origin and content-type checks."""
        try:
            length: int | None = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = None
        if length is not None and length < 0:
            length = None
        if self._origin() not in self.server.origins:
            self._discard(length)
            raise _Forbidden(
                "POST requests must come from the dashboard page itself "
                f"(Origin or Referer {self.server.origins[0]})"
            )
        if self.headers.get_content_type() != "application/json":
            self._discard(length)
            raise _Forbidden("POST requests must send Content-Type: application/json")
        if length is None:
            raise _BadRequest("Content-Length must be the body size in bytes")
        if length > MAX_BODY_BYTES:
            self._discard(length)
            raise _TooLarge(f"request body is {length} bytes; the limit is {MAX_BODY_BYTES}")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError as error:
            raise _BadRequest(f"request body is not valid JSON: {error}") from error
        if not isinstance(body, dict):
            raise _BadRequest("request body must be a JSON object")
        return body

    def _origin(self) -> str:
        origin = self.headers.get("Origin")
        if origin is not None:
            return origin
        referer = urlsplit(self.headers.get("Referer", ""))
        return f"{referer.scheme}://{referer.netloc}"

    def _discard(self, length: int | None) -> None:
        remaining = min(length or 0, _DISCARD_LIMIT)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, MAX_BODY_BYTES))
            if not chunk:
                return
            remaining -= len(chunk)

    # ------------------------------------------------------------ runs and launches
    def _run_dir(self, run_id: str) -> Path:
        runs_dir = self.server.runs_dir
        candidate = runs_dir / run_id
        is_child = (
            run_id not in ("", ".", "..")
            and "/" not in run_id
            and "\\" not in run_id
            and candidate.parent.resolve() == runs_dir.resolve()
        )
        if not is_child or not (candidate / RUN_INFO_FILE).is_file():
            raise _NotFound(f"no run '{run_id}'")
        return candidate

    def _launch_id(self, launch_id: str) -> str:
        """``launch_id`` when it names a launch record of this runs folder; 404 otherwise."""
        try:
            folder = launch_folder(self.server.runs_dir, launch_id)
        except LaunchError as error:
            raise _NotFound(str(error)) from error
        if not (folder / LAUNCH_FILE).is_file():
            raise _NotFound(f"no launch '{launch_id}'")
        return launch_id

    def _load(self, run_id: str) -> RunRecord:
        return load_run(self._run_dir(run_id))

    def _records(self) -> list[RunRecord]:
        return [record for record, _ in self._scan() if record is not None]

    def _scan(self) -> list[tuple[RunRecord | None, dict[str, str] | None]]:
        if not self.server.runs_dir.is_dir():
            return []
        out = []
        for run_dir in find_runs([self.server.runs_dir]):
            try:
                out.append((load_run(run_dir), None))
            except _UNREADABLE as error:
                out.append((None, {"id": run_dir.name, "error": str(error)}))
        return out

    def _run_list(self) -> dict[str, Any]:
        scanned = self._scan()
        presets = self._description_presets()
        runs_dir = self.server.runs_dir
        return {
            "runs": [
                run_summary(record, presets=presets) for record, _ in scanned if record is not None
            ],
            "skipped": [problem for _, problem in scanned if problem is not None],
            "runs_dir": runs_dir.as_posix(),
            "runs_dir_resolved": runs_dir.resolve().as_posix(),
        }

    def _presets(self) -> list[Preset]:
        try:
            return list_presets(self.server.configs_dir)
        except ConfigError as error:
            raise _BadRequest(str(error)) from error

    def _description_presets(self) -> list[Preset]:
        """The presets for the old-run description fallback; a broken preset only loses it."""
        try:
            return list_presets(self.server.configs_dir)
        except ConfigError:
            return []

    # ------------------------------------------------------------ replay
    def _create_replay(self, request: dict[str, Any]) -> ReplaySession:
        speed = request.get("speed", DEFAULT_SPEED)
        if type(speed) is not int or speed not in REPLAY_SPEEDS:
            choices = ", ".join(str(choice) for choice in REPLAY_SPEEDS)
            raise _BadRequest(
                f"speed must be one of {choices} steps per second (0 = as fast as possible), "
                f"got {speed!r}"
            )
        try:
            specs = parse_board_specs(request.get("boards"))
            boards = [(spec, self._run_dir(spec.run)) for spec in specs]
            return self.server.replays.create(boards, speed=speed)
        except ReplayError as error:
            raise _BadRequest(str(error)) from error

    def _replay(self, session_id: str) -> ReplaySession:
        session = self.server.replays.get(session_id)
        if session is None:
            raise _NotFound(f"no replay session '{session_id}'; start a new one")
        return session

    def _stream_replay(self, session: ReplaySession) -> None:
        """Server-Sent Events: one ``data:`` event per frame until the session or the page ends."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True  # no Content-Length: the stream ends when the socket does
        session.attach()
        try:
            seq = -1  # the first frame sent is the newest one, whatever its number
            while True:
                frame = session.next_frame(seq, timeout=STREAM_KEEPALIVE_SECONDS)
                if frame is None:
                    self._write_event(": keepalive\n\n")
                elif frame.get("closed"):
                    self._write_event("event: end\ndata: {}\n\n")
                    return
                else:
                    seq = frame["seq"]
                    data = json.dumps(json_safe(frame), allow_nan=False)
                    self._write_event(f"data: {data}\n\n")
        except _CLIENT_GONE:
            pass  # the page closed the stream; nothing is left to tell it
        finally:
            session.detach()

    def _write_event(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    @staticmethod
    def _offset(query: dict[str, str]) -> int:
        try:
            offset = int(query.get("since", "0"))
        except ValueError as error:
            raise _BadRequest("since must be a byte offset (an integer >= 0)") from error
        if offset < 0:
            raise _BadRequest("since must be a byte offset (an integer >= 0)")
        return offset

    # ------------------------------------------------------------ responses
    def _send_static(self, name: str) -> None:
        if not _STATIC_NAME.fullmatch(name):
            raise _NotFound(f"no file '{name}'")
        path = self.server.static / name
        content_type = CONTENT_TYPES.get(Path(name).suffix)
        if content_type is None or not path.is_file():
            raise _NotFound(f"no file '{name}'")
        self._send_file(path, content_type)

    def _send_file(self, path: Traversable, content_type: str) -> None:
        self._send(path.read_bytes(), content_type, cache="no-cache")

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
        self._send(body, "application/json", status=status, cache="no-store")

    def _send(
        self, body: bytes, content_type: str, *, status: HTTPStatus = HTTPStatus.OK, cache: str
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)
