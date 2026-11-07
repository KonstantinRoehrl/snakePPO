"""The dashboard's HTTP server: static frontend, bundled plotly.js and a read-only JSON API.

Routes (GET only; every other method gets 405)::

    /                               static/index.html
    /static/<file>                  a file directly inside static/
    /vendor/plotly.min.js           the plotly.js bundled in the ``plotly`` Python package
    /api/runs                       {"runs": [summary], "skipped": [{"id", "error"}]}
    /api/runs/<id>                  {"run": summary, "config": {...}, "info": {...}}
    /api/runs/<id>/metrics?since=N  {"lines": [...], "offset": int, "reset": bool}
    /api/results[?baseline=G]       {"rows": [...], "left_out": int, "baseline": str | null}
    /api/diff?base=<id>&runs=<id>,<id>
                                    {"base": id, "differences": {id: {key: [base, run]}}}

Run ids are folder names of direct children of the runs folder that hold a ``run.json``.
"""

from __future__ import annotations

import errno
import importlib.resources
import json
import re
import sys
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import yaml

from snakeppo.dashboard.api import (
    diff_payload,
    json_safe,
    metrics_payload,
    results_payload,
    run_detail,
    run_summary,
)
from snakeppo.runs import RUN_INFO_FILE, RunRecord, find_runs, load_run, read_metrics_since

DEFAULT_PORT = 8765
HOST = "127.0.0.1"
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


def static_dir() -> Traversable:
    return importlib.resources.files("snakeppo.dashboard") / "static"


def plotly_js() -> Traversable:
    return importlib.resources.files("plotly") / "package_data" / "plotly.min.js"


class DashboardServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already in use.
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True

    def __init__(self, runs_dir: Path, port: int, static: Traversable) -> None:
        self.runs_dir = Path(runs_dir)
        self.static = static
        super().__init__((HOST, port), _Handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


def make_server(
    runs_dir: Path, port: int = DEFAULT_PORT, *, static: Traversable | None = None
) -> DashboardServer:
    """Bind the server (``port=0`` picks a free one); raises ``OSError`` if the port is busy."""
    return DashboardServer(runs_dir, port, static if static is not None else static_dir())


def serve(runs_dir: Path, port: int = DEFAULT_PORT, *, open_browser: bool = False) -> int:
    """Serve until Ctrl+C; a busy port is reported as a ``ValueError`` naming ``--port``."""
    try:
        server = make_server(runs_dir, port)
    except OSError as error:
        if error.errno in _ADDRESS_IN_USE:
            raise ValueError(
                f"port {port} is already in use; pick another one with --port"
            ) from error
        raise
    if not Path(runs_dir).is_dir():
        print(f"Note: {runs_dir} does not exist yet; runs appear once training writes them.")
    print(f"Dashboard for {runs_dir} at {server.url} (Ctrl+C stops it)")
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped the dashboard.")
    finally:
        server.server_close()
    return 0


class _NotFound(Exception):
    pass


class _BadRequest(Exception):
    pass


class _Handler(BaseHTTPRequestHandler):
    server: DashboardServer

    # ------------------------------------------------------------ dispatch
    def do_GET(self) -> None:
        url = urlsplit(self.path)
        query = {key: values[-1] for key, values in parse_qs(url.query).items()}
        parts = [unquote(part) for part in url.path.split("/") if part]
        try:
            self._route(parts, query)
        except _NotFound as error:
            self._send_json({"error": str(error) or "not found"}, HTTPStatus.NOT_FOUND)
        except _BadRequest as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # report it to the page instead of dropping the connection
            traceback.print_exc()
            self._send_json({"error": repr(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _not_allowed

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # the page polls every two seconds; per-request lines would flood the terminal

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
                self._send_json(run_detail(self._load(run_id)))
            case ["api", "runs", run_id, "metrics"]:
                chunk = read_metrics_since(self._run_dir(run_id), self._offset(query))
                self._send_json(metrics_payload(chunk))
            case ["api", "results"]:
                try:
                    payload = results_payload(self._records(), query.get("baseline") or None)
                except ValueError as error:
                    raise _BadRequest(str(error)) from error
                self._send_json(payload)
            case ["api", "diff"]:
                base = self._load(query.get("base", ""))
                others = [self._load(rid) for rid in query.get("runs", "").split(",") if rid]
                self._send_json(diff_payload(base, others))
            case _:
                raise _NotFound(f"no route {'/' + '/'.join(parts)}")

    # ------------------------------------------------------------ runs
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
        return {
            "runs": [run_summary(record) for record, _ in scanned if record is not None],
            "skipped": [problem for _, problem in scanned if problem is not None],
        }

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
