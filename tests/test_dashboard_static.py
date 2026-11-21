"""The shipped frontend: served by the real server, built from local files only."""

import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from snakeppo.dashboard.server import make_server, static_dir

STATIC = Path(str(static_dir()))
EXPECTED_FILES = {
    "index.html",
    "app.js",
    "api.js",
    "charts.js",
    "colors.js",
    "board.js",
    "launch.js",
    "replay.js",
    "style.css",
    "launch.css",
    "replay.css",
}


def _shipped():
    return sorted(STATIC.iterdir()) if STATIC.is_dir() else []


@pytest.fixture
def server(tmp_path):
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _status_and_body(server, path):
    try:
        with urllib.request.urlopen(server.url.rstrip("/") + path, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def test_the_frontend_files_ship_inside_the_package():
    assert {path.name for path in _shipped()} >= EXPECTED_FILES


def test_index_and_every_asset_it_references_are_served(server):
    status, body = _status_and_body(server, "/")
    assert status == 200
    html = body.decode("utf-8")
    references = re.findall(r'(?:src|href)="([^"#]+)"', html)
    assert "/vendor/plotly.min.js" in references
    assert "/static/app.js" in references and "/static/style.css" in references
    for reference in references:
        assert reference.startswith("/"), f"{reference} is not a local path"
        assert _status_and_body(server, reference)[0] == 200, reference


def test_modules_import_only_sibling_files():
    for script in (path for path in _shipped() if path.suffix == ".js"):
        text = script.read_text("utf-8")
        imports = re.findall(r"""(?:from|import)\s*\(?\s*["']([^"']+)["']""", text)
        for spec in imports:
            assert spec.startswith("./"), f"{script.name} imports {spec}"
            assert (STATIC / spec[2:]).is_file(), f"{script.name} imports missing {spec}"


def test_nothing_is_loaded_from_the_internet():
    for path in _shipped():
        text = path.read_text("utf-8")
        assert not re.search(r"https?://", text), f"{path.name} names an external URL"


def test_every_run_colour_slot_has_a_light_and_two_matching_dark_tokens():
    size = re.search(r"RUN_PALETTE_SIZE = (\d+);", (STATIC / "colors.js").read_text("utf-8"))
    assert size, "colors.js defines RUN_PALETTE_SIZE"
    css = (STATIC / "style.css").read_text("utf-8")
    values: dict[int, list[str]] = {}
    for slot, value in re.findall(r"--run-(\d+):\s*(#[0-9a-fA-F]{6});", css):
        values.setdefault(int(slot), []).append(value.lower())
    assert sorted(values) == list(range(1, int(size.group(1)) + 1))
    for slot, found in values.items():
        # :root (light), the prefers-color-scheme dark block, then :root[data-theme="dark"].
        assert len(found) == 3, f"--run-{slot} is defined {len(found)} times, expected 3"
        assert found[1] == found[2], f"--run-{slot} differs between the two dark blocks"
