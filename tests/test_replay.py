import threading
import time
from pathlib import Path

import pytest
import torch

from snakeppo.agents import PolicyPlayer
from snakeppo.cli import main
from snakeppo.dashboard import replay
from snakeppo.dashboard.replay import (
    CELL_BODY,
    CELL_EMPTY,
    CELL_FOOD,
    CELL_HEAD,
    CELL_WALL,
    DEFAULT_SPEED,
    REPLAY_TORCH_THREADS,
    BoardSpec,
    ReplayError,
    ReplayManager,
    parse_board_specs,
)
from snakeppo.env import END_REASONS

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
WAIT = 10.0  # seconds a test waits for the stepping thread before it fails


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One finished smoke run (cnn on 8x8) with best.pt and last.pt, shared by the module."""
    runs_dir = tmp_path_factory.mktemp("runs")
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={runs_dir.as_posix()}"]
    assert main(args) == 0
    (run_dir,) = runs_dir.iterdir()
    return run_dir


@pytest.fixture(autouse=True)
def _keep_torch_threads():
    """Replay sets the process-wide torch thread count; give the rest of the suite its own back."""
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def manager():
    replays = ReplayManager()
    yield replays
    replays.close()


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _board(run_dir, **fields):
    return BoardSpec(run=run_dir.name, **fields), run_dir


def _frame_after(session, after, check=lambda frame: True):
    """The first published frame newer than ``after`` that passes ``check``."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        frame = session.next_frame(after, timeout=WAIT)
        assert frame is not None, "no frame arrived"
        assert "closed" not in frame, "the session closed"
        if check(frame):
            return frame
        after = frame["seq"]
    raise AssertionError("no frame passed the check in time")


def _settle(session):
    """The last frame once a paused session stops publishing (one in-flight step may land)."""
    frame = session.next_frame(0, timeout=WAIT)
    while (newer := session.next_frame(frame["seq"], timeout=0.5)) is not None:
        frame = newer
    return frame


def _wait_until(condition):
    deadline = time.monotonic() + WAIT
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.02)


def _replay_threads():
    return [t for t in threading.enumerate() if t.name.startswith("snakeppo-replay")]


# ------------------------------------------------------------------ board specs
def test_parse_board_specs_fills_the_defaults():
    assert parse_board_specs([{"run": "a"}]) == [BoardSpec("a", "best", True, None)]
    raw = [{"run": "a", "checkpoint": "last", "greedy": False, "board_size": 12}, {"run": "b"}]
    assert parse_board_specs(raw) == [BoardSpec("a", "last", False, 12), BoardSpec("b")]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"run": "a"}, "boards must be a list of 1 to 4 board objects, got dict"),
        ([], "boards must be a list of 1 to 4 board objects, got 0"),
        ([{"run": "a"}] * 5, "boards must be a list of 1 to 4 board objects, got 5"),
        (["a"], "boards[0] must be an object, got str"),
        ([{"run": "a"}, {}], "boards[1].run must be a run id, got None"),
        ([{"run": ""}], "boards[0].run must be a run id, got ''"),
        ([{"run": 3}], "boards[0].run must be a run id, got 3"),
        ([{"run": "a", "checkpoint": "final"}], "boards[0].checkpoint must be one of best, last"),
        ([{"run": "a", "greedy": "yes"}], "boards[0].greedy must be true or false, got 'yes'"),
        ([{"run": "a", "board_size": "12"}], "boards[0].board_size must be a whole number"),
        ([{"run": "a", "board_size": 12.0}], "boards[0].board_size must be a whole number"),
        ([{"run": "a", "board_size": True}], "boards[0].board_size must be a whole number"),
        ([{"run": "a", "colour": "red"}], "boards[0] has unknown field 'colour'"),
    ],
)
def test_parse_board_specs_names_the_bad_field(raw, message):
    with pytest.raises(ReplayError) as error:
        parse_board_specs(raw)
    assert message in str(error.value)


# ------------------------------------------------------------------ sessions
def test_a_session_describes_its_boards_and_streams_valid_frames(manager, smoke_run):
    run = smoke_run.name
    session = manager.create([_board(smoke_run), _board(smoke_run, board_size=12, greedy=False)])
    assert torch.get_num_threads() == REPLAY_TORCH_THREADS
    assert len(session.id) == 32
    assert manager.get(session.id) is session
    assert session.describe() == {
        "id": session.id,
        "paused": False,
        "speed": DEFAULT_SPEED,
        "boards": [
            {"run": run, "checkpoint": "best", "greedy": True, "board_size": 8, "agent": "cnn"},
            {"run": run, "checkpoint": "best", "greedy": False, "board_size": 12, "agent": "cnn"},
        ],
    }
    first = session.next_frame(0, timeout=WAIT)
    assert first["seq"] >= 1 and first["paused"] is False and first["speed"] == DEFAULT_SPEED
    assert [board["size"] for board in first["boards"]] == [8, 12]
    for board in first["boards"]:
        size, cells = board["size"], board["cells"]
        assert len(cells) == size * size
        assert set(cells) <= {str(code) for code in range(5)}
        rows = [cells[row * size : (row + 1) * size] for row in range(size)]
        ring = rows[0] + rows[-1] + "".join(row[0] + row[-1] for row in rows)
        assert set(ring) == {str(CELL_WALL)}
        assert cells.count(str(CELL_HEAD)) == 1
        assert cells.count(str(CELL_FOOD)) == 1
        body = board["body"]
        assert len(body) == board["length"] >= 1
        assert cells[body[0]] == str(CELL_HEAD)
        assert all(cells[index] == str(CELL_BODY) for index in body[1:])
        assert cells.count(str(CELL_BODY)) == len(body) - 1
        assert str(CELL_EMPTY) in cells
        assert isinstance(board["score"], int) and isinstance(board["since_apple"], int)
    later = session.next_frame(first["seq"], timeout=WAIT)
    assert later["seq"] > first["seq"]


def test_finished_episodes_restart_on_their_board_and_update_the_stats(manager, smoke_run):
    session = manager.create([_board(smoke_run)], speed=0)
    frame = _frame_after(session, 0, lambda frame: frame["boards"][0]["episodes"] >= 2)
    board = frame["boards"][0]
    assert board["last_end"] in END_REASONS
    assert isinstance(board["mean_score"], float) and board["mean_score"] >= 0.0
    assert board["length"] >= 1


def test_pause_step_resume_and_speed(manager, smoke_run):
    session = manager.create([_board(smoke_run)])
    with pytest.raises(ReplayError, match="step works only while paused"):
        session.control("step")
    assert session.control("pause") == {"paused": True, "speed": DEFAULT_SPEED}
    paused = _settle(session)
    assert paused["paused"] is True
    assert session.next_frame(paused["seq"], timeout=0.3) is None
    session.control("step")
    stepped = session.next_frame(paused["seq"], timeout=WAIT)
    assert stepped["seq"] == paused["seq"] + 1
    assert session.next_frame(stepped["seq"], timeout=0.3) is None
    assert session.control("speed", 60) == {"paused": True, "speed": 60}
    assert session.next_frame(stepped["seq"], timeout=WAIT)["speed"] == 60
    assert session.control("resume") == {"paused": False, "speed": 60}
    assert session.describe()["paused"] is False
    _frame_after(session, stepped["seq"], lambda frame: not frame["paused"])


@pytest.mark.parametrize(
    ("action", "value", "message"),
    [
        ("jump", None, "unknown replay action 'jump'; expected one of pause, resume, step"),
        ("speed", 7, "speed must be one of 2, 5, 10, 25, 60, 0 (0 = as fast as possible), got 7"),
        ("speed", "fast", "speed must be one of 2, 5, 10, 25, 60, 0"),
        ("speed", True, "speed must be one of 2, 5, 10, 25, 60, 0"),
        ("speed", None, "speed must be one of 2, 5, 10, 25, 60, 0"),
    ],
)
def test_control_refuses_unknown_actions_and_speeds(manager, smoke_run, action, value, message):
    session = manager.create([_board(smoke_run)])
    with pytest.raises(ReplayError) as error:
        session.control(action, value)
    assert message in str(error.value)


def test_restart_starts_new_episodes_with_zeroed_stats_and_a_new_player(
    manager, smoke_run, monkeypatch
):
    session = manager.create([_board(smoke_run)], speed=0)
    _frame_after(session, 0, lambda frame: frame["boards"][0]["episodes"] >= 1)
    session.control("pause")
    before = _settle(session)
    players = []

    def counting_player(*args, **kwargs):
        players.append(args)
        return PolicyPlayer(*args, **kwargs)

    monkeypatch.setattr(replay, "PolicyPlayer", counting_player)
    session.control("restart")
    restarted = session.next_frame(before["seq"], timeout=WAIT)
    board = restarted["boards"][0]
    assert (board["episodes"], board["mean_score"], board["last_end"]) == (0, None, None)
    assert (board["score"], board["length"], board["since_apple"]) == (0, 1, 0)
    assert restarted["paused"] is True
    assert len(players) == 1


def test_bad_boards_are_refused_and_keep_the_current_session(manager, smoke_run, tmp_path):
    session = manager.create([_board(smoke_run)])
    with pytest.raises(ReplayError) as error:
        manager.create([_board(smoke_run), _board(smoke_run, board_size=4)])
    assert str(error.value) == (
        f"board 2 ({smoke_run.name}): --board-size must be at least 5, got 4"
    )
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(ReplayError, match=r"^board 1 \(empty-run\): no best checkpoint in "):
        manager.create([_board(empty)])
    with pytest.raises(ReplayError, match="speed must be one of"):
        manager.create([_board(smoke_run)], speed=3)
    assert not session.closed
    assert manager.get(session.id) is session


def test_creating_a_session_closes_the_previous_one(manager, smoke_run):
    first = manager.create([_board(smoke_run)])
    second = manager.create([_board(smoke_run, checkpoint="last")])
    assert first.closed and not second.closed
    assert first.next_frame(0, timeout=WAIT) == {"closed": True}
    assert manager.get(first.id) is None
    assert manager.get(second.id) is second
    assert manager.get("not-a-session") is None
    assert second.describe()["boards"][0]["checkpoint"] == "last"


def test_a_session_without_a_stream_closes_after_the_idle_time(smoke_run):
    clock = FakeClock()
    replays = ReplayManager(idle_seconds=30.0, clock=clock)
    try:
        session = replays.create([_board(smoke_run)])
        clock.now = 29.0
        time.sleep(0.5)
        assert not session.closed
        clock.now = 30.5
        _wait_until(lambda: session.closed)
        assert replays.get(session.id) is None
        assert session.next_frame(0, timeout=WAIT) == {"closed": True}
    finally:
        replays.close()


def test_an_attached_stream_keeps_a_session_open_and_the_idle_clock_starts_at_detach(smoke_run):
    clock = FakeClock()
    replays = ReplayManager(idle_seconds=30.0, clock=clock)
    try:
        session = replays.create([_board(smoke_run)])
        session.attach()
        clock.now = 100.0
        time.sleep(0.5)
        assert not session.closed
        session.detach()
        clock.now = 129.0
        time.sleep(0.5)
        assert not session.closed
        clock.now = 130.5
        _wait_until(lambda: session.closed)
    finally:
        replays.close()


def test_closing_ends_the_stepping_thread_and_wakes_waiting_streams(smoke_run):
    replays = ReplayManager()
    session = replays.create([_board(smoke_run)])
    session.control("pause")
    _settle(session)
    (thread,) = _replay_threads()
    assert thread.daemon
    woken = []
    waiter = threading.Thread(target=lambda: woken.append(session.next_frame(10**9, WAIT)))
    waiter.start()
    replays.close()
    waiter.join(WAIT)
    assert woken == [{"closed": True}]
    assert session.closed
    assert not thread.is_alive()
    assert _replay_threads() == []
