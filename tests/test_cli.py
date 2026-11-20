import shutil
import threading
import webbrowser
from pathlib import Path

import pytest
import torch

from snakeppo.agents import available_agents
from snakeppo.cli import main, speed_for_sleep
from snakeppo.config import load_config
from snakeppo.dashboard import server
from snakeppo.results import truncate
from snakeppo.runs import load_checkpoint

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def _smoke_run(runs_dir, *overrides):
    """Train the smoke config through the CLI, with ``--set`` overrides; returns the run folder."""
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={runs_dir.as_posix()}"]
    for override in overrides:
        args += ["--set", override]
    assert main(args) == 0
    (run_dir,) = runs_dir.iterdir()
    return run_dir


def _edit_saved_config(checkpoint_path, section, key, value):
    """Change one key of the config a checkpoint carries, as a hand edit would."""
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint["config"][section][key] = value
    torch.save(checkpoint, checkpoint_path)


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "train" in capsys.readouterr().out


def test_agents_lists_every_architecture_and_whether_it_plays_any_board_size(capsys):
    assert main(["agents"]) == 0
    _, header, *rows = capsys.readouterr().out.splitlines()
    assert header.split() == ["name", "parameters", "any", "size", "description"]
    any_size = {row.split()[0]: row.split()[2] for row in rows}
    assert sorted(any_size) == available_agents()
    assert {
        name: any_size[name] for name in ("cnn", "recurrent", "mlp", "cnn-spatial", "resnet")
    } == {
        "cnn": "yes",
        "recurrent": "yes",
        "mlp": "no",
        "cnn-spatial": "no",
        "resnet": "no",
    }


def test_rewards_lists_every_strategy(capsys):
    assert main(["rewards"]) == 0
    out = capsys.readouterr().out
    assert "reward.strategy=NAME" in out
    for name in ("classic", "sparse", "distance", "potential", "hunger", "free_space"):
        assert f"  {name} " in out


def test_train_eval_compare_round_trip(tmp_path, capsys):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main([*args, "--name", "cli", "--seed", "3"]) == 0
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name.endswith("_cli_seed3")
    assert str(run_dir) in capsys.readouterr().out

    assert main(["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "over 2 episodes (greedy)" in out and "ended by wall" in out
    last = run_dir / "checkpoints" / "last.pt"
    assert main(["eval", str(last), "--stochastic", "--episodes", "2", "--device", "cpu"]) == 0
    assert "(stochastic)" in capsys.readouterr().out

    assert main(["compare", str(tmp_path)]) == 0
    header, line = capsys.readouterr().out.splitlines()
    assert header.split()[-1] == "description" and run_dir.name in line
    # --name does not change the description: it comes from the preset file
    assert line.rstrip().endswith(truncate(load_config(SMOKE).train.description))


def test_config_errors_are_reported_without_a_traceback(tmp_path, capsys):
    assert main(["train", "--set", "ppo.nope=1"]) == 2
    assert "snakeppo train: error: unknown key 'ppo.nope'" in capsys.readouterr().err
    assert main(["eval", str(tmp_path)]) == 2
    assert "no best checkpoint" in capsys.readouterr().err


def test_compare_without_runs_says_so(tmp_path, capsys):
    assert main(["compare", str(tmp_path)]) == 1
    assert "No run folders found" in capsys.readouterr().out


def _train_seeds(runs_dir, group, seeds, *extra):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={runs_dir.as_posix()}"]
    for seed in seeds:
        assert main([*args, "--group", group, "--seed", str(seed), *extra]) == 0


def test_results_judges_groups_in_every_format(tmp_path, capsys):
    _train_seeds(tmp_path, "base", (1, 2, 3))
    _train_seeds(tmp_path, "kl", (1, 2, 3), "--set", "ppo.target_kl=0.0001")
    capsys.readouterr()
    # smoke.yaml differs from the code defaults, so no group qualifies as the default reference
    assert main(["results", str(tmp_path)]) == 0
    assert "withheld: no baseline group" in capsys.readouterr().out
    assert main(["results", str(tmp_path), "--baseline", "base"]) == 0
    text = capsys.readouterr().out
    assert "vs reference" in text
    kl_line = next(line for line in text.splitlines() if line.startswith("kl "))
    assert "ppo.target_kl=0.0001" in kl_line and " base " in kl_line
    base_line = next(line for line in text.splitlines() if line.startswith("base "))
    assert base_line.rstrip().endswith("reference")
    assert truncate(load_config(SMOKE).train.description) in base_line
    assert main(["results", str(tmp_path), "--format", "csv"]) == 0
    assert capsys.readouterr().out.startswith("group,train.total_env_steps")
    assert main(["results", str(tmp_path), "--baseline", "kl", "--format", "markdown"]) == 0
    assert capsys.readouterr().out.startswith("| group |")
    assert main(["results", str(tmp_path), "--baseline", "nope"]) == 2
    assert "unknown baseline group 'nope'" in capsys.readouterr().err


def test_results_without_groups_says_how_to_make_one(tmp_path, capsys):
    assert main(["results", str(tmp_path)]) == 1
    assert "No run folders found" in capsys.readouterr().out
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main(args) == 0
    capsys.readouterr()
    assert main(["results", str(tmp_path)]) == 1
    assert "No seed groups found (1 ungrouped run(s))" in capsys.readouterr().out


def test_serve_binds_localhost_opens_the_browser_and_stops_on_ctrl_c(tmp_path, monkeypatch, capsys):
    from snakeppo.dashboard import server

    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)

    def interrupted(self, poll_interval=0.5):
        raise KeyboardInterrupt

    monkeypatch.setattr(server.DashboardServer, "serve_forever", interrupted)
    assert main(["serve", "--runs-dir", str(tmp_path), "--port", "0", "--open"]) == 0
    out = capsys.readouterr().out
    (url,) = opened
    assert url.startswith("http://127.0.0.1:") and url in out
    assert "Stopped the dashboard." in out


def test_serve_passes_the_configs_folder_and_says_it_is_the_main_ui(monkeypatch, capsys):
    from snakeppo.dashboard import server

    calls = []
    monkeypatch.setattr(server, "serve", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    assert main(["serve", "--runs-dir", "r", "--port", "0", "--configs-dir", "presets"]) == 0
    assert calls == [((Path("r"), 0), {"open_browser": False, "configs_dir": Path("presets")})]
    assert main(["serve"]) == 0
    assert calls[-1][1]["configs_dir"] is None
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "open the local dashboard (main UI)" in capsys.readouterr().out


def test_serve_reports_a_busy_port(tmp_path, capsys):
    from snakeppo.dashboard.server import make_server

    busy = make_server(tmp_path, 0)
    try:
        port = busy.server_address[1]
        assert main(["serve", "--runs-dir", str(tmp_path), "--port", str(port)]) == 2
        assert f"port {port} is already in use; pick another one with --port" in (
            capsys.readouterr().err
        )
    finally:
        busy.server_close()


@pytest.mark.parametrize("name", ["2", "off"])
def test_name_flag_is_taken_as_text_not_parsed_as_yaml(tmp_path, name):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main([*args, "--name", name, "--seed", "5"]) == 0
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name.endswith(f"_{name}_seed5")


def test_agent_flag_is_taken_as_text_not_parsed_as_yaml(capsys):
    assert main(["train", "--agent", "off"]) == 2
    assert "unknown agent 'off'" in capsys.readouterr().err


def test_set_still_parses_values_as_yaml(capsys):
    assert main(["train", "--set", "train.name=2"]) == 2
    assert "train.name must be str, got 2" in capsys.readouterr().err


def test_bad_agent_kwargs_are_reported_without_a_traceback_or_run_folder(tmp_path, capsys):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main([*args, "--set", "agent.kwargs={foo: 1}"]) == 2
    err = capsys.readouterr().err
    assert "snakeppo train: error: agent 'cnn'" in err
    assert "'foo'" in err
    assert "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


def test_eval_plays_a_body_order_run_and_refuses_it_under_another_observation(tmp_path, capsys):
    run_dir = _smoke_run(tmp_path, "observation.body=order")
    capsys.readouterr()
    assert main(["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]) == 0
    assert "over 2 episodes (greedy)" in capsys.readouterr().out

    best = run_dir / "checkpoints" / "best.pt"
    _edit_saved_config(best, "observation", "body", "binary")
    assert main(["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]) == 2
    err = capsys.readouterr().err
    assert (
        f"snakeppo eval: error: {best} was trained on channels body_order, head, food, wall on a "
        "8x8 full view, but its config describes channels body, head, food, wall on a 8x8 full "
        "view; refusing to load it with the wrong observation"
    ) in err
    assert "Traceback" not in err


def test_eval_refuses_a_checkpoint_that_does_not_fit_its_configured_network(tmp_path, capsys):
    run_dir = _smoke_run(tmp_path)
    best = run_dir / "checkpoints" / "best.pt"
    _edit_saved_config(best, "agent", "name", "cnn-small")
    capsys.readouterr()
    assert main(["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]) == 2
    err = capsys.readouterr().err
    assert (
        f"snakeppo eval: error: {best} does not fit a 'cnn-small' network for channels body, "
        "head, food, wall on a 8x8 full view: Error(s) in loading state_dict"
    ) in err
    assert "Traceback" not in err


def test_a_local_view_with_an_even_board_network_fails_before_any_run_folder(tmp_path, capsys):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    view = ["--set", "observation.view=local", "--set", "observation.view_size=5"]
    assert main([*args, *view, "--agent", "cnn-spatial"]) == 2
    err = capsys.readouterr().err
    assert (
        "snakeppo train: error: agent 'cnn-spatial' needs an even input size, but "
        "observation.view: local feeds it observation.view_size (5) cells a side"
    ) in err
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope="module")
def render_run(tmp_path_factory):
    """One smoke run shared by the ``eval --render`` tests, which only read it."""
    return _smoke_run(tmp_path_factory.mktemp("runs"))


@pytest.fixture
def opened(monkeypatch):
    """URLs ``webbrowser.open`` was asked to show; no real browser starts."""
    urls = []
    monkeypatch.setattr(webbrowser, "open", urls.append)
    return urls


@pytest.fixture
def dashboard_on(tmp_path):
    """Start real dashboards in the background; yields ``start(runs_dir) -> port``."""
    started = []

    def start(runs_dir):
        dashboard = server.make_server(runs_dir, 0)
        threading.Thread(target=dashboard.serve_forever, daemon=True).start()
        started.append(dashboard)
        return dashboard.server_address[1]

    yield start
    for dashboard in started:
        dashboard.shutdown()
        dashboard.server_close()


def _boards(run_dir, checkpoint, mode, size, count):
    return ",".join([f"{run_dir.name}%7C{checkpoint}%7C{mode}%7C{size}"] * count)


@pytest.mark.parametrize(
    ("sleep", "speed"),
    [(0.0, 0), (-1.0, 0), (0.1, 10), (0.05, 25), (0.2, 5), (0.5, 2), (2.0, 2), (0.016, 60)],
)
def test_speed_for_sleep_picks_the_nearest_replay_speed(sleep, speed):
    assert speed_for_sleep(sleep) == speed


def test_eval_render_opens_the_replay_on_the_dashboard_that_serves_its_runs_folder(
    render_run, opened, dashboard_on, monkeypatch, capsys
):
    def second_dashboard(*args, **kwargs):
        raise AssertionError("eval --render started a dashboard although one serves the run")

    port = dashboard_on(render_run.parent)
    monkeypatch.setattr(server, "serve", second_dashboard)
    args = ["eval", str(render_run), "--render", "--envs", "2", "--stochastic"]
    args += ["--board-size", "12", "--sleep", "0.2", "--port", str(port)]
    assert main(args) == 0
    boards = _boards(render_run, "best", "stochastic", "12", 2)
    assert opened == [f"http://127.0.0.1:{port}/#tab=replay&boards={boards}&speed=5"]
    assert f"http://127.0.0.1:{port}/" in capsys.readouterr().out


def test_eval_render_serves_the_runs_folder_when_no_dashboard_answers(
    render_run, opened, monkeypatch
):
    served = []

    def interrupted(self, poll_interval=0.5):
        served.append(Path(self.runs_dir).resolve())
        raise KeyboardInterrupt

    monkeypatch.setattr(server.DashboardServer, "serve_forever", interrupted)
    last = render_run / "checkpoints" / "last.pt"
    # nothing can answer on port 0, so a dashboard starts there (on a free port)
    assert main(["eval", str(last), "--render", "--port", "0"]) == 0
    assert served == [render_run.parent.resolve()]
    (url,) = opened
    boards = _boards(render_run, "last", "greedy", "", 4)
    assert url.startswith("http://127.0.0.1:")
    assert url.endswith(f"/#tab=replay&boards={boards}&speed=10")


def test_eval_render_starts_its_own_dashboard_when_the_port_serves_another_folder(
    render_run, opened, dashboard_on, tmp_path, monkeypatch
):
    calls = []

    def fake_serve(runs_dir, port=server.DEFAULT_PORT, *, open_browser=False, **options):
        calls.append((Path(runs_dir).resolve(), port, open_browser, options.get("open_path")))
        return 0

    other = tmp_path / "other-runs"
    other.mkdir()
    port = dashboard_on(other)
    monkeypatch.setattr(server, "serve", fake_serve)
    args = ["eval", str(render_run), "--render", "--envs", "1", "--sleep", "0", "--port", str(port)]
    assert main(args) == 0
    boards = _boards(render_run, "best", "greedy", "", 1)
    assert calls == [(render_run.parent.resolve(), 0, True, f"#tab=replay&boards={boards}&speed=0")]
    assert opened == []


@pytest.fixture
def no_dashboard(monkeypatch, opened):
    def touched(*args, **kwargs):
        raise AssertionError("a refused eval --render reached the dashboard")

    monkeypatch.setattr(server, "served_runs_dir", touched)
    monkeypatch.setattr(server, "serve", touched)
    yield
    assert opened == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            ["--episodes", "1"],
            "--episodes cannot be combined with --render: the browser replay plays until you "
            "close it; drop --render to score a fixed number of episodes",
        ),
        (["--envs", "5"], "--envs must be 1 to 4 with --render, got 5"),
        (["--envs", "0"], "--envs must be 1 to 4 with --render, got 0"),
        (["--board-size", "4"], "--board-size must be at least 5, got 4"),
    ],
    ids=["episodes", "too-many-boards", "no-board", "board-too-small"],
)
def test_eval_render_refuses_options_the_replay_cannot_honour(
    render_run, no_dashboard, capsys, options, message
):
    assert main(["eval", str(render_run), "--render", *options]) == 2
    err = capsys.readouterr().err
    assert f"snakeppo eval: error: {message}" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "layout",
    [("model.pt",), ("loose", "checkpoints", "best.pt"), ("copied", "checkpoints", "old.pt")],
    ids=["outside-a-run", "no-run-json", "not-best-or-last"],
)
def test_eval_render_refuses_a_checkpoint_that_is_not_a_runs_best_or_last(
    render_run, no_dashboard, tmp_path, capsys, layout
):
    target = tmp_path.joinpath(*layout)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(render_run / "checkpoints" / "best.pt", target)
    if layout[0] == "copied":
        shutil.copy(render_run / "run.json", tmp_path / "copied" / "run.json")
    assert main(["eval", str(target), "--render"]) == 2
    err = capsys.readouterr().err
    assert (
        f"snakeppo eval: error: --render plays the best.pt or last.pt checkpoint of a run "
        f"folder, and {target} is not one; score it with headless 'snakeppo eval {target}'"
    ) in err
    assert "Traceback" not in err


def test_eval_plays_a_size_agnostic_run_on_another_board(tmp_path, capsys):
    run_dir = _smoke_run(tmp_path)
    capsys.readouterr()
    evaluate = ["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]
    assert main(evaluate) == 0
    assert "on a 8x8 board over 2 episodes (greedy)" in capsys.readouterr().out
    assert main([*evaluate, "--board-size", "12"]) == 0
    assert "on a 12x12 board over 2 episodes (greedy)" in capsys.readouterr().out
    assert main([*evaluate, "--board-size", "4"]) == 2
    err = capsys.readouterr().err
    assert "snakeppo eval: error: --board-size must be at least 5, got 4" in err
    assert "Traceback" not in err


def test_eval_on_another_board_does_not_revalidate_the_run_config(tmp_path, capsys):
    run_dir = _smoke_run(tmp_path)
    best = run_dir / "checkpoints" / "best.pt"
    # valid on the 8x8 training board, but start_board 7 would not fit a 6x6 board
    curriculum = {
        "grow": "board",
        "stages": 2,
        "start_board": 7,
        "min_updates": 1,
        "max_updates": 1,
    }
    for key, value in curriculum.items():
        _edit_saved_config(best, "curriculum", key, value)
    capsys.readouterr()
    args = ["eval", str(run_dir), "--episodes", "2", "--device", "cpu", "--board-size", "6"]
    assert main(args) == 0
    assert "on a 6x6 board over 2 episodes" in capsys.readouterr().out


def test_eval_refuses_another_board_for_a_board_bound_run(tmp_path, capsys):
    run_dir = _smoke_run(tmp_path, "agent.name=cnn-spatial")
    capsys.readouterr()
    evaluate = ["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]
    assert main([*evaluate, "--board-size", "8"]) == 0
    assert "on a 8x8 board over 2 episodes" in capsys.readouterr().out
    assert main([*evaluate, "--board-size", "12"]) == 2
    err = capsys.readouterr().err
    assert (
        "snakeppo eval: error: agent 'cnn-spatial' is board-bound: it trained on a 8x8 board "
        "and cannot play 12x12; use a size-agnostic architecture (see 'snakeppo agents') or "
        "train with observation.view: local"
    ) in err
    assert "Traceback" not in err


def test_eval_plays_a_local_view_run_of_a_board_bound_network_on_any_board(tmp_path, capsys):
    view = ("observation.view=local", "observation.view_size=5")
    run_dir = _smoke_run(tmp_path, "agent.name=mlp", *view)
    capsys.readouterr()
    args = ["eval", str(run_dir), "--episodes", "2", "--device", "cpu", "--board-size", "12"]
    assert main(args) == 0
    assert "on a 12x12 board over 2 episodes" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ("agent.name=cnn-spatial", "observation.view=local", "observation.view_size=5"),
            "agent 'cnn-spatial' needs an even input size, but observation.view: local feeds it "
            "observation.view_size (5) cells a side",
        ),
        (("agent.kwargs={foo: 1}",), "agent 'cnn' does not accept agent.kwargs 'foo'"),
        (
            ("agent.name=recurrent", "env.num_envs=5", "ppo.minibatch_size=32"),
            "agent 'recurrent' trains on whole envs: ppo.rollout_steps * env.num_envs // "
            "ppo.minibatch_size gives 2 minibatches, which must divide env.num_envs (5)",
        ),
    ],
    ids=["local-view-even-board-network", "unknown-agent-kwargs", "recurrent-batching"],
)
def test_sweep_builds_the_network_once_before_any_seed_starts(
    tmp_path, monkeypatch, capsys, overrides, message
):
    from snakeppo import sweep

    def started(*args, **kwargs):
        raise AssertionError("sweep started its seeds with a config whose network cannot build")

    monkeypatch.setattr(sweep, "run_sweep", started)
    args = ["sweep", "--seeds", "1,2", "--config", str(SMOKE)]
    for override in (f"train.runs_dir={tmp_path.as_posix()}", *overrides):
        args += ["--set", override]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert f"snakeppo sweep: error: {message}" in err
    assert "Traceback" not in err
    assert list(tmp_path.iterdir()) == []


def _saved_description(monkeypatch, *args):
    """The train.description ``snakeppo train --config smoke.yaml ARGS`` would save, untrained."""
    from snakeppo.ppo import trainer

    configs = []
    monkeypatch.setattr(trainer, "train", lambda config, **kwargs: configs.append(config))
    assert main(["train", "--config", str(SMOKE), *args]) == 0
    (config,) = configs
    return config.train.description


@pytest.mark.parametrize(
    ("args", "suffix"),
    [
        ((), ""),
        (
            ("--seed", "3", "--name", "x", "--group", "g", "--set", "train.runs_dir=elsewhere"),
            "",
        ),
        (
            ("--set", "ppo.lr=0.0005", "--set", " train.total_env_steps=1024 ", "--agent", "mlp"),
            " Changed: ppo.lr=0.0005, train.total_env_steps=1024, agent.name=mlp",
        ),
    ],
    ids=["preset", "bookkeeping-only", "changes"],
)
def test_train_saves_the_preset_description_with_its_changes(monkeypatch, args, suffix):
    preset = load_config(SMOKE).train.description
    assert _saved_description(monkeypatch, *args) == preset + suffix


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("--set", "ppo.lr=0.0005", "--set", "train.description=Plain text."), "Plain text."),
        (("--set", "ppo.lr=0.0005", "--description", "My # note: yes"), "My # note: yes"),
        (("--description=-starts with a dash",), "-starts with a dash"),
        (("--description", ""), ""),
    ],
    ids=["set", "flag-verbatim", "flag-dash", "flag-empty"],
)
def test_an_explicit_description_wins_without_a_suffix(monkeypatch, args, expected):
    assert _saved_description(monkeypatch, *args) == expected


def test_resume_refuses_a_description(tmp_path, capsys):
    assert main(["train", "--resume", str(tmp_path), "--description", "x"]) == 2
    assert "cannot be combined with --description" in capsys.readouterr().err


def test_sweep_gives_every_seed_its_description(tmp_path, monkeypatch):
    from snakeppo import sweep

    seen = []

    def run_sweep(seeds, train_args, **kwargs):
        seen.append(list(train_args))
        return []

    monkeypatch.setattr(sweep, "run_sweep", run_sweep)
    args = ["sweep", "--seeds", "1,2", "--config", str(SMOKE)]
    args += ["--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main([*args, "--description", "-a dash # and: colon"]) == 0
    assert seen[-1][-1] == "--description=-a dash # and: colon"
    assert main(args) == 0
    assert not any(arg.startswith("--description") for arg in seen[-1])
