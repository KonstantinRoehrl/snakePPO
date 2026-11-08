from pathlib import Path

import pytest
import torch

from snakeppo.agents import PolicyPlayer, available_agents
from snakeppo.cli import main
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
    assert run_dir.name in capsys.readouterr().out


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


@pytest.mark.parametrize("board_size", [None, 12])
def test_eval_render_draws_a_game_until_its_episode_ends(tmp_path, monkeypatch, capsys, board_size):
    from snakeppo import viewer

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    drawn, ended = set(), []
    real_draw, real_end_episodes = viewer.Viewer.draw, PolicyPlayer.end_episodes

    def draw(self, grids, scores, body_order):
        drawn.add((self.board_size, tuple(grids.shape[-2:]), tuple(body_order.shape[-2:])))
        return real_draw(self, grids, scores, body_order)

    def end_episodes(self, done):
        ended.append(int(done.sum()))
        real_end_episodes(self, done)

    run_dir = _smoke_run(tmp_path)
    capsys.readouterr()
    # patched after training, whose evals may play through PolicyPlayer too
    monkeypatch.setattr(viewer.Viewer, "draw", draw)
    monkeypatch.setattr(PolicyPlayer, "end_episodes", end_episodes)
    args = ["eval", str(run_dir), "--render", "--episodes", "1", "--envs", "1", "--sleep", "0"]
    if board_size is not None:
        args += ["--board-size", str(board_size)]
    assert main([*args, "--device", "cpu"]) == 0
    assert "Episode 1 finished with score" in capsys.readouterr().out
    side = board_size or 8
    assert drawn == {(side, (side, side), (side, side))}
    assert sum(ended) == 1


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
