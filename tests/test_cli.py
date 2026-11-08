from pathlib import Path

import pytest

from snakeppo.cli import main

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "train" in capsys.readouterr().out


def test_agents_lists_every_architecture(capsys):
    assert main(["agents"]) == 0
    out = capsys.readouterr().out
    for name in ("cnn", "cnn-spatial", "resnet"):
        assert name in out


def test_train_eval_compare_round_trip(tmp_path, capsys):
    args = ["train", "--config", str(SMOKE), "--set", f"train.runs_dir={tmp_path.as_posix()}"]
    assert main([*args, "--name", "cli", "--seed", "3"]) == 0
    (run_dir,) = tmp_path.iterdir()
    assert run_dir.name.endswith("_cli_seed3")
    assert str(run_dir) in capsys.readouterr().out

    assert main(["eval", str(run_dir), "--episodes", "2", "--device", "cpu"]) == 0
    assert "over 2 episodes (greedy)" in capsys.readouterr().out
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
