"""``snakeppo`` command line: train, eval, agents, rewards, compare, sweep, results, serve."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from snakeppo.agents import REGISTRY, PolicyPlayer, available_agents, build_agent, count_parameters
from snakeppo.agents.networks import PolicyNetwork
from snakeppo.builders import (
    EVAL_SEED_OFFSET,
    load_policy,
    make_play_env,
    make_policy,
    resolve_device,
)
from snakeppo.config import (
    DEFAULT_CONFIGS_DIR,
    Config,
    ConfigError,
    EnvConfig,
    Preset,
    config_from_dict,
    describe_changes,
    list_presets,
    load_config,
    run_description,
    validate,
)
from snakeppo.dashboard.server import DEFAULT_PORT
from snakeppo.evaluation import evaluate
from snakeppo.results import BOOKKEEPING_KEYS, FORMATS, format_results, results_table, truncate
from snakeppo.rewards import REWARDS, available_rewards
from snakeppo.runs import (
    CONFIG_FILE,
    find_runs,
    load_checkpoint,
    load_run,
    resolve_checkpoint,
    summarize_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snakeppo", description="Train and evaluate PPO agents that play Snake."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train an agent and write a run folder")
    _experiment_options(train)
    train.add_argument("--seed", type=int, help="seed for weights, environments and sampling")
    train.add_argument("--group", help="seed-group id recorded in run.json (sweeps set it)")
    train.add_argument(
        "--resume",
        type=Path,
        metavar="RUN",
        help="continue an interrupted run folder exactly; takes no other option",
    )

    ev = commands.add_parser("eval", help="evaluate a trained run or checkpoint")
    ev.add_argument("target", type=Path, help="run folder or .pt checkpoint file")
    ev.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ev.add_argument("--episodes", type=int, help="episodes to play (default: train.eval_episodes)")
    ev.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    ev.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ev.add_argument(
        "--board-size",
        type=int,
        metavar="N",
        help="play on an NxN board (default: the run's board; at least 5)",
    )
    ev.add_argument("--render", action="store_true", help="watch the agent play in a window")
    ev.add_argument("--envs", type=int, default=4, help="games shown with --render (1-4)")
    ev.add_argument("--sleep", type=float, default=0.05, help="seconds between rendered steps")

    agents = commands.add_parser("agents", help="list available architectures")
    agents.add_argument("--board-size", type=int, default=16)

    commands.add_parser("rewards", help="list available reward strategies")

    compare = commands.add_parser("compare", help="summarise runs as a table")
    compare.add_argument("paths", nargs="*", type=Path, default=[Path("runs")])

    sweep = commands.add_parser("sweep", help="train one run per seed as one seed group")
    sweep.add_argument(
        "--seeds", required=True, type=_seed_list, help="comma-separated seeds, e.g. 1,2,3"
    )
    _experiment_options(sweep)
    sweep.add_argument("--jobs", type=int, default=1, help="seeds trained at once (default 1)")
    sweep.add_argument(
        "--group",
        help="seed group to join or name (default: NAME@START-TIME); "
        "seeds already in it are resumed or left alone",
    )

    results = commands.add_parser(
        "results", help="judge seed groups against their baseline group across seeds"
    )
    results.add_argument("paths", nargs="*", type=Path, default=[Path("runs")])
    results.add_argument("--baseline", metavar="GROUP", help="compare every group with GROUP")
    results.add_argument("--format", choices=FORMATS, default="text")

    serve = commands.add_parser("serve", help="open the local dashboard (main UI)")
    serve.add_argument("--runs-dir", type=Path, default=Path("runs"))
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    serve.add_argument(
        "--configs-dir",
        type=Path,
        metavar="PATH",
        help="preset folder the Launch view offers (default: configs in this folder)",
    )
    return parser


def _experiment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML experiment file (partial is fine)")
    parser.add_argument("--agent", help="architecture name; see 'snakeppo agents'")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value; repeatable",
    )
    parser.add_argument("--name", help="run name used in the run folder name")
    parser.add_argument(
        "--description",
        metavar="TEXT",
        help="plain-language note saved with the run, for people only "
        "(default: the preset's, plus the options changed here)",
    )


def _experiment_config(args: argparse.Namespace, overrides: Sequence[str]) -> Config:
    config = load_config(args.config, overrides)
    # Applied after parsing so names like "2" or "off" stay text instead of becoming YAML values.
    if args.agent:
        config = validate(replace(config, agent=replace(config.agent, name=args.agent)))
    if args.name:
        config = validate(replace(config, train=replace(config.train, name=args.name)))
    description = _description(args, overrides, config.train.description)
    return validate(replace(config, train=replace(config.train, description=description)))


def _description(args: argparse.Namespace, overrides: Sequence[str], loaded: str) -> str:
    """``--description`` verbatim, else an explicit ``--set train.description``, else the
    loaded (preset) text plus the learning changes made on this command line."""
    if args.description is not None:
        return args.description
    keys = [override.partition("=")[0].strip() for override in overrides]
    if "train.description" in keys:
        return loaded
    # bookkeeping (seed, name, group, runs folder, ...) is left out, so every seed of a sweep
    # gets the same text
    changes = [
        override.strip()
        for override, key in zip(overrides, keys, strict=True)
        if key not in BOOKKEEPING_KEYS
    ]
    if args.agent:
        changes.append(f"agent.name={args.agent}")
    return describe_changes(loaded, changes)


def _seed_list(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from error


def _train(args: argparse.Namespace) -> int:
    from snakeppo.ppo.trainer import resume, train

    if args.resume is not None:
        given = [
            flag
            for flag, value in (
                ("--config", args.config),
                ("--set", args.overrides),
                ("--agent", args.agent),
                ("--name", args.name),
                ("--seed", args.seed),
                ("--group", args.group),
                ("--description", args.description),
            )
            if value not in (None, [])
        ]
        if given:
            raise ValueError(
                f"--resume continues a run exactly and cannot be combined with {', '.join(given)}"
            )
        resume(args.resume)
        return 0

    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"env.seed={args.seed}")
    config = _experiment_config(args, overrides)
    if args.group:
        config = validate(replace(config, train=replace(config.train, group=args.group)))
    train(config)
    return 0


def _load_model(args: argparse.Namespace) -> tuple[PolicyNetwork, Config, Path, torch.device]:
    run_dir, checkpoint_path = resolve_checkpoint(args.target, args.checkpoint)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = config_from_dict(checkpoint["config"])
    model = load_policy(checkpoint, config, checkpoint_path)
    return model.to(device).eval(), config, checkpoint_path, device


def _play_env_config(config: Config, board_size: int | None) -> EnvConfig:
    """The run's env section, on ``board_size`` if given.

    Not re-validated: the rest of the config describes training (a curriculum's ``start_board``
    may not fit a smaller play board), and eval and watch read only the env and observation.
    """
    if board_size is None:
        return config.env
    if board_size < 5:
        raise ValueError(f"--board-size must be at least 5, got {board_size}")
    trained, name = config.env.board_size, config.agent.name
    if (
        board_size != trained
        and config.observation.view == "full"
        and not REGISTRY[name].size_agnostic
    ):
        raise ValueError(
            f"agent '{name}' is board-bound: it trained on a {trained}x{trained} board and "
            f"cannot play {board_size}x{board_size}; use a size-agnostic architecture "
            "(see 'snakeppo agents') or train with observation.view: local"
        )
    return replace(config.env, board_size=board_size)


def _eval(args: argparse.Namespace) -> int:
    model, config, checkpoint_path, device = _load_model(args)
    play = _play_env_config(config, args.board_size)
    greedy = not args.stochastic
    if args.render:
        return _watch(model, config, play, device, args, greedy)
    episodes = args.episodes or config.train.eval_episodes
    stats = evaluate(
        model,
        play,
        config.observation,
        episodes=episodes,
        device=device,
        seed=config.env.seed + EVAL_SEED_OFFSET,
        greedy=greedy,
    )
    mode = "greedy" if greedy else "stochastic"
    per_apple = "-" if stats.steps_per_apple is None else f"{stats.steps_per_apple:.1f}"
    print(
        f"Evaluated {checkpoint_path} on a {play.board_size}x{play.board_size} board "
        f"over {stats.episodes} episodes ({mode}): "
        f"score mean {stats.score_mean:.2f}, max {stats.score_max:.0f}, "
        f"min {stats.score_min:.0f}; episode length mean {stats.length_mean:.1f}; "
        f"ended by wall {stats.end_wall:.0%}, body {stats.end_body:.0%}, "
        f"starvation {stats.end_starved:.0%}; steps per apple {per_apple}"
    )
    return 0


def _watch(
    model: PolicyNetwork,
    config: Config,
    play: EnvConfig,
    device: torch.device,
    args: argparse.Namespace,
    greedy: bool,
) -> int:
    from snakeppo.viewer import Viewer

    env = make_play_env(
        play,
        config.observation,
        num_envs=max(1, min(args.envs, 4)),
        device=device,
        seed=play.seed + EVAL_SEED_OFFSET,
    )
    player = PolicyPlayer(model, env.num_envs, device)
    obs = env.reset()
    finished = 0
    print("Close the window to stop.")
    with Viewer(env.board_size, env.num_envs) as viewer, torch.no_grad():
        while viewer.draw(env.render_grid(), env.score.tolist(), env.render_body_order()):
            obs, _, _, _, info = env.step(player.act(obs, greedy))
            player.end_episodes(info["done"])
            for score in info["episode_score"][info["done"]].tolist():
                finished += 1
                print(f"Episode {finished} finished with score {score:.0f}")
            if args.episodes and finished >= args.episodes:
                break
            time.sleep(args.sleep)
    return 0


def _agents(args: argparse.Namespace) -> int:
    print(f"Architectures (parameter counts for a {args.board_size}x{args.board_size} board):")
    print(f"  {'name':<12} {'parameters':>12}  {'any size':<8}  description")
    for name in available_agents():
        spec = REGISTRY[name]
        try:
            params = f"{count_parameters(build_agent(name, args.board_size)):,}"
        except ValueError as error:
            params = f"n/a ({error})"
        any_size = "yes" if spec.size_agnostic else "no"
        print(f"  {name:<12} {params:>12}  {any_size:<8}  {spec.description}")
    return 0


def _rewards(args: argparse.Namespace) -> int:
    print("Reward strategies (choose one with --set reward.strategy=NAME; default classic):")
    for name in available_rewards():
        print(f"  {name:<11} {REWARDS[name].description}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    runs = find_runs(args.paths)
    if not runs:
        joined = ", ".join(str(p) for p in args.paths)
        print(f"No run folders found in {joined}.")
        return 1
    presets = list_presets(DEFAULT_CONFIGS_DIR)
    rows = [
        summarize_run(run) | {"description": truncate(_run_description(run, presets)) or None}
        for run in runs
    ]
    rows.sort(key=lambda r: -1e9 if r["best_eval"] is None else r["best_eval"], reverse=True)
    header = ("run", "agent", "seed", "env_steps", "best_eval", "last_eval", "description")
    table = [header] + [
        tuple(
            "-" if row[k] is None else f"{row[k]:.2f}" if "eval" in k else str(row[k])
            for k in header
        )
        for row in rows
    ]
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    for line in table:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)))
    return 0


def _run_description(run_dir: Path, presets: Sequence[Preset]) -> str:
    """The run's description; empty when its saved config no longer loads (compare reads the
    other columns from the raw file, so such a run still gets its row)."""
    try:
        config = load_config(run_dir / CONFIG_FILE)
    except ConfigError:
        return ""
    return run_description(config, presets)


def _sweep(args: argparse.Namespace) -> int:
    from snakeppo.processes import LAUNCH_ENV, STOP_REQUEST_FILE, launch_dir
    from snakeppo.sweep import format_outcomes, group_id, run_sweep

    # resolve the config and build its network exactly as each child will, so a bad option
    # (agent, agent.kwargs, observation) fails once, up front
    config = _experiment_config(args, args.overrides)
    make_policy(config)
    train_args = [] if args.config is None else ["--config", str(args.config)]
    for override in args.overrides:
        train_args += ["--set", override]
    for flag, value in (("--agent", args.agent), ("--name", args.name)):
        if value:
            train_args += [flag, value]
    if args.description is not None:
        # one token, so a text that starts with "-" is not read as an option
        train_args.append(f"--description={args.description}")
    group = args.group or group_id(config.train.name)
    seeds = ", ".join(map(str, args.seeds))
    print(f"Sweep {group}: seeds {seeds}, {args.jobs} at a time")
    runs_dir = Path(config.train.runs_dir)
    # a sweep the dashboard launched is stopped through a file in its launch folder
    launch = os.environ.get(LAUNCH_ENV) or None
    stop_file = launch_dir(runs_dir, launch) / STOP_REQUEST_FILE if launch else None
    outcomes = run_sweep(
        args.seeds,
        train_args,
        runs_dir=runs_dir,
        group=group,
        jobs=args.jobs,
        stop_file=stop_file,
        detached=launch is not None,
    )
    print(format_outcomes(group, outcomes))
    if stop_file is not None and stop_file.exists():
        return 130
    return 0 if all(outcome.returncode in (0, None) for outcome in outcomes) else 1


def _results(args: argparse.Namespace) -> int:
    runs = find_runs(args.paths)
    if not runs:
        joined = ", ".join(str(p) for p in args.paths)
        print(f"No run folders found in {joined}.")
        return 1
    presets = list_presets(DEFAULT_CONFIGS_DIR)
    table = results_table([load_run(run) for run in runs], args.baseline, presets)
    if not table.rows:
        print(
            f"No seed groups found ({table.left_out} ungrouped run(s)); "
            "create one with 'snakeppo sweep' or 'snakeppo train --group'."
        )
        return 1
    print(format_results(table, args.format), end="")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from snakeppo.dashboard.server import serve

    return serve(args.runs_dir, args.port, open_browser=args.open, configs_dir=args.configs_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "train": _train,
        "eval": _eval,
        "agents": _agents,
        "rewards": _rewards,
        "compare": _compare,
        "sweep": _sweep,
        "results": _results,
        "serve": _serve,
    }
    try:
        return handlers[args.command](args)
    except (ConfigError, ValueError, FileNotFoundError) as error:
        print(f"snakeppo {args.command}: error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"snakeppo {args.command}: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
