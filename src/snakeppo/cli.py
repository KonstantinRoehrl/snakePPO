"""``snakeppo`` command line: train, eval, agents, compare, sweep, results, serve."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from snakeppo.agents import REGISTRY, available_agents, build_agent, count_parameters
from snakeppo.agents.networks import PolicyNetwork
from snakeppo.builders import EVAL_SEED_OFFSET, make_env, resolve_device
from snakeppo.config import Config, ConfigError, config_from_dict, load_config, validate
from snakeppo.dashboard.server import DEFAULT_PORT
from snakeppo.evaluation import evaluate
from snakeppo.results import FORMATS, format_results, results_table
from snakeppo.runs import (
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
    ev.add_argument("--render", action="store_true", help="watch the agent play in a window")
    ev.add_argument("--envs", type=int, default=4, help="games shown with --render (1-4)")
    ev.add_argument("--sleep", type=float, default=0.05, help="seconds between rendered steps")

    agents = commands.add_parser("agents", help="list available architectures")
    agents.add_argument("--board-size", type=int, default=16)

    compare = commands.add_parser("compare", help="summarise runs as a table")
    compare.add_argument("paths", nargs="*", type=Path, default=[Path("runs")])

    sweep = commands.add_parser("sweep", help="train one run per seed as one seed group")
    sweep.add_argument(
        "--seeds", required=True, type=_seed_list, help="comma-separated seeds, e.g. 1,2,3"
    )
    _experiment_options(sweep)
    sweep.add_argument("--jobs", type=int, default=1, help="seeds trained at once (default 1)")

    results = commands.add_parser(
        "results", help="judge seed groups against their baseline group across seeds"
    )
    results.add_argument("paths", nargs="*", type=Path, default=[Path("runs")])
    results.add_argument("--baseline", metavar="GROUP", help="compare every group with GROUP")
    results.add_argument("--format", choices=FORMATS, default="text")

    serve = commands.add_parser("serve", help="open the read-only local dashboard")
    serve.add_argument("--runs-dir", type=Path, default=Path("runs"))
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--open", action="store_true", help="open the dashboard in the browser")
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


def _experiment_config(args: argparse.Namespace, overrides: Sequence[str]) -> Config:
    config = load_config(args.config, overrides)
    # Applied after parsing so names like "2" or "off" stay text instead of becoming YAML values.
    if args.agent:
        config = validate(replace(config, agent=replace(config.agent, name=args.agent)))
    if args.name:
        config = validate(replace(config, train=replace(config.train, name=args.name)))
    return config


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
    model = build_agent(config.agent.name, config.env.board_size, **config.agent.kwargs)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, checkpoint_path, device


def _eval(args: argparse.Namespace) -> int:
    model, config, checkpoint_path, device = _load_model(args)
    greedy = not args.stochastic
    if args.render:
        return _watch(model, config, device, args, greedy)
    episodes = args.episodes or config.train.eval_episodes
    stats = evaluate(
        model,
        config.env,
        config.reward,
        episodes=episodes,
        device=device,
        seed=config.env.seed + EVAL_SEED_OFFSET,
        greedy=greedy,
    )
    mode = "greedy" if greedy else "stochastic"
    print(
        f"Evaluated {checkpoint_path} over {stats.episodes} episodes ({mode}): "
        f"score mean {stats.score_mean:.2f}, max {stats.score_max:.0f}, "
        f"min {stats.score_min:.0f}; episode length mean {stats.length_mean:.1f}"
    )
    return 0


def _watch(
    model: PolicyNetwork,
    config: Config,
    device: torch.device,
    args: argparse.Namespace,
    greedy: bool,
) -> int:
    from snakeppo.viewer import Viewer

    env = make_env(
        config.env,
        config.reward,
        num_envs=max(1, min(args.envs, 4)),
        device=device,
        seed=config.env.seed + EVAL_SEED_OFFSET,
    )
    obs = env.reset()
    finished = 0
    print("Close the window to stop.")
    with Viewer(config.env.board_size, env.num_envs) as viewer, torch.no_grad():
        while viewer.draw(env.render_grid(), env.score.tolist()):
            actions, _, _ = model.act(obs, greedy=greedy)
            obs, _, _, _, info = env.step(actions)
            for score in info["episode_score"][info["done"]].tolist():
                finished += 1
                print(f"Episode {finished} finished with score {score:.0f}")
            if args.episodes and finished >= args.episodes:
                break
            time.sleep(args.sleep)
    return 0


def _agents(args: argparse.Namespace) -> int:
    print(f"Architectures (parameter counts for a {args.board_size}x{args.board_size} board):")
    for name in available_agents():
        spec = REGISTRY[name]
        try:
            params = f"{count_parameters(build_agent(name, args.board_size)):,}"
        except ValueError as error:
            params = f"n/a ({error})"
        print(f"  {name:<12} {params:>12}  {spec.description}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    runs = find_runs(args.paths)
    if not runs:
        joined = ", ".join(str(p) for p in args.paths)
        print(f"No run folders found in {joined}.")
        return 1
    rows = [summarize_run(run) for run in runs]
    rows.sort(key=lambda r: -1e9 if r["best_eval"] is None else r["best_eval"], reverse=True)
    header = ("run", "agent", "seed", "env_steps", "best_eval", "last_eval")
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


def _sweep(args: argparse.Namespace) -> int:
    from snakeppo.sweep import format_outcomes, group_id, run_sweep

    # resolve the config exactly as each child will, so a bad option fails once, up front
    config = _experiment_config(args, args.overrides)
    if config.agent.name not in REGISTRY:
        raise ValueError(f"unknown agent '{config.agent.name}'; see 'snakeppo agents'")
    train_args = [] if args.config is None else ["--config", str(args.config)]
    for override in args.overrides:
        train_args += ["--set", override]
    for flag, value in (("--agent", args.agent), ("--name", args.name)):
        if value:
            train_args += [flag, value]
    group = group_id(config.train.name)
    seeds = ", ".join(map(str, args.seeds))
    print(f"Sweep {group}: seeds {seeds}, {args.jobs} at a time")
    outcomes = run_sweep(
        args.seeds,
        train_args,
        runs_dir=Path(config.train.runs_dir),
        group=group,
        jobs=args.jobs,
    )
    print(format_outcomes(group, outcomes))
    return 0 if all(outcome.returncode == 0 for outcome in outcomes) else 1


def _results(args: argparse.Namespace) -> int:
    runs = find_runs(args.paths)
    if not runs:
        joined = ", ".join(str(p) for p in args.paths)
        print(f"No run folders found in {joined}.")
        return 1
    table = results_table([load_run(run) for run in runs], args.baseline)
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

    return serve(args.runs_dir, args.port, open_browser=args.open)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "train": _train,
        "eval": _eval,
        "agents": _agents,
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
