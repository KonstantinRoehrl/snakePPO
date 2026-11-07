"""``snakeppo`` command line: train, eval, agents, compare."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from snakeppo.agents import REGISTRY, available_agents, build_agent, count_parameters
from snakeppo.agents.networks import PolicyNetwork
from snakeppo.builders import EVAL_SEED_OFFSET, make_env, resolve_device
from snakeppo.config import Config, ConfigError, config_from_dict, load_config
from snakeppo.evaluation import evaluate
from snakeppo.runs import find_runs, load_checkpoint, resolve_checkpoint, summarize_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snakeppo", description="Train and evaluate PPO agents that play Snake."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train an agent and write a run folder")
    train.add_argument("--config", type=Path, help="YAML experiment file (partial is fine)")
    train.add_argument("--agent", help="architecture name; see 'snakeppo agents'")
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value; repeatable",
    )
    train.add_argument("--name", help="run name used in the run folder name")
    train.add_argument("--seed", type=int, help="seed for weights, environments and sampling")

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
    return parser


def _train(args: argparse.Namespace) -> int:
    from snakeppo.ppo.trainer import train

    overrides = list(args.overrides)
    if args.agent:
        overrides.append(f"agent.name={args.agent}")
    if args.name:
        overrides.append(f"train.name={args.name}")
    if args.seed is not None:
        overrides.append(f"env.seed={args.seed}")
    config = load_config(args.config, overrides)
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {"train": _train, "eval": _eval, "agents": _agents, "compare": _compare}
    try:
        return handlers[args.command](args)
    except (ConfigError, ValueError, FileNotFoundError) as error:
        print(f"snakeppo {args.command}: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
