"""The PPO training loop: fixed-window rollouts, GAE, clipped updates, periodic eval.

A recurrent policy carries a hidden state ``(N, K, H)`` through the rollout, zeroed where an
episode starts and after a curriculum promotion's reset; its update replays groups of whole envs
from the rollout's start state instead of shuffling single steps.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from snakeppo.agents import RecurrentPolicy, count_parameters, episode_starts
from snakeppo.agents.networks import PolicyNetwork
from snakeppo.builders import (
    EVAL_SEED_OFFSET,
    capture_rng_state,
    load_policy,
    make_env,
    make_policy,
    observation_record,
    recurrent_minibatches,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from snakeppo.config import (
    Config,
    config_from_dict,
    config_to_dict,
    save_config,
    total_updates,
    validate,
)
from snakeppo.curriculum import PROMOTED_NONE, Curriculum
from snakeppo.env import TorchSnakeEnv
from snakeppo.evaluation import evaluate
from snakeppo.ppo.functional import (
    compute_gae,
    entropy_coef_at,
    explained_variance,
    lr_at,
    ppo_loss,
)
from snakeppo.runs import (
    CHECKPOINT_DIR,
    CONFIG_FILE,
    RESUMABLE_SCHEMAS,
    RUN_INFO_FILE,
    SCHEMA_VERSION,
    MetricsLogger,
    create_run_dir,
    last_write_time,
    load_checkpoint,
    load_run,
    save_checkpoint,
    stall_window,
    truncate_metrics,
    update_run_info,
    write_run_info,
)


class RolloutBuffer:
    """Preallocated (T, N) storage; observations are stored as ``obs_dtype`` and widened on read.

    ``uint8`` is exact only for an observation whose every value is 0 or 1 (the default one);
    the trainer passes ``float32`` for any other, so fractions are never truncated.
    """

    def __init__(
        self,
        steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        obs_dtype: torch.dtype = torch.uint8,
    ) -> None:
        shape = (steps, num_envs)
        self.obs = torch.zeros((*shape, *obs_shape), dtype=obs_dtype, device=device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.next_values = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self.done = torch.zeros(shape, dtype=torch.bool, device=device)


@dataclass(frozen=True)
class Progress:
    """Where a run's loop starts: zeros for a new run, the checkpoint's values on resume."""

    update: int = 0
    env_steps: int = 0
    best_eval: float = -math.inf
    wall_time: float = 0.0


def train(config: Config, *, progress: bool = True) -> Path:
    """Train one run described by ``config``; returns the run folder it wrote."""
    validate(config)
    device = resolve_device(config.train.device)
    seed_everything(config.env.seed, config.train.deterministic)
    env = _make_train_env(config, device)
    curriculum = _make_curriculum(config, env)
    env_cfg = config.env
    model = make_policy(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr, eps=1e-5)

    run_dir = create_run_dir(config.train.runs_dir, config.train.name, env_cfg.seed)
    save_config(config, run_dir / CONFIG_FILE)
    write_run_info(
        run_dir,
        name=config.train.name,
        agent=config.agent.name,
        param_count=count_parameters(model),
        device=str(device),
        total_updates=total_updates(config),
        group=config.train.group,
        observation=observation_record(config),
    )
    print(f"Training '{config.agent.name}' on {device}; writing run folder {run_dir}")
    obs = env.reset()
    hidden = None
    if isinstance(model, RecurrentPolicy):
        hidden = model.initial_state(env_cfg.num_envs, device)
    session = _Session(run_dir, config, device, env, model, optimizer, curriculum)
    return _run(session, obs, hidden, Progress(), progress=progress)


def _make_train_env(config: Config, device: torch.device) -> TorchSnakeEnv:
    env_cfg = config.env
    return make_env(
        env_cfg,
        config.reward,
        config.observation,
        num_envs=env_cfg.num_envs,
        device=device,
        seed=env_cfg.seed,
        discount=config.ppo.gamma,
    )


def _make_curriculum(config: Config, env: TorchSnakeEnv) -> Curriculum | None:
    """The run's curriculum, with ``env`` set to its first stage; ``None`` when it is off."""
    if config.curriculum.grow == "off":
        return None
    curriculum = Curriculum(config.curriculum, config.env.board_size)
    env.set_curriculum(curriculum.current.arena, curriculum.current.goal)
    return curriculum


class ResumeError(ValueError):
    """A run folder that cannot be continued exactly; the message says why."""


def _check_resumable(schema: int | None, what: Path) -> None:
    if schema in RESUMABLE_SCHEMAS:
        return
    if schema is None or schema < RESUMABLE_SCHEMAS[0]:
        raise ResumeError(f"{what} was written before resume support; cannot be resumed exactly")
    raise ResumeError(
        f"{what} has schema {schema}, newer than this snakeppo can resume ({SCHEMA_VERSION})"
    )


def resume(run_dir: str | Path, *, progress: bool = True) -> Path:
    """Continue an interrupted, failed or stalled run in place from ``checkpoints/last.pt``.

    The continuation is exact on CPU: the same metrics an uninterrupted run would have logged,
    apart from ``wall_time`` and ``train/fps``. A schema-2 or 3 run continues as schema 4: its
    config has no observation section and its checkpoints no observation record, so it loads
    with the default observation, which is the one it trained on.
    """
    run_dir = Path(run_dir)
    if not (run_dir / RUN_INFO_FILE).is_file():
        raise FileNotFoundError(f"{run_dir} is not a run folder (no {RUN_INFO_FILE})")
    record = load_run(run_dir)
    _check_resumable(record.info.get("schema_version", 1), run_dir)
    if record.status == "finished":
        raise ResumeError(f"{run_dir} already finished all {record.total_updates} updates")
    if record.status == "running":
        age = time.time() - last_write_time(run_dir)
        raise ResumeError(
            f"{run_dir} is still being trained: its metrics were written {age:.0f} s ago. If no "
            f"process is training it, retry once it shows as stalled "
            f"({stall_window(record.metrics):.0f} s without a write)"
        )
    last = run_dir / CHECKPOINT_DIR / "last.pt"
    if not last.is_file():
        raise ResumeError(f"{run_dir} has no {CHECKPOINT_DIR}/last.pt to resume from")
    checkpoint = load_checkpoint(last)
    _check_resumable(checkpoint.get("schema_version"), last)

    config = config_from_dict(checkpoint["config"])
    device = resolve_device(config.train.device)
    trained_on = torch.device(record.info["device"]).type
    if device.type != trained_on:
        raise ResumeError(
            f"{run_dir} trained on {trained_on} but would resume on {device.type}; "
            f"its {trained_on} random state cannot be restored there"
        )
    if device.type != "cpu" and not config.train.deterministic:
        print(
            "Note: exact continuation is only guaranteed on CPU or with train.deterministic: true"
        )

    seed_everything(config.env.seed, config.train.deterministic)
    env_cfg = config.env
    env = _make_train_env(config, device)
    curriculum = _make_curriculum(config, env)
    if curriculum is not None:
        if "curriculum" not in checkpoint:
            raise ResumeError(f"{last} has no curriculum state to continue from")
        curriculum.load_state_dict(checkpoint["curriculum"])
        env.set_curriculum(curriculum.current.arena, curriculum.current.goal)
    env.load_state_dict(checkpoint["env"])
    model = load_policy(checkpoint, config, last).to(device)
    hidden = None
    if isinstance(model, RecurrentPolicy):
        if "policy_state" not in checkpoint:
            raise ResumeError(f"{last} has no policy_state to continue a recurrent run from")
        hidden = checkpoint["policy_state"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr, eps=1e-5)
    optimizer.load_state_dict(checkpoint["optimizer"])

    update, env_steps = checkpoint["update"], checkpoint["env_steps"]
    dropped = truncate_metrics(run_dir, after_update=update)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    update_run_info(
        run_dir,
        status="running",
        resumed=[*record.info["resumed"], stamp],
        schema_version=SCHEMA_VERSION,
        observation=observation_record(config),
    )
    print(
        f"Resuming {run_dir} after update {update} of {record.total_updates} on {device} "
        f"(dropped {dropped} metrics lines logged after that checkpoint)"
    )
    # last: building the model above drew from the global RNG
    restore_rng_state(checkpoint["rng"])
    start = Progress(update, env_steps, checkpoint["best_eval"], checkpoint["wall_time"])
    session = _Session(run_dir, config, device, env, model, optimizer, curriculum)
    first_new_step = env_steps + config.ppo.rollout_steps * env_cfg.num_envs
    return _run(session, env.observe(), hidden, start, progress=progress, purge_step=first_new_step)


@dataclass(frozen=True)
class _Session:
    """The live objects one training process works on."""

    run_dir: Path
    config: Config
    device: torch.device
    env: TorchSnakeEnv
    model: PolicyNetwork
    optimizer: torch.optim.Optimizer
    curriculum: Curriculum | None = None


def _run(
    session: _Session,
    obs: torch.Tensor,
    hidden: torch.Tensor | None,
    start: Progress,
    *,
    progress: bool,
    purge_step: int | None = None,
) -> Path:
    """Loop from ``start.update + 1`` to the last update, recording the outcome in run.json.

    ``hidden`` is a recurrent model's state for ``obs`` (N, K, H); ``None`` for feed-forward.
    """
    run_dir = session.run_dir
    try:
        best_eval = _loop(session, obs, hidden, start, progress=progress, purge_step=purge_step)
    except KeyboardInterrupt:
        update_run_info(run_dir, status="interrupted")
        print(f"Interrupted. Continue with: snakeppo train --resume {run_dir}")
        raise
    except Exception:
        update_run_info(run_dir, status="failed")
        raise
    update_run_info(run_dir, status="finished")
    print(f"Finished training: best eval score {best_eval:.2f}. Run folder: {run_dir}")
    return run_dir


def _loop(
    session: _Session,
    obs: torch.Tensor,
    hidden: torch.Tensor | None,
    start: Progress,
    *,
    progress: bool,
    purge_step: int | None,
) -> float:
    run_dir, config, device = session.run_dir, session.config, session.device
    env, model, optimizer = session.env, session.model, session.optimizer
    env_cfg, ppo = config.env, config.ppo
    steps, n = ppo.rollout_steps, env_cfg.num_envs
    batch_size = steps * n
    num_updates = total_updates(config)
    obs_dtype = torch.uint8 if env.obs_is_binary else torch.float32
    buffer = RolloutBuffer(steps, n, env.obs_shape, device, obs_dtype)
    env_steps, best_eval = start.env_steps, start.best_eval
    checkpoints = run_dir / CHECKPOINT_DIR
    config_dict = config_to_dict(config)
    observation = observation_record(config)
    groups = recurrent_minibatches(config)
    no_starts = torch.zeros(n, dtype=torch.bool, device=device)

    with MetricsLogger(run_dir, wall_time_offset=start.wall_time, purge_step=purge_step) as logger:
        bar = tqdm(
            range(start.update + 1, num_updates + 1),
            desc="Training",
            unit="update",
            initial=start.update,
            total=num_updates,
            disable=not progress,
        )
        for update in bar:
            # the first update runs at the start value, the last one exactly at the end value
            fraction = (update - 1) / max(num_updates - 1, 1)
            lr = lr_at(ppo.lr_schedule, ppo.lr, fraction)
            entropy_coef = entropy_coef_at(
                ppo.entropy_schedule, ppo.entropy_coef, ppo.entropy_coef_final, fraction
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            started = time.perf_counter()
            finished_scores, finished_lengths, finished_returns = [], [], []
            rollout_start = hidden  # a recurrent update replays every env from here
            with torch.no_grad():
                for t in range(steps):
                    if hidden is None:
                        actions, log_probs, values = model.act(obs)
                    else:
                        actions, log_probs, values, hidden = model.act(obs, hidden, no_starts)
                    next_obs, reward, terminated, truncated, info = env.step(actions)
                    buffer.obs[t] = obs.to(buffer.obs.dtype)
                    buffer.actions[t] = actions
                    buffer.log_probs[t] = log_probs
                    buffer.values[t] = values
                    buffer.rewards[t] = reward
                    buffer.terminated[t] = terminated
                    buffer.done[t] = info["done"]
                    buffer.next_values[t] = 0.0
                    if truncated.any():
                        if hidden is None:
                            _, final_values = model(info["final_obs"][truncated])
                        else:  # from the state that produced this step, before its reset below
                            _, final_values, _ = model(
                                info["final_obs"][truncated],
                                hidden[truncated],
                                no_starts[truncated],
                            )
                        buffer.next_values[t, truncated] = final_values
                    done = info["done"]
                    if done.any():
                        finished_scores.append(info["episode_score"][done])
                        finished_lengths.append(info["episode_length"][done].float())
                        finished_returns.append(info["episode_return"][done])
                    obs = next_obs
                    if hidden is not None:  # the next step starts a new episode where done
                        hidden = hidden.masked_fill(done[:, None, None], 0.0)
                if hidden is None:
                    _, last_values = model(obs)
                else:
                    _, last_values, _ = model(obs, hidden, no_starts)
                # next state of a non-final step is the following step's observation
                not_done = ~buffer.done
                following = torch.cat([buffer.values[1:], last_values.unsqueeze(0)])
                buffer.next_values = torch.where(not_done, following, buffer.next_values)
                advantages, returns = compute_gae(
                    buffer.rewards,
                    buffer.values,
                    buffer.next_values,
                    buffer.terminated,
                    buffer.done,
                    ppo.gamma,
                    ppo.gae_lambda,
                )
            env_steps += batch_size

            flat_obs = buffer.obs.reshape(batch_size, *env.obs_shape)
            flat_actions = buffer.actions.reshape(-1)
            flat_log_probs = buffer.log_probs.reshape(-1)
            flat_values = buffer.values.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            if ppo.normalize_advantages:
                flat_advantages = (flat_advantages - flat_advantages.mean()) / (
                    flat_advantages.std() + 1e-8
                )

            totals = {key: 0.0 for key in ("total", "policy", "value", "entropy", "kl", "clip")}
            grad_norm_sum, minibatches, epochs_run, stop = 0.0, 0, 0, False
            for _ in range(ppo.update_epochs):
                epochs_run += 1
                if rollout_start is None:
                    batches = _shuffled_steps(model, flat_obs, ppo.minibatch_size, device)
                else:
                    batches = _env_groups(model, buffer, rollout_start, groups, device)
                for mb, logits, new_values in batches:
                    dist = torch.distributions.Categorical(logits=logits)
                    out = ppo_loss(
                        dist.log_prob(flat_actions[mb]),
                        flat_log_probs[mb],
                        flat_advantages[mb],
                        new_values,
                        flat_values[mb],
                        flat_returns[mb],
                        dist.entropy(),
                        clip_eps=ppo.clip_eps,
                        value_coef=ppo.value_coef,
                        entropy_coef=entropy_coef,
                        clip_value_loss=ppo.clip_value_loss,
                    )
                    optimizer.zero_grad()
                    out.total.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), ppo.max_grad_norm
                    )
                    optimizer.step()
                    minibatches += 1
                    grad_norm_sum += grad_norm.item()
                    for key, value in (
                        ("total", out.total),
                        ("policy", out.policy),
                        ("value", out.value),
                        ("entropy", out.entropy),
                        ("kl", out.approx_kl),
                        ("clip", out.clip_fraction),
                    ):
                        totals[key] += value.item()
                    if ppo.target_kl is not None and out.approx_kl.item() > ppo.target_kl:
                        stop = True
                        break
                if stop:
                    break

            elapsed = time.perf_counter() - started
            metrics = {
                "loss/total": totals["total"] / minibatches,
                "loss/policy": totals["policy"] / minibatches,
                "loss/value": totals["value"] / minibatches,
                "loss/entropy": totals["entropy"] / minibatches,
                "ppo/approx_kl": totals["kl"] / minibatches,
                "ppo/clip_fraction": totals["clip"] / minibatches,
                "ppo/explained_variance": explained_variance(flat_values, flat_returns),
                "ppo/lr": lr,
                "ppo/entropy_coef": entropy_coef,
                "ppo/grad_norm": grad_norm_sum / minibatches,
                "ppo/epochs": epochs_run,
                "train/fps": batch_size / elapsed,
                "train/reward_mean": buffer.rewards.mean().item(),
            }
            if finished_scores:
                scores = torch.cat(finished_scores)
                metrics |= {
                    "train/episodes": scores.numel(),
                    "train/score_mean": scores.mean().item(),
                    "train/score_max": scores.max().item(),
                    "train/length_mean": torch.cat(finished_lengths).mean().item(),
                    "train/return_mean": torch.cat(finished_returns).mean().item(),
                }
            if session.curriculum is not None:
                # between updates, and before eval and checkpoint, so a resume sees the new stage
                stage_scores = torch.cat(finished_scores).tolist() if finished_scores else []
                result = session.curriculum.end_update(stage_scores)
                metrics |= result.metrics
                if result.promoted != PROMOTED_NONE:
                    stage = session.curriculum.current
                    env.set_curriculum(stage.arena, stage.goal)
                    obs = env.reset()  # partial episodes are dropped, never counted
                    if hidden is not None:
                        hidden = torch.zeros_like(hidden)

            is_eval = update % config.train.eval_interval == 0 or update == num_updates
            improved = False
            if is_eval:
                stats = evaluate(
                    model,
                    env_cfg,
                    config.observation,
                    episodes=config.train.eval_episodes,
                    device=device,
                    seed=env_cfg.seed + EVAL_SEED_OFFSET,
                )
                metrics |= stats.as_metrics()
                improved = stats.score_mean > best_eval
                best_eval = max(best_eval, stats.score_mean)
                bar.set_postfix(eval_score=f"{stats.score_mean:.2f}", best=f"{best_eval:.2f}")
            # the metrics line goes first, so every checkpoint's update has its line
            logger.log(update, env_steps, metrics)
            if is_eval or update % config.train.checkpoint_interval == 0:
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "update": update,
                    "env_steps": env_steps,
                    "config": config_dict,
                    "best_eval": best_eval,
                    "wall_time": logger.elapsed(),
                    "rng": capture_rng_state(device),
                    "env": env.state_dict(),
                    "observation": observation,
                }
                if session.curriculum is not None:
                    state["curriculum"] = session.curriculum.state_dict()
                if hidden is not None:
                    state["policy_state"] = hidden.cpu()
                save_checkpoint(checkpoints / "last.pt", **state)
                if improved:
                    save_checkpoint(checkpoints / "best.pt", **state)
    return best_eval


def _shuffled_steps(
    model: PolicyNetwork, flat_obs: torch.Tensor, minibatch_size: int, device: torch.device
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One epoch of feed-forward minibatches of shuffled single steps.

    Yields ``(index, logits, values)``; ``index`` picks the minibatch's entries of the flattened
    (T * N) rollout.
    """
    batch_size = len(flat_obs)
    order = torch.randperm(batch_size, device=device)
    for start_index in range(0, batch_size, minibatch_size):
        mb = order[start_index : start_index + minibatch_size]
        logits, values = model(flat_obs[mb].float())
        yield mb, logits, values


def _env_groups(
    model: RecurrentPolicy,
    buffer: RolloutBuffer,
    start_state: torch.Tensor,
    groups: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One epoch of recurrent minibatches: ``groups`` shuffled groups of whole envs.

    Each group replays its envs' whole rollout from their start state, resetting the state where
    an episode starts, and yields ``(index, logits, values)`` over every step of those envs;
    ``index`` picks the same entries of the flattened (T * N) rollout.
    """
    steps, num_envs = buffer.done.shape
    starts = episode_starts(buffer.done)
    order = torch.randperm(num_envs, device=device)
    step_offsets = torch.arange(steps, device=device)[:, None] * num_envs
    for envs in order.view(groups, -1):
        logits, values, _ = model.unroll(
            buffer.obs[:, envs].float(), start_state[envs], starts[:, envs]
        )
        yield (step_offsets + envs).reshape(-1), logits.flatten(0, 1), values.flatten(0, 1)
