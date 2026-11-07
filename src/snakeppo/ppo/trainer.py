"""The PPO training loop: fixed-window rollouts, GAE, clipped updates, periodic eval."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from snakeppo.agents import build_agent, count_parameters
from snakeppo.builders import EVAL_SEED_OFFSET, make_env, resolve_device, seed_everything
from snakeppo.config import Config, config_to_dict, save_config, validate
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
    MetricsLogger,
    create_run_dir,
    save_checkpoint,
    total_updates,
    write_run_info,
)


class RolloutBuffer:
    """Preallocated (T, N) storage; observations are kept as uint8 and widened on read."""

    def __init__(
        self, steps: int, num_envs: int, obs_shape: tuple[int, ...], device: torch.device
    ) -> None:
        shape = (steps, num_envs)
        self.obs = torch.zeros((*shape, *obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.next_values = torch.zeros(shape, device=device)
        self.rewards = torch.zeros(shape, device=device)
        self.terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self.done = torch.zeros(shape, dtype=torch.bool, device=device)


def train(config: Config, *, progress: bool = True) -> Path:
    """Train one run described by ``config``; returns the run folder it wrote."""
    validate(config)
    device = resolve_device(config.train.device)
    seed_everything(config.env.seed, config.train.deterministic)
    env_cfg, ppo = config.env, config.ppo
    env = make_env(
        env_cfg, config.reward, num_envs=env_cfg.num_envs, device=device, seed=env_cfg.seed
    )
    model = build_agent(config.agent.name, env_cfg.board_size, **config.agent.kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.lr, eps=1e-5)

    steps, n = ppo.rollout_steps, env_cfg.num_envs
    batch_size = steps * n
    num_updates = total_updates(config)

    run_dir = create_run_dir(config.train.runs_dir, config.train.name, env_cfg.seed)
    save_config(config, run_dir / CONFIG_FILE)
    write_run_info(
        run_dir,
        name=config.train.name,
        agent=config.agent.name,
        param_count=count_parameters(model),
        device=str(device),
        total_updates=num_updates,
        group=None,
    )
    config_dict = config_to_dict(config)
    print(f"Training '{config.agent.name}' on {device}; writing run folder {run_dir}")
    buffer = RolloutBuffer(steps, n, env.obs_shape, device)
    obs = env.reset()
    env_steps, best_eval = 0, -math.inf
    checkpoints = run_dir / CHECKPOINT_DIR

    with MetricsLogger(run_dir) as logger:
        bar = tqdm(range(1, num_updates + 1), desc="Training", unit="update", disable=not progress)
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
            with torch.no_grad():
                for t in range(steps):
                    actions, log_probs, values = model.act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(actions)
                    buffer.obs[t] = obs.to(torch.uint8)
                    buffer.actions[t] = actions
                    buffer.log_probs[t] = log_probs
                    buffer.values[t] = values
                    buffer.rewards[t] = reward
                    buffer.terminated[t] = terminated
                    buffer.done[t] = info["done"]
                    buffer.next_values[t] = 0.0
                    if truncated.any():
                        _, final_values = model(info["final_obs"][truncated])
                        buffer.next_values[t, truncated] = final_values
                    done = info["done"]
                    if done.any():
                        finished_scores.append(info["episode_score"][done])
                        finished_lengths.append(info["episode_length"][done].float())
                        finished_returns.append(info["episode_return"][done])
                    obs = next_obs
                _, last_values = model(obs)
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
                order = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, ppo.minibatch_size):
                    mb = order[start : start + ppo.minibatch_size]
                    logits, new_values = model(flat_obs[mb].float())
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

            if update % config.train.eval_interval == 0 or update == num_updates:
                stats = evaluate(
                    model,
                    env_cfg,
                    config.reward,
                    episodes=config.train.eval_episodes,
                    device=device,
                    seed=env_cfg.seed + EVAL_SEED_OFFSET,
                )
                metrics |= stats.as_metrics()
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "update": update,
                    "env_steps": env_steps,
                    "config": config_dict,
                }
                save_checkpoint(checkpoints / "last.pt", **state)
                if stats.score_mean > best_eval:
                    best_eval = stats.score_mean
                    save_checkpoint(checkpoints / "best.pt", **state)
                bar.set_postfix(eval_score=f"{stats.score_mean:.2f}", best=f"{best_eval:.2f}")
            logger.log(update, env_steps, metrics)

    print(f"Finished training: best eval score {best_eval:.2f}. Run folder: {run_dir}")
    return run_dir
