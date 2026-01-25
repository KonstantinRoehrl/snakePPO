import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from agent import PPOAgent
from environment import TorchSnakeEnv

# ==========================
# Setup
# ==========================
os.makedirs(cfg.MODEL_DIR, exist_ok=True)
writer = SummaryWriter(cfg.LOG_DIR)

env = TorchSnakeEnv(device=cfg.DEVICE)
agent = PPOAgent().to(cfg.DEVICE)
optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.LR)

best_eval = -1
step = 0

# ==========================
# Episode Tracking
# ==========================
episode_lengths = torch.zeros(cfg.NUM_ENVS, device=cfg.DEVICE)
episode_length_buffer = []

# ==========================
# Training Loop
# ==========================
while True:
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

    obs, rotated_obs = env.reset()
    episode_lengths.zero_()

    # ==========================
    # Rollout Collection
    # ==========================
    for t in tqdm(range(cfg.ROLLOUT_STEPS), desc="Gathering rollouts"):
        with torch.no_grad():
            actions, logp, values, dist = agent.sample_action(
                rotated_obs
            )

        next_obs, next_rotated_obs, rewards, dones = env.step(actions)

        # Track episode lengths
        episode_lengths += 1

        if dones.any():
            completed_lengths = episode_lengths[dones]
            episode_length_buffer.extend(completed_lengths.cpu().tolist())
            episode_lengths[dones] = 0

        # Store rollout data
        obs_buf.append(rotated_obs)
        act_buf.append(actions)
        logp_buf.append(logp)
        rew_buf.append(rewards)
        done_buf.append(dones)
        val_buf.append(values.squeeze())

        obs = next_obs
        rotated_obs = next_rotated_obs

        # ✅ Stop rollout early if all envs are done
        if dones.all():
            print(f"Stopping rollout early at step {t+1} — all envs finished.")
            break

    T = len(obs_buf)

    # Skip update if no rollout collected
    if T == 0:
        print("Skipping PPO update — no rollout steps collected.")
        continue

    # ==========================
    # Stack Rollout Buffers
    # ==========================
    obs_buf = torch.stack(obs_buf)
    act_buf = torch.stack(act_buf)
    logp_buf = torch.stack(logp_buf)
    rew_buf = torch.stack(rew_buf)
    done_buf = torch.stack(done_buf)
    val_buf = torch.stack(val_buf)

    # ==========================
    # Bootstrap Value
    # ==========================
    with torch.no_grad():
        _, last_values = agent(rotated_obs)
        last_values = last_values.squeeze()

    # ==========================
    # GAE Advantage Calculation
    # ==========================
    advantages = torch.zeros_like(rew_buf)
    gae = torch.zeros(cfg.NUM_ENVS, device=cfg.DEVICE)

    for t in reversed(range(T)):
        mask = 1.0 - done_buf[t].float()
        next_val = last_values if t == T - 1 else val_buf[t + 1]
        delta = rew_buf[t] + cfg.GAMMA * next_val * mask - val_buf[t]
        gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * mask * gae
        advantages[t] = gae

    returns = advantages + val_buf
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ==========================
    # Flatten for PPO
    # ==========================
    flat_obs = obs_buf.reshape(T * cfg.NUM_ENVS, 4, cfg.GAME_SIZE, cfg.GAME_SIZE)
    flat_act = act_buf.flatten()
    flat_logp = logp_buf.flatten()
    flat_adv = advantages.flatten()
    flat_ret = returns.flatten()

    total_steps = flat_obs.size(0)

    # ==========================
    # PPO Training
    # ==========================
    policy_losses = []
    value_losses = []
    entropy_losses = []
    total_losses = []

    for _ in tqdm(range(cfg.PPO_EPOCHS), desc="PPO training"):
        idx = torch.randperm(total_steps)

        for start in range(0, total_steps, cfg.MINIBATCH_SIZE):
            mb = idx[start:start + cfg.MINIBATCH_SIZE]

            logits, values = agent(flat_obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(flat_act[mb])

            ratio = torch.exp(new_logp - flat_logp[mb])
            clipped = torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS)

            policy_loss = -torch.min(
                ratio * flat_adv[mb],
                clipped * flat_adv[mb]
            ).mean()

            value_loss = (values.squeeze() - flat_ret[mb]).pow(2).mean()
            entropy_loss = dist.entropy().mean() * cfg.ENTROPY_COEF

            loss = policy_loss + cfg.VALUE_COEF * value_loss - entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(loss.item())

    cfg.ENTROPY_COEF *= 0.995  # Decay entropy coefficient

    # ==========================
    # Metrics & Logging
    # ==========================
    avg_score = env.score.mean().item()
    avg_reward = rew_buf.mean().item()
    avg_advantage = advantages.mean().item()

    avg_policy_loss = sum(policy_losses) / len(policy_losses)
    avg_value_loss = sum(value_losses) / len(value_losses)
    avg_entropy = sum(entropy_losses) / len(entropy_losses)
    avg_total_loss = sum(total_losses) / len(total_losses)

    if len(episode_length_buffer) > 0:
        avg_episode_length = sum(episode_length_buffer) / len(episode_length_buffer)
        episode_length_buffer.clear()
    else:
        avg_episode_length = 0

    writer.add_scalar("Performance/avg_score", avg_score, step)
    writer.add_scalar("Performance/avg_reward", avg_reward, step)
    writer.add_scalar("Performance/avg_episode_length", avg_episode_length, step)
    writer.add_scalar("Performance/avg_advantage", avg_advantage, step)

    writer.add_scalars("Losses_Combined", {
        "Policy": avg_policy_loss,
        "Value": avg_value_loss,
        "Entropy": avg_entropy,
        "Total": avg_total_loss
    }, step)

    # ==========================
    # Save Best Model
    # ==========================
    if avg_score > best_eval:
        best_eval = avg_score
        torch.save(agent.state_dict(), os.path.join(cfg.MODEL_DIR, "best_model.pt"))
        print(f"→ New best model saved! Score: {best_eval:.2f}")

    step += 1
