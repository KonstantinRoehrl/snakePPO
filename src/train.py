import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg
from agent import (DeepPPOAgent, PPOAgent, RecurrentPPOAgent,
                   SpatialAwarePPOAgent)
from environment import TorchSnakeEnv

# ==========================
# Argument Parsing
# ==========================
parser = argparse.ArgumentParser(description='Train Snake AI with PPO')
parser.add_argument('--agent', type=str, default='PPOAgent',
                    choices=['PPOAgent', 'SpatialAwarePPOAgent', 'RecurrentPPOAgent', 'DeepPPOAgent'],
                    help='Agent architecture to use')
args = parser.parse_args()

# ==========================
# Agent Selection
# ==========================
AGENT_CLASSES = {
    'PPOAgent': PPOAgent,
    'SpatialAwarePPOAgent': SpatialAwarePPOAgent,
    'RecurrentPPOAgent': RecurrentPPOAgent,
    'DeepPPOAgent': DeepPPOAgent
}

agent_name = args.agent
AgentClass = AGENT_CLASSES[agent_name]

# ==========================
# Setup Directories
# ==========================
model_dir = os.path.join(cfg.MODEL_DIR, agent_name)
log_dir = os.path.join(cfg.LOG_DIR)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

print(f"{'='*60}")
print(f"Training Configuration:")
print(f"  Device: {cfg.DEVICE}")
print(f"  Agent: {agent_name}")
print(f"  Epochs: {cfg.EPOCHS}")
print(f"  Model Directory: {model_dir}")
print(f"  Log Directory: {log_dir}")
print(f"{'='*60}\n")

# ==========================
# Environment and Agent Setup
# ==========================
env = TorchSnakeEnv(device=cfg.DEVICE)
agent = AgentClass().to(cfg.DEVICE)
optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.LR)

best_eval_score = -1

# Evaluation settings
EVAL_EPISODES = 64

# ==========================
# Episode Tracking
# ==========================
episode_lengths = torch.zeros(cfg.NUM_ENVS, device=cfg.DEVICE)
episode_length_sum = torch.tensor(0.0, device=cfg.DEVICE)
episode_length_count = torch.tensor(0, device=cfg.DEVICE)

# Pre-allocate constant tensors
ones_mask = torch.ones(cfg.NUM_ENVS, device=cfg.DEVICE)
gamma_lambda = cfg.GAMMA * cfg.GAE_LAMBDA

# ==========================
# Evaluation Function
# ==========================
def evaluate_agent(agent, num_episodes=EVAL_EPISODES):
    """Run evaluation episodes and return average score"""
    agent.eval()
    
    # Reset hidden state for recurrent agents
    if isinstance(agent, RecurrentPPOAgent):
        agent.reset_hidden(batch_size=num_episodes, device=cfg.DEVICE)
    
    # Create separate eval environment
    eval_env = TorchSnakeEnv(size=cfg.GAME_SIZE, num_envs=num_episodes, device=cfg.DEVICE)
    
    obs, rotated_obs = eval_env.reset()
    eval_done = torch.zeros(num_episodes, dtype=torch.bool, device=cfg.DEVICE)
    
    # Track scores to detect progress
    last_scores = eval_env.score.clone()
    steps_since_progress = 0
    
    with torch.no_grad():
        for eval_step in tqdm(range(cfg.ROLLOUT_STEPS), desc="Evaluating", leave=False):
            # Use greedy action selection (argmax) for evaluation
            logits, _ = agent(rotated_obs)
            actions = logits.argmax(dim=1)
            
            obs, rotated_obs, rewards, dones = eval_env.step(actions)
            eval_done |= dones
            
            # Check if any score has changed (apple eaten)
            current_scores = eval_env.score
            if (current_scores != last_scores).any():
                # Progress made, reset counter
                steps_since_progress = 0
                last_scores = current_scores.clone()
            else:
                steps_since_progress += 1
            
            # Stop early if all episodes are done
            if eval_done.all():
                break
            
            # Stop if no progress for EVAL_STEPS
            if steps_since_progress >= cfg.EVAL_STEPS:
                break
    
    avg_score = eval_env.score.mean()
    max_score = eval_env.score.max()
    min_score = eval_env.score.min()
    
    agent.train()
    
    return avg_score, max_score, min_score

# ==========================
# Training Loop
# ==========================
for epoch in range(cfg.EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{cfg.EPOCHS}")
    print(f"{'='*60}")
    
    agent.train()
    
    # Reset hidden state for recurrent agents at the start of each epoch
    if isinstance(agent, RecurrentPPOAgent):
        agent.reset_hidden(batch_size=cfg.NUM_ENVS, device=cfg.DEVICE)
    
    # Pre-allocate rollout buffers with exact size
    obs_buf = torch.zeros((cfg.ROLLOUT_STEPS, cfg.NUM_ENVS, 4, cfg.GAME_SIZE, cfg.GAME_SIZE), device=cfg.DEVICE)
    act_buf = torch.zeros((cfg.ROLLOUT_STEPS, cfg.NUM_ENVS), dtype=torch.long, device=cfg.DEVICE)
    logp_buf = torch.zeros((cfg.ROLLOUT_STEPS, cfg.NUM_ENVS), device=cfg.DEVICE)
    rew_buf = torch.zeros((cfg.ROLLOUT_STEPS, cfg.NUM_ENVS), device=cfg.DEVICE)
    done_buf = torch.zeros((cfg.ROLLOUT_STEPS, cfg.NUM_ENVS), dtype=torch.bool, device=cfg.DEVICE)
    val_buf = torch.zeros((cfg.ROLLOUT_STEPS, cfg.NUM_ENVS), device=cfg.DEVICE)

    obs, rotated_obs = env.reset()
    episode_lengths.zero_()

    # ==========================
    # Rollout Collection
    # ==========================
    actual_steps = 0
    last_scores = env.score.clone()
    steps_since_progress = 0    

    for t in tqdm(range(cfg.ROLLOUT_STEPS), desc="Gathering rollouts"):
        with torch.no_grad():
            actions, logp, values, dist = agent.sample_action(rotated_obs)

        next_obs, next_rotated_obs, rewards, dones = env.step(actions)

        # Track episode lengths
        episode_lengths += 1

        if dones.any():
            completed_lengths = episode_lengths[dones]
            episode_length_sum += completed_lengths.sum()
            episode_length_count += dones.sum()
            episode_lengths[dones] = 0
            
            # Reset hidden state for done environments in recurrent agents
            if isinstance(agent, RecurrentPPOAgent) and agent.hidden is not None:
                # Zero out hidden states for finished episodes
                done_indices = dones.nonzero(as_tuple=True)[0]
                agent.hidden[0][:, done_indices, :] = 0
                agent.hidden[1][:, done_indices, :] = 0

        # Store rollout data directly in pre-allocated buffers
        obs_buf[t] = rotated_obs
        act_buf[t] = actions
        logp_buf[t] = logp
        rew_buf[t] = rewards
        done_buf[t] = dones
        val_buf[t] = values.squeeze()

        obs = next_obs
        rotated_obs = next_rotated_obs
        actual_steps = t + 1


        # Stop rollout early if all envs are done
        if dones.all():
            print(f"Stopping rollout early at step {t+1} — all envs finished.")
            break
        
        # Check if any score has changed (apple eaten)
        current_scores = env.score
        if (current_scores != last_scores).any():
            # Progress made, reset counter
            steps_since_progress = 0
            last_scores = current_scores.clone()
        else:
            steps_since_progress += 1

        # Stop if no progress for TRAIN_MAX_STEPS_WITHOUT_PROGRESS
        if steps_since_progress >= cfg.TRAIN_MAX_STEPS_WITHOUT_PROGRESS:
            print(f"Stopping rollout early at step {t+1} — No apple eaten for {cfg.TRAIN_MAX_STEPS_WITHOUT_PROGRESS} steps")
            break

    T = actual_steps

    # Skip update if no rollout collected
    if T == 0:
        print("Skipping PPO update — no rollout steps collected.")
        continue

    # Slice buffers to actual length collected
    obs_buf = obs_buf[:T]
    act_buf = act_buf[:T]
    logp_buf = logp_buf[:T]
    rew_buf = rew_buf[:T]
    done_buf = done_buf[:T]
    val_buf = val_buf[:T]

    # ==========================
    # Bootstrap Value
    # ==========================
    with torch.no_grad():
        _, last_values = agent(rotated_obs)
        last_values = last_values.squeeze()

    # ==========================
    # GAE Advantage Calculation (Optimized)
    # ==========================
    advantages = torch.zeros_like(rew_buf)
    gae = torch.zeros(cfg.NUM_ENVS, device=cfg.DEVICE)

    mask_cache = 1.0 - done_buf.float()  # Pre-compute all masks

    for t in reversed(range(T)):
        mask = mask_cache[t]
        next_val = last_values if t == T - 1 else val_buf[t + 1]
        delta = rew_buf[t] + cfg.GAMMA * next_val * mask - val_buf[t]
        gae = delta + gamma_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + val_buf
    
    # Normalize advantages with numerical stability
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # ==========================
    # Flatten for PPO
    # ==========================
    flat_obs = obs_buf.reshape(T * cfg.NUM_ENVS, 4, cfg.GAME_SIZE, cfg.GAME_SIZE)
    flat_act = act_buf.reshape(-1)
    flat_logp = logp_buf.reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)

    total_steps = flat_obs.size(0)

    # ==========================
    # PPO Training
    # ==========================
    policy_losses = []
    value_losses = []
    entropy_losses = []
    total_losses = []

    for _ in tqdm(range(cfg.PPO_EPOCHS), desc="PPO training"):
        idx = torch.randperm(total_steps, device=cfg.DEVICE)

        for start in range(0, total_steps, cfg.MINIBATCH_SIZE):
            end = min(start + cfg.MINIBATCH_SIZE, total_steps)
            mb = idx[start:end]

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
            entropy_loss = dist.entropy().mean()

            loss = policy_loss + cfg.VALUE_COEF * value_loss - cfg.ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()

            # Store detached values to avoid memory accumulation
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_losses.append(entropy_loss.detach())
            total_losses.append(loss.detach())

    cfg.ENTROPY_COEF *= 0.995  # Decay entropy coefficient

    # ==========================
    # Training Metrics & Logging
    # ==========================
    with torch.no_grad():
        train_avg_score = env.score.mean()
        avg_reward = rew_buf.mean()
        avg_advantage = adv_mean
        
        avg_policy_loss = torch.stack(policy_losses).mean()
        avg_value_loss = torch.stack(value_losses).mean()
        avg_entropy = torch.stack(entropy_losses).mean()
        avg_total_loss = torch.stack(total_losses).mean()

        if episode_length_count > 0:
            avg_episode_length = episode_length_sum / episode_length_count
            episode_length_sum.zero_()
            episode_length_count.zero_()
        else:
            avg_episode_length = torch.tensor(0.0, device=cfg.DEVICE)
        
        # Shared comparison plots
        writer.add_scalars("Performance/avg_reward", {
        agent_name: avg_reward.item()
        }, epoch
        )
        writer.add_scalars("Performance/avg_advantage", {
        agent_name: avg_advantage.item()
        }, epoch
        )
        writer.add_scalars("Loss/total_loss", {
        agent_name: avg_total_loss.item()
        }, epoch
        )
        writer.add_scalars("Loss/entropy_loss", {
        agent_name: avg_entropy.item()
        }, epoch
        )
        writer.add_scalars("Loss/value_loss", {
        agent_name: avg_value_loss.item()
        }, epoch
        )
        writer.add_scalars("Loss/policy_loss", {
        agent_name: avg_policy_loss.item()
        }, epoch
        )
    # ==========================
    # Evaluation
    # ==========================
    eval_avg_score, eval_max_score, eval_min_score = evaluate_agent(agent, num_episodes=EVAL_EPISODES)
    
    # Shared comparison plot for eval scores
    writer.add_scalars("Score/avg_eval_score", {
        agent_name: eval_avg_score.item()
    }, epoch
    )
    writer.add_scalars("Score/avg_train_score", {
        agent_name: train_avg_score.item()
    }, epoch
    )
    writer.add_scalars("Score/max_eval_score", {
        agent_name: eval_max_score.item()
    }, epoch
    )
    writer.add_scalars("Score/min_eval_score", {
        agent_name: eval_min_score.item()
    }, epoch
    )
    
    # Save best model based on evaluation score
    if eval_avg_score > best_eval_score:
        best_eval_score = eval_avg_score.item()
        model_path = os.path.join(model_dir, "best_model.pt")
        torch.save(agent.state_dict(), model_path)
        print(f"\033[92mNew best model saved! Eval Score: {best_eval_score:.2f}\033[0m")


writer.close()