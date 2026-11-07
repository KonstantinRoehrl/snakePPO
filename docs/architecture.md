# Architecture

snakePPO is a research bench for Proximal Policy Optimization (PPO) on Snake. Every
improvement over plain PPO is a config switch, and the code defaults are the vanilla
baseline, so any run can be compared against `configs/baseline.yaml` by changing one value.

## Module map

All modules live under `src/snakeppo/`.

| Module | Responsibility |
| --- | --- |
| `config` | Typed dataclass config (`env`, `reward`, `agent`, `ppo`, `train`), YAML loading, `--set` overrides and `validate`. |
| `builders` | Turns config sections into live objects: `resolve_device`, `seed_everything`, `make_env`, `EVAL_SEED_OFFSET`. |
| `env` | `TorchSnakeEnv`: a vectorized, auto-resetting Snake that runs entirely in torch. |
| `agents/networks` | `PolicyNetwork` base class and the `CNNPolicy`, `SpatialCNNPolicy` and `ResNetPolicy` networks. |
| `agents/registry` | `AgentSpec`, `register`, `build_agent` and `available_agents`: every network is built by name from here. |
| `ppo/functional` | Pure PPO math: `compute_gae`, `ppo_loss`, `lr_at`, `entropy_coef_at`, `explained_variance`. |
| `ppo/trainer` | `RolloutBuffer` and `train`: the rollout, update, logging and checkpoint loop. |
| `evaluation` | `evaluate` and `EvalStats`: score a policy over a fixed number of complete episodes. |
| `runs` | Run folders: `create_run_dir`, `run.json`, `MetricsLogger`, checkpoints, `find_runs`, `summarize_run`. |
| `viewer` | Pygame window showing up to four games side by side (`snakeppo eval --render`). |
| `cli` | The `snakeppo` command: `train`, `eval`, `agents`, `compare`. |

## Data flow of one update

1. **Config.** `load_config` merges defaults, the `--config` YAML file and `--set` overrides,
   then `validate` rejects meaningless values.
2. **Builders.** `train` resolves the device, calls `seed_everything(env.seed,
   train.deterministic)`, builds the training env with `make_env` and the model with
   `build_agent(agent.name, env.board_size, **agent.kwargs)`, and creates the run folder.
3. **Rollout.** For `ppo.rollout_steps` steps the model acts in all `env.num_envs` envs
   without gradients; observations, actions, log-probs, values, rewards, `terminated` and
   `done` go into a preallocated `(T, N)` `RolloutBuffer` (observations stored as `uint8`).
4. **GAE.** `compute_gae` turns the rollout into advantages and returns. A death
   (`terminated`) stops bootstrapping: it has no future value. A starvation cut
   (`truncated`) still bootstraps from the value of the pre-reset `info["final_obs"]`.
   Both end the advantage trace at the episode boundary.
5. **Minibatch updates.** The flattened batch is shuffled and split into
   `ppo.minibatch_size` minibatches for `ppo.update_epochs` epochs; each minibatch takes one
   Adam step on `ppo_loss`, with gradients clipped to `ppo.max_grad_norm`. With
   `ppo.target_kl` set, the update stops early once a minibatch's approximate KL exceeds it.
6. **Logging.** `MetricsLogger` appends one line to `metrics.jsonl` and mirrors every metric
   to TensorBoard with env steps on the x-axis.
7. **Evaluation and checkpoints.** Every `train.eval_interval` updates and on the final
   update, `evaluate` plays `train.eval_episodes` greedy games on a separate env seeded
   with `env.seed + EVAL_SEED_OFFSET` (10000). The trainer then writes
   `checkpoints/last.pt`, and `checkpoints/best.pt` when the eval mean score improves.

## Environment contract

`TorchSnakeEnv` (`src/snakeppo/env.py`) holds `num_envs` boards of size `board_size`, with
a wall ring on the outer cells.

- **Actions.** Three relative actions (`NUM_ACTIONS = 3`): 0 straight, 1 turn left,
  2 turn right.
- **Observation.** `(N, 4, S, S)` float32 with `NUM_CHANNELS = 4` channels in this order:
  body, head, food, wall ring. `observe` rotates the grid so every snake faces up
  (egocentric); `render_grid` returns the unrotated grid for display.
- **`step` returns** `(obs, reward, terminated, truncated, info)`.
  - `terminated`: the snake died by hitting a wall or its own body.
  - `truncated`: the snake survived but went `starvation_steps` steps without eating.
- **Auto-reset.** Every env that finished is reset inside the same `step` call, so `obs`
  is already the first observation of the next episode.
- **`info` keys.** `done` (`terminated | truncated`), `final_obs` (the pre-reset
  observation; equal to `obs` where not done), and `episode_score`, `episode_length` and
  `episode_return` (valid where `done`).

## Invariants

- **Switches default off.** Every learning-altering option defaults to the plain PPO
  baseline; turning one on is an explicit config change.
- **No import-time side effects.** Importing a module never trains, opens a window or
  writes files. The CLI imports the trainer and the pygame viewer only inside the
  command that needs them.
- **Versioned run format.** `run.json` carries `schema_version` (currently 1) for the
  run-folder layout, the `metrics.jsonl` keys and the checkpoint contents described in
  [experiments.md](experiments.md).
- **`runs/` is never committed.** It is listed in `.gitignore`.
