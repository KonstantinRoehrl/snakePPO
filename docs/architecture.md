# Architecture

snakePPO is a research bench for Proximal Policy Optimization (PPO) on Snake. Every
improvement over plain PPO is a config switch, and the code defaults are the vanilla
baseline, so any run can be compared against `configs/baseline.yaml` by changing one value.

## Module map

All modules live under `src/snakeppo/`.

| Module | Responsibility |
| --- | --- |
| `config` | Typed dataclass config (`env`, `reward`, `agent`, `ppo`, `train`), YAML loading, `--set` overrides and `validate`. |
| `builders` | Turns config sections into live objects: `resolve_device`, `seed_everything`, `make_env`, `EVAL_SEED_OFFSET`, and `capture_rng_state`/`restore_rng_state` for checkpoints. |
| `env` | `TorchSnakeEnv`: a vectorized, auto-resetting Snake that runs entirely in torch; `state_dict`/`load_state_dict` save and restore every board. |
| `agents/networks` | `PolicyNetwork` base class and the `CNNPolicy`, `SpatialCNNPolicy` and `ResNetPolicy` networks. |
| `agents/registry` | `AgentSpec`, `register`, `build_agent` and `available_agents`: every network is built by name from here. |
| `ppo/functional` | Pure PPO math: `compute_gae`, `ppo_loss`, `lr_at`, `entropy_coef_at`, `explained_variance`. |
| `ppo/trainer` | `RolloutBuffer`, `train` and `resume` (raising `ResumeError` for a run it cannot continue exactly): the rollout, update, logging and checkpoint loop. |
| `evaluation` | `evaluate` and `EvalStats`: score a policy over a fixed number of complete episodes. |
| `runs` | Run folders (schema 2): `create_run_dir`, `run.json`, `MetricsLogger`, `read_metrics_since`, checkpoints, `replace_atomically`, `load_run`/`RunRecord` (with the derived status), `find_runs`, `summarize_run`. |
| `results` | The multi-seed comparison protocol: run scores, switches, seed groups, references and bootstrap verdicts (`results_table`, `format_results`). |
| `sweep` | `run_sweep`: one `snakeppo train` child process per seed, all in one seed group. |
| `viewer` | Pygame window showing up to four games side by side (`snakeppo eval --render`). |
| `dashboard/api` | Turns `RunRecord`s and results into the dashboard's JSON payloads. |
| `dashboard/server` | The read-only stdlib HTTP server behind `snakeppo serve`, bound to 127.0.0.1. |
| `dashboard/static` | The dashboard frontend: plain ES modules with no build step, charts drawn with the bundled plotly.js. Its visual system is in [DESIGN.md](../DESIGN.md). |
| `cli` | The `snakeppo` command: `train`, `eval`, `agents`, `compare`, `sweep`, `results`, `serve`. |

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
   with `env.seed + EVAL_SEED_OFFSET` (10000). The trainer writes the update's metrics line
   first. Then, on eval updates, the final update and every `train.checkpoint_interval`
   updates, it writes `checkpoints/last.pt` with the RNG and env state, and
   `checkpoints/best.pt` when the eval mean score improves. When the loop ends, `run.json`'s
   `status` records how: `finished`, `interrupted` or `failed`.

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
- **Versioned run format.** `run.json` carries `schema_version` (currently 2) for the
  run-folder layout, the `metrics.jsonl` keys and the checkpoint contents described in
  [experiments.md](experiments.md). Readers still accept schema-1 folders.
- **Atomic writes.** `run.json`, the metrics rewrite on resume and the checkpoints are written
  to a temporary file and renamed into place (`replace_atomically`).
- **The dashboard is read-only.** Its server answers GET requests only and never writes to a
  run folder.
- **`runs/` is never committed.** It is listed in `.gitignore`.
