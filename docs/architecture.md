# Architecture

snakePPO is a research bench for Proximal Policy Optimization (PPO) on Snake. Every
improvement over plain PPO is a config switch, and the code defaults are the vanilla
baseline, so any run can be compared against `configs/baseline.yaml` by changing one value.

## Module map

All modules live under `src/snakeppo/`.

| Module | Responsibility |
| --- | --- |
| `config` | Typed dataclass config (`env`, `reward`, `agent`, `ppo`, `train`, `curriculum`, `observation`), YAML loading, `--set` overrides, `validate`, the curriculum's stage ladder (`CurriculumStage`, `curriculum_ladder`), and the preset listing and run descriptions (`Preset`, `list_presets`, `run_description`, `describe_changes`). |
| `builders` | Turns config sections into live objects: `resolve_device`, `seed_everything`, `make_env` (the training env, with its reward strategy), `make_play_env` (the full-board env for eval and Replay), both with the run's observation, `plays_any_size` and `play_env_config` (the board a run plays on in eval and Replay, refusing another size for a board-bound agent with the full view), `observation_record`, `make_policy` (a new run's network, sized from its observation) and `load_policy` (a checkpoint's network, refused when its observation differs), `recurrent_minibatches`, `EVAL_SEED_OFFSET`, and `capture_rng_state`/`restore_rng_state` for checkpoints. |
| `observation` | `ObservationSpec` (the observation switches, built from `ObservationConfig`: channels, size, `is_binary`, the `record` a run stores), `describe_record`, and the pure tensor helpers `body_order_grid`, `tail_grid` and `local_view`. |
| `env` | `TorchSnakeEnv`: a vectorized, auto-resetting Snake that runs entirely in torch; it builds the observation its `ObservationSpec` describes, and `render_grid`/`render_body_order` give the unrotated board for display; `set_curriculum` plays it in a smaller walled arena or up to a goal length; `state_dict`/`load_state_dict` save and restore every board. |
| `rewards` | `StepContext`, the reward strategies (`ClassicReward`, `SparseReward`, `DistanceReward`, `PotentialReward`, `HungerReward`, `FreeSpaceReward`), `RewardSpec`, `register`, `build_reward` and `available_rewards`: every per-step reward is built by name from here. |
| `curriculum` | `Curriculum`: which stage a run trains in, the promotion gate, and its `state_dict`/`load_state_dict` for checkpoints. |
| `agents/networks` | `PolicyNetwork` base class, `DeterministicAdaptiveAvgPool2d`, the shared `conv_features` stack, and the feed-forward networks `CNNPolicy`, `SpatialCNNPolicy`, `ResNetPolicy`, `MLPPolicy`, `ScalableCNNPolicy` and `AttentionPolicy` (with `sinusoidal_positions_2d`). |
| `agents/recurrent` | `RecurrentPolicy` (a policy with a hidden state), `RecurrentCNNPolicy` (GRU or LSTM), `episode_starts`, and `PolicyPlayer`, which plays games with any policy and carries a recurrent one's state. |
| `agents/registry` | `AgentSpec` (with `requires_even_board` and `size_agnostic`), `register`, `build_agent` and `available_agents`: every network is built by name from here. |
| `ppo/functional` | Pure PPO math: `compute_gae`, `ppo_loss`, `lr_at`, `entropy_coef_at`, `explained_variance`. |
| `ppo/trainer` | `RolloutBuffer`, `train` and `resume` (raising `ResumeError` for a run it cannot continue exactly): the rollout, update, logging and checkpoint loop, for feed-forward and recurrent policies. At the end of every update it checks the run's stop request and then ends as Ctrl+C does (`StopRequested`). |
| `evaluation` | `evaluate` and `EvalStats`: score a policy over a fixed number of complete episodes, with how they ended and steps per apple. Eval games see the run's observation, ignore the reward strategy and the curriculum, and play through `PolicyPlayer`. |
| `processes` | Process records and stop requests, shared by `runs`, the trainer, the sweep and the dashboard: `process_record` (pid, creation time from `psutil`, host), `is_alive` (pid and creation time, so a recycled pid never counts), `request_stop`, `stop_requested` and `clear_stop_request` (the `stop-request` file), `launch_dir` (`runs/.launches/<id>`), `spawn_detached` (a child that outlives the server), `child_creationflags` and `terminate` (the whole process tree, then a kill). |
| `runs` | Run folders (schema 5): `create_run_dir`, `run.json` (with the training process's record), `MetricsLogger`, `read_metrics_since`, checkpoints, `replace_atomically`, `load_run`/`RunRecord` (with the derived status: `stalled` at once when a `running` run's process is gone), `find_runs`, `summarize_run`. |
| `results` | The multi-seed comparison protocol: run scores, switches, seed groups, references and bootstrap verdicts (`results_table`, `format_results`), with each group's description. |
| `sweep` | `run_sweep`: one `snakeppo train` child process per seed, all in one seed group. Joining a group (`--group`) leaves its finished, failed and running seeds alone, resumes its interrupted and stalled ones (`resume_command`) and starts the missing ones; after a launch's stop request it starts no further seeds and reports them as skipped (`SeedOutcome`). |
| `dashboard/api` | Turns `RunRecord`s and results into the dashboard's JSON payloads, with each run's description, liveness (`alive`, `pid`), `launch`, `stop_requested` and whether it plays any board size. |
| `dashboard/launches` | Browser launches: `parse_launch_request`, `launch_argv` (exactly the `snakeppo train` or `sweep` command the CLI runs, validated before anything starts), `start_launch`, `resume_run`, `resume_sweep`, `list_launches`, and `stop_run` and `stop_sweep` (a stop request, then `processes.terminate` after `STOP_GRACE_SECONDS`), with each launch's `launch.json` and `output.log` under `runs/.launches/<id>/`. |
| `dashboard/replay` | `ReplayManager` and `ReplaySession`: at most one replay session, stepping up to four single-game play envs (`BoardSpec`, `parse_board_specs`) in lockstep on the CPU with one torch thread at a speed from `REPLAY_SPEEDS`, and handing its frames to the SSE stream; a session closes after `IDLE_SECONDS` with no stream connected. |
| `dashboard/server` | The stdlib HTTP server behind `snakeppo serve`, bound to 127.0.0.1: the read routes, the guarded write routes and the replay stream (see Dashboard routes). |
| `dashboard/static` | The dashboard frontend: plain ES modules with no build step, charts drawn with the bundled plotly.js, run colours from `colors.js` (a stable hash of the run id), the Launch tab in `launch.js` and the Replay tab in `replay.js`, which draws boards with `board.js`. Its visual system is in [DESIGN.md](../DESIGN.md). |
| `cli` | The `snakeppo` command: `train`, `eval` (`--render` opens the dashboard's Replay), `agents`, `rewards`, `compare`, `sweep`, `results`, `serve`. |

## Data flow of one update

1. **Config.** `load_config` merges defaults, the `--config` YAML file and `--set` overrides,
   then `validate` rejects meaningless values.
2. **Builders.** `train` resolves the device, calls `seed_everything(env.seed,
   train.deterministic)`, builds the training env with `make_env` and the model with
   `make_policy` (`build_agent(agent.name, <observation side>, in_channels=<channel count>,
   **agent.kwargs)`), and only then creates the run folder, so a config whose network cannot
   be built leaves no folder.
3. **Rollout.** For `ppo.rollout_steps` steps the model acts in all `env.num_envs` envs
   without gradients; observations, actions, log-probs, values, rewards, `terminated` and
   `done` go into a preallocated `(T, N)` `RolloutBuffer` (observations stored as `uint8`
   when every value is 0 or 1, the default observation, and as `float32` otherwise). A
   recurrent policy also carries its hidden state from step to step, zeroed where an episode
   ends.
4. **GAE.** `compute_gae` turns the rollout into advantages and returns. A death
   (`terminated`) stops bootstrapping: it has no future value. A starvation or goal cut
   (`truncated`) still bootstraps from the value of the pre-reset `info["final_obs"]`.
   Both end the advantage trace at the episode boundary.
5. **Minibatch updates.** The flattened batch is shuffled and split into
   `ppo.minibatch_size` minibatches for `ppo.update_epochs` epochs; each minibatch takes one
   Adam step on `ppo_loss`, with gradients clipped to `ppo.max_grad_norm`. With
   `ppo.target_kl` set, the update stops early once a minibatch's approximate KL exceeds it.
   A recurrent policy's minibatches are groups of whole envs instead (`recurrent_minibatches`
   groups), each replaying its envs' rollout from the rollout's start state.
   A curriculum run then records the update's finished episodes and may promote: the env
   gets the next stage's arena and goal and every env is reset (a recurrent state too),
   before eval and checkpoints.
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
a wall ring on the outer cells. `set_curriculum` shrinks the playable area to a walled arena
centred on the board and can end episodes at a goal length.

- **Actions.** Three relative actions (`NUM_ACTIONS = 3`): 0 straight, 1 turn left,
  2 turn right.
- **Observation.** `(N, *obs_shape)` float32, built by `observe` from the env's
  `ObservationSpec` (`observation`). `obs_channels` names the channels, always in this order:
  the body (`body`, or `body_order` with `observation.body: order`), `head`, `food`, `wall`
  (the arena's border and everything outside it), then `tail` and `hunger` when on.
  `obs_shape` is `(len(obs_channels), side, side)`, where `side` is `board_size` with the full
  view and `observation.view_size` with the local view. `obs_is_binary` is true when every
  value is 0 or 1 (only the default observation), which lets the rollout buffer store `uint8`.
  `observe` rotates the observation so every snake faces up (egocentric). The local view crops
  the head-centred square before rotating, which equals cropping after rotating because the
  head is the crop's centre cell.
- **Display grids.** `render_grid` returns the unrotated `(N, NUM_CHANNELS, S, S)` grid
  (`NUM_CHANNELS = 4`: body, head, food, walls) whatever the observation, and
  `render_body_order` the unrotated `(N, S, S)` body order (1.0 at the head down to 0.1 at the
  tail). A replay frame carries the `render_grid` cells as one digit each and the body as
  cell indices head first; the browser shades the body with the same 1.0-to-0.1 order
  (`bodyShade` in `static/board.js`).
- **`step` returns** `(obs, reward, terminated, truncated, info)`.
  - `terminated`: the snake died by hitting a wall or its own body.
  - `truncated`: the snake survived but starved (`starvation_steps` steps without eating)
    or reached the goal length.
  - `reward`: computed by the env's reward strategy (`rewards`).
- **Auto-reset.** Every env that finished is reset inside the same `step` call, so `obs`
  is already the first observation of the next episode.
- **`info` keys.** `done` (`terminated | truncated`), `final_obs` (the pre-reset
  observation; equal to `obs` where not done), and `episode_score`, `episode_length` and
  `episode_return` and `end_reason` (an `END_*` code: `END_WALL`, `END_BODY`,
  `END_STARVED`, `END_GOAL`; valid where `done`).

## Dashboard routes

`snakeppo.dashboard.server` answers these routes. Run ids are folder names of direct children
of the runs folder that hold a `run.json`; launch ids are folder names under `runs/.launches/`.

| Route | Answer |
| --- | --- |
| GET `/`, `/static/<file>` | The frontend. |
| GET `/vendor/plotly.min.js` | The plotly.js bundled in the `plotly` Python package. |
| GET `/api/runs` | `{"runs": [summary], "skipped": [{"id", "error"}], "runs_dir", "runs_dir_resolved"}`. |
| GET `/api/runs/<id>` | `{"run": summary, "config", "info"}`. |
| GET `/api/runs/<id>/metrics?since=N` | `{"lines", "offset", "reset"}`: the metrics lines after byte `N`. |
| GET `/api/results[?baseline=G]` | The `snakeppo results` table. |
| GET `/api/diff?base=<id>&runs=<id>,<id>` | Each run's learning-config differences from the base run. |
| GET `/api/presets` | `{"presets": [{"path", "name", "description"}], "bookkeeping_keys", "configs_dir"}`. |
| GET `/api/launches` | `{"launches": [...]}`: every launch record, newest first, with its state and the tail of its `output.log`. |
| POST `/api/launch` | Starts a train or sweep launch: 202 `{"launch"}`. |
| POST `/api/runs/<id>/stop` | Asks a running run to stop: 202 `{"ok": true}`. |
| POST `/api/runs/<id>/resume` | Resumes an interrupted, failed or stalled run: 202 `{"launch"}`. |
| POST `/api/launches/<id>/stop` | Stops a sweep launch and its running seeds: 202 `{"ok": true}`. |
| POST `/api/launches/<id>/resume` | Continues a stopped sweep as the same seed group: 202 `{"launch"}`. |
| POST `/api/replay` | Creates the replay session from `{"boards": [...], "speed"}`, ending any other: 201 `{"session"}`. |
| POST `/api/replay/<id>/control` | `{"action", "value"}` with the action `pause`, `resume`, `step`, `speed` or `restart`: 200 `{"paused", "speed"}`. |
| GET `/api/replay/<id>/stream` | Server-Sent Events: one `data:` frame per step for all boards, a `: keepalive` comment after 15 s without one, and `event: end` when the session closes. |

Errors are `{"error": "<text>"}`: 400 for a bad request or an invalid config, 403 for a failed
origin check, 404 for an unknown run, launch or replay id, 405 for any other method, 409 when
the run or launch is not in a state that allows the action, 413 for a body over 64 KiB and 500 for an unexpected server error.

## Invariants

- **Switches default off.** Every learning-altering option defaults to the plain PPO
  baseline; turning one on is an explicit config change.
- **No import-time side effects.** Importing a module never trains, opens a browser,
  writes files, looks up the hostname, starts a thread or sets torch's thread count. The
  CLI imports the trainer only inside the commands that need it.
- **Versioned run format.** `run.json` carries `schema_version` (currently 5) for the
  run-folder layout, the `metrics.jsonl` keys, the checkpoint contents and the launch
  records described in [experiments.md](experiments.md). Readers still accept schema-1 to
  schema-4 folders.
- **One observation per run.** A run's network is built from its observation and its
  checkpoints record that observation; `load_policy` refuses to load a checkpoint against a
  config describing another one.
- **Atomic writes.** `run.json`, `launch.json`, the metrics rewrite on resume and the
  checkpoints are written to a temporary file and renamed into place (`replace_atomically`).
- **The dashboard never changes how a run learns.** It starts and resumes runs only by running
  the command a terminal would (`sys.executable -m snakeppo.cli train|sweep …` as an argument
  list, detached), and stops them by writing their `stop-request`; its own files are the launch
  records under `runs/.launches/`. It never trains in its own process and never writes
  `run.json`, `metrics.jsonl` or a checkpoint.
- **Dashboard writes are guarded.** The server binds to 127.0.0.1 only. Every POST must send
  `Content-Type: application/json` and an `Origin` (or, without one, a `Referer`) of
  `http://127.0.0.1:<port>` or `http://localhost:<port>` for the server's own port, or it gets
  403 before any work; a body over 64 KiB (`MAX_BODY_BYTES`) gets 413, and run and launch ids
  must name a direct child of their folder.
- **Replay leaves training its resources.** Replay plays on the CPU with one torch thread, and
  at most one replay session runs at a time.
- **`runs/` is never committed.** It is listed in `.gitignore`.
