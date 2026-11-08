# Running experiments

## Install

Install [uv](https://docs.astral.sh/uv/), then run this from the repository root:

```sh
uv sync
```

On Windows and Linux this installs the CUDA build of torch from the PyTorch cu130 index; on
macOS it installs torch from PyPI. On a machine without an NVIDIA GPU, install the CPU-only
build instead, as CI does:

```sh
uv sync --no-group cuda --group cpu
```

## Presets and config load order

| File | Purpose |
| --- | --- |
| `configs/baseline.yaml` | Vanilla PPO: every value is the code default, stated explicitly. Copy it to start an experiment. |
| `configs/tuned.yaml` | Baseline with every PPO switch turned on. |
| `configs/smoke.yaml` | Tiny CPU run (4 updates) used by the tests and CI. |
| `configs/rewards/<strategy>.yaml` | Baseline with one reward strategy turned on: sparse, distance, potential, hunger or free_space. |
| `configs/curriculum/<grow>.yaml` | Baseline with one curriculum: board, length or both. |

```sh
uv run snakeppo train --config configs/baseline.yaml
uv run snakeppo train --config configs/tuned.yaml --set ppo.target_kl=null --seed 1
```

Values are applied in this order, each step overriding the previous one:

1. the code defaults in `src/snakeppo/config.py`;
2. the `--config` YAML file (a partial file is fine);
3. each `--set SECTION.KEY=VALUE`, in order (the value is parsed as YAML, so `null`,
   `true` and numbers work). The `--set` values are validated together after the last one, so
   their order does not matter;
4. `--agent`, `--name`, `--seed` and `--group`, which set `agent.name`, `train.name`,
   `env.seed` and `train.group` (`--agent`, `--name` and `--group` are plain text, so
   `--name 2` and `--name off` work).

Unknown sections or keys and values of the wrong type are rejected with an error.

YAML reads a bare `off` as false, which a text key rejects: write `grow: 'off'` in a file, or
leave the key out (`off` is the default).

## Switches

Each switch below defaults to the vanilla baseline.

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `ppo.lr_schedule` | `constant`, `linear` | `constant` | `linear` decays `ppo.lr` to 0 over the run. |
| `ppo.clip_value_loss` | `true`, `false` | `false` | Clips the value update to `ppo.clip_eps` around the rollout's values. |
| `ppo.entropy_schedule` | `constant`, `linear`, `exponential` | `constant` | Moves the entropy coefficient from `ppo.entropy_coef` to `ppo.entropy_coef_final` over the run. |
| `ppo.entropy_coef_final` | float (> 0 for `exponential`) | `0.0005` | End value of the entropy schedule; unused with `constant`. `exponential` also needs `ppo.entropy_coef` > 0. |
| `ppo.target_kl` | float or `null` | `null` | Stops an update's epochs early once a minibatch's approximate KL exceeds it. |
| `train.deterministic` | `true`, `false` | `false` | Forces deterministic torch kernels so GPU runs repeat exactly (slower). |
| `reward.strategy` | `classic`, `sparse`, `distance`, `potential`, `hunger`, `free_space` | `classic` | Selects the per-step reward; see Reward strategies. |
| `reward.distance_scale` | float ≥ 0 | `0.1` | Read only by `distance`. |
| `reward.potential_scale` | float ≥ 0 | `0.1` | Read only by `potential`. |
| `reward.timeout_penalty` | float ≥ 0 | `1.0` | Read only by `hunger`. |
| `reward.hunger_penalty` | float ≥ 0 | `0.01` | Read only by `hunger`. |
| `reward.free_space_penalty` | float ≥ 0 | `0.1` | Read only by `free_space`. |
| `curriculum.grow` | `off`, `board`, `length`, `both` | `off` | Trains on a ladder of easier stages first; see Curriculum. |
| `curriculum.stages` | int ≥ 2 | `5` | Number of stages, the final full-board stage included. Read only when `curriculum.grow` is on. |
| `curriculum.start_board` | int, 5 to `env.board_size` | `8` | Arena side (walls included) of the first stage. Read only when `curriculum.grow` is on. |
| `curriculum.start_goal` | int ≥ 2 | `4` | Goal length of the first stage. Read only when `curriculum.grow` is on. |
| `curriculum.promote_fraction` | float in (0, 1] | `0.3` | Share of the stage's capacity the promotion window must average. Read only when `curriculum.grow` is on. |
| `curriculum.promote_window` | int ≥ 1 | `64` | Finished in-stage episodes the promotion gate averages. Read only when `curriculum.grow` is on. |
| `curriculum.min_updates` | int ≥ 1 | `10` | Updates a stage runs before the gate may promote it. Read only when `curriculum.grow` is on. |
| `curriculum.max_updates` | int ≥ `curriculum.min_updates` | `100` | Updates after which a stage is promoted anyway. Read only when `curriculum.grow` is on. |

### Reward strategies

In every strategy, `reward.apple` (`1.0`) and `reward.death` (`-3.0`) replace the step's reward
when the snake eats or dies. `d` is the Manhattan distance from the head to the food.

| Strategy | Reward of a step |
| --- | --- |
| `classic` | The baseline: `reward.approach` (`0.02`) per unit the squared distance to the food shrank, minus `reward.retreat` (`0.03`) per unit it grew. |
| `sparse` | `0`: apples and deaths only. |
| `distance` | `reward.distance_scale · (d_before − d_after)`: plus the scale per cell moved toward the food, minus it per cell away. |
| `potential` | `sparse` plus `γ · φ(s') − φ(s)` on every step, eating and dying included, with `φ = −reward.potential_scale · d` and `φ = 0` in a terminal state, so the optimal policy is unchanged. |
| `hunger` | `classic`, minus `reward.hunger_penalty · steps since food / env.starvation_steps`, minus `reward.timeout_penalty` on the step the snake starves. |
| `free_space` | `classic`, minus `reward.free_space_penalty` times the share of free playable cells the head can no longer reach. |

The added `hunger` and `free_space` terms apply only to steps that neither eat nor die.
`potential` takes γ from `ppo.gamma`. `uv run snakeppo rewards` lists the strategies.

### Curriculum

With `curriculum.grow` on, a run trains through a ladder of stages whose last is always the
full board with no goal. `board` widens a walled arena linearly from `curriculum.start_board`
to `env.board_size` (rounded half up), `length` doubles a goal length from
`curriculum.start_goal`, and `both` does both on one ladder. With the defaults on a 16 board,
`both` trains on arenas 8, 10, 12, 14, 16 with goals 4, 8, 16, 32, none.

- **Arena.** A walled square centred in the full-size observation tensor; the wall channel
  marks its border and everything outside it.
- **Goal.** Reaching the goal length ends the episode as a truncation, with `end_reason` goal.
- **Capacity.** The most apples one episode of a stage can eat: `min((arena − 2)², goal − 1)`,
  or `(arena − 2)²` with no goal.
- **Gate.** A stage is promoted once its last `curriculum.promote_window` finished in-stage
  episodes average at least `curriculum.promote_fraction · capacity`, no earlier than
  `curriculum.min_updates` updates in the stage, and is forced at `curriculum.max_updates`.
  The final stage is never left.
- **Timing.** Promotion happens between updates, before that update's eval and checkpoint, and
  resets every training env into the new stage.
- **Budget.** `validate` requires `(stages − 1) · max_updates` to be below the run's total
  updates, so training always ends on the full board.

## Bookkeeping keys

These keys never change what a run learns and never count as switches.

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `train.checkpoint_interval` | int ≥ 1 | `10` | Writes `last.pt` every N updates, plus on every eval update and the final update. |
| `train.group` | text or `null` | `null` | The seed group the run belongs to; set by `snakeppo sweep` or `train --group`. |

## Seeding and reproducibility

One value, `env.seed` (default 42, or `--seed`), seeds python, numpy, torch and the
training environments. Two runs with the same config and seed produce bitwise-identical
curves on CPU, or on GPU with `train.deterministic: true`; otherwise GPU kernels may make
them drift. Evaluation always uses a separate environment seeded with
`env.seed + 10000` (`EVAL_SEED_OFFSET`), so every eval plays the same games.

## Run folders

Each `snakeppo train` writes one folder under `train.runs_dir` (default `runs/`):

```text
runs/<YYYY-MM-DD_HHMMSS>_<train.name>_seed<env.seed>/
    run.json        run identity (below)
    config.yaml     the fully resolved config
    metrics.jsonl   one JSON object per update
    tb/             TensorBoard event files
    checkpoints/
        best.pt     checkpoint with the best eval mean score so far
        last.pt     the most recent checkpoint (every `train.checkpoint_interval` updates,
                    every eval update and the final update)
```

A `_2`, `_3`, ... suffix is added if the folder name already exists.

`run.json` fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Run-folder format version, currently `3`. Covers this layout, the metrics keys and the checkpoint contents. |
| `name` | `train.name`. |
| `created` | Local timestamp with UTC offset, to the second. |
| `git_sha` | Commit the run was trained from, or `null` outside a git checkout. |
| `agent` | `agent.name`. |
| `param_count` | Number of model parameters. |
| `device` | Device the run trained on, for example `cuda` or `cpu`. |
| `status` | `running` while training, then `finished`, `interrupted` (Ctrl+C) or `failed` (any other error). Readers also report `stalled` (below) and, for schema-1 runs, `unknown`. |
| `total_updates` | Number of PPO updates the run loops to. |
| `group` | `train.group`: the seed group, or `null`. |
| `resumed` | One local timestamp per resume, oldest first. |

`run.json` and the checkpoints are written atomically (a temporary file, then a rename), and
each update's metrics line is written before its checkpoint, so every checkpoint's update has
its line in `metrics.jsonl`.

A run whose stored `status` is `running` is reported as `stalled` once nothing has been
written to `metrics.jsonl` for longer than the stall window, `max(60 s, 5 × median update gap
of the last 10 updates)`. That usually means the training process is gone.

Schema-1 folders stay readable. They store no status: it is inferred as `finished` when the
last update equals the run's total, otherwise `unknown`. They have no group. Schema-2 and
schema-3 runs are resumable.

### `metrics.jsonl`

One line per PPO update. Every key except `update`, `env_steps` and `wall_time` is also
written to TensorBoard under the same name.

| Key | Meaning | Present |
| --- | --- | --- |
| `update` | Update number, starting at 1. | always |
| `env_steps` | Total environment steps so far. | always |
| `wall_time` | Seconds since training started. | always |
| `loss/total` | Mean total PPO loss over the update's minibatches. | always |
| `loss/policy` | Mean clipped-surrogate policy loss. | always |
| `loss/value` | Mean value loss. | always |
| `loss/entropy` | Mean policy entropy. | always |
| `ppo/approx_kl` | Mean approximate KL between the rollout and updated policy. | always |
| `ppo/clip_fraction` | Mean fraction of samples whose probability ratio was clipped. | always |
| `ppo/explained_variance` | How well the rollout's values predicted the returns (1 is perfect). | always |
| `ppo/lr` | Learning rate used this update. | always |
| `ppo/entropy_coef` | Entropy coefficient used this update. | always |
| `ppo/grad_norm` | Mean gradient norm before clipping. | always |
| `ppo/epochs` | Epochs actually run (fewer than `ppo.update_epochs` when `ppo.target_kl` stopped early). | always |
| `train/fps` | Environment steps per second over the update. | always |
| `train/reward_mean` | Mean per-step reward in the rollout. | always |
| `train/episodes` | Episodes that finished during the rollout. | when at least one finished |
| `train/score_mean` | Mean apples eaten by those episodes. | when at least one finished |
| `train/score_max` | Best score among those episodes. | when at least one finished |
| `train/length_mean` | Mean length in steps of those episodes. | when at least one finished |
| `train/return_mean` | Mean undiscounted return of those episodes. | when at least one finished |
| `eval/score_mean` | Mean score of the greedy eval games. | eval updates |
| `eval/score_max` | Best eval score. | eval updates |
| `eval/score_min` | Worst eval score. | eval updates |
| `eval/length_mean` | Mean eval episode length in steps. | eval updates |
| `eval/end_wall` | Fraction of eval games that ended on a wall; the three `eval/end_*` keys sum to 1. | eval updates |
| `eval/end_body` | Fraction of eval games that ended on the snake's own body. | eval updates |
| `eval/end_starved` | Fraction of eval games that ended by starving. | eval updates |
| `eval/steps_per_apple` | Total eval steps divided by total apples eaten. | eval updates with at least one apple |
| `curriculum/stage` | Stage the update trained in, from 0. | when `curriculum.grow` is on |
| `curriculum/board_size` | Arena side of that stage, walls included. | when `curriculum.grow` is on |
| `curriculum/goal_length` | Goal length of that stage (0 = none). | when `curriculum.grow` is on |
| `curriculum/window_score` | Mean score of the promotion window. | when `curriculum.grow` is on and the promotion window is full |
| `curriculum/promoted` | 0 none, 1 by the gate, 2 forced by `curriculum.max_updates`. | when `curriculum.grow` is on |

Eval updates are every `train.eval_interval`-th update and the final update.

### Checkpoints

`best.pt` and `last.pt` are dictionaries saved with `torch.save`:

| Key | Content |
| --- | --- |
| `model` | Model `state_dict`. |
| `optimizer` | Adam optimizer `state_dict`. |
| `update` | Update number at save time. |
| `env_steps` | Environment steps at save time. |
| `config` | The full config as a dict, enough to rebuild the model. |
| `schema_version` | Checkpoint format version (`3`). |
| `best_eval` | Best eval mean score so far. |
| `wall_time` | Seconds of training at save time, including time before a resume. |
| `rng` | Random state: `python`, `numpy`, `torch` (CPU) and `cuda` (`null` off CUDA). |
| `env` | The training env's full state, so a resume continues the same games. |
| `curriculum` | Stage, updates in the stage and the promotion window; only when `curriculum.grow` is on. |

`uv run snakeppo eval <run folder or .pt file>` evaluates a checkpoint (`--checkpoint best`
or `last`, default `best`); `--render` watches it play.

## Resuming a run

```sh
uv run snakeppo train --resume runs/<run>
```

This continues the run in place from `checkpoints/last.pt`. It drops the `metrics.jsonl` lines
logged after that checkpoint, purges the stale TensorBoard events and adds a timestamp to
`resumed`. It takes no other option. It resumes schema-2 and schema-3 runs; a schema-2 run
continues as schema 3 and its `run.json` then says `3`. It refuses a finished run, a schema-1
run, a run of a newer schema, a run with no `last.pt`, a run trained on a different device
type, and a run still `running` within the stall window. The continuation is exact on CPU, and on GPU with `train.deterministic: true`.

## Seed sweeps

```sh
uv run snakeppo sweep --seeds 1,2,3 --config configs/baseline.yaml [--set …] [--jobs 2]
```

This trains one run per seed as one seed group, with the id `<train.name>@<YYYYmmdd-HHMMSS>`.
Each seed runs in its own `snakeppo train` child process. With `--jobs` above 1 the children's
output is captured, and a failed seed's last 20 lines (`FAILURE_TAIL_LINES` in `sweep.py`) are
printed when it ends. The command exits with code 1 if any seed failed. Add a seed to an
existing group by hand with `uv run snakeppo train … --seed 4 --group <id>`.

## Comparing runs

**What Score means.** Score is `eval/score_mean`: apples eaten in greedy play on the full board
with no goal. It compares across reward strategies and curricula. `train/return_mean` and
`train/reward_mean` are shaped values and never compare across `reward.strategy`; under a
curriculum `train/score_mean` is stage-relative.

`snakeppo compare` prints one row per run, sorted by best eval score:

```sh
uv run snakeppo compare            # every run under runs/
uv run snakeppo compare runs/a runs/b
```

Each path is a run folder or a folder of run folders.

To judge whether a switch helped, compare seed groups with the fair-comparison protocol:

```sh
uv run snakeppo results [paths] [--baseline GROUP] [--format text|markdown|csv]
```

The protocol is implemented in `src/snakeppo/results.py`:

- **Run score.** The mean of the run's last 3 `eval/score_mean` values. A run with fewer evals
  uses what it has, and the table's footer says so.
- **Bookkeeping keys** (`BOOKKEEPING_KEYS`) never count: `env.seed`, `train.name`,
  `train.group`, `train.runs_dir`, `train.device`, `train.deterministic`,
  `train.checkpoint_interval`.
- **Protocol keys** (`PROTOCOL_KEYS`) set the budget and how runs are measured:
  `train.total_env_steps`, `train.eval_interval`, `train.eval_episodes`. They are shown as the
  budget, never as a switch.
- **Switches** are a config's differences from the code defaults, minus the protocol keys.
- **Switch units.** A selector and the keys only it reads count as one switch
  (`SELECTOR_KEYS`): `reward.strategy` with `reward.distance_scale`, `reward.potential_scale`,
  `reward.timeout_penalty`, `reward.hunger_penalty` and `reward.free_space_penalty`;
  `curriculum.grow` with every other `curriculum.*` key.
- **Groups.** Runs are grouped by `group`. Ungrouped runs are left out of the table, and a group
  whose runs have different configs gets no verdict.
- **Reference.** The newest group with no switches and the same protocol keys, or the group
  named by `--baseline`. A group trained from `configs/smoke.yaml` has switches, so comparing
  against it needs `--baseline`.
- **Pairing.** Seeds are paired by `env.seed` over finished runs.
- **Verdict.** The mean paired score difference with a 95% percentile bootstrap CI (10,000
  resamples, fixed seed): `better` when the whole CI is above zero, `worse` when it is below,
  otherwise `inconclusive`.

A withheld verdict shows one of these reasons:

- `mixed configs`
- `no baseline group with the same budget and eval settings`
- `the reference group has mixed configs`
- `duplicate seed <seed>` or `duplicate seed <seed> in the reference group`
- `budget or eval settings differ from the reference (<keys>)`
- `<n> learning switches differ from the reference (at most 1)`
- `only <n> paired finished seeds (need 3)`

## Dashboard

```sh
uv run snakeppo serve [--runs-dir runs] [--port 8765] [--open]
```

This serves a local dashboard at `http://127.0.0.1:8765`. It binds to 127.0.0.1 only, is
read-only and never writes to a run folder. It has three tabs:

- **Live** follows a run as it trains, polling about every 2 s, and shows how long ago its
  metrics were last written. A `running` run that stays quiet past the stall window (see Run
  folders) shows as `stalled`.
- **Compare** overlays the curves of the ticked runs and lists their config differences. It
  warns when the ticked runs differ in `reward.strategy` or `curriculum.grow`: their training
  curves then measure different things, while the eval score stays comparable.
- **Results** shows the `snakeppo results` table. When a group has no reference, it explains why
  and offers the comparable groups to pick from.

The **Evaluation** card shows steps per apple and how the eval games ended; the **Curriculum**
card shows the stage when a run has one.

TensorBoard still works, with env steps on the x-axis:

```sh
uv run tensorboard --logdir runs
```
