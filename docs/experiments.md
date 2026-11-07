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

```sh
uv run snakeppo train --config configs/baseline.yaml
uv run snakeppo train --config configs/tuned.yaml --set ppo.target_kl=null --seed 1
```

Values are applied in this order, each step overriding the previous one:

1. the code defaults in `src/snakeppo/config.py`;
2. the `--config` YAML file (a partial file is fine);
3. each `--set SECTION.KEY=VALUE`, in order (the value is parsed as YAML, so `null`,
   `true` and numbers work);
4. `--agent`, `--name` and `--seed`, which set `agent.name`, `train.name` and `env.seed`.

Unknown sections or keys and values of the wrong type are rejected with an error.

## Switches

Each switch below defaults to the vanilla baseline.

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `ppo.lr_schedule` | `constant`, `linear` | `constant` | `linear` decays `ppo.lr` to 0 over the run. |
| `ppo.clip_value_loss` | `true`, `false` | `false` | Clips the value update to `ppo.clip_eps` around the rollout's values. |
| `ppo.entropy_schedule` | `constant`, `linear`, `exponential` | `constant` | Moves the entropy coefficient from `ppo.entropy_coef` to `ppo.entropy_coef_final` over the run. |
| `ppo.entropy_coef_final` | float (> 0 for `exponential`) | `0.0005` | End value of the entropy schedule; unused with `constant`. |
| `ppo.target_kl` | float or `null` | `null` | Stops an update's epochs early once a minibatch's approximate KL exceeds it. |
| `train.deterministic` | `true`, `false` | `false` | Forces deterministic torch kernels so GPU runs repeat exactly (slower). |

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
        last.pt     checkpoint from the most recent eval
```

A `_2`, `_3`, ... suffix is added if the folder name already exists.

`run.json` fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Run-folder format version, currently `1`. Covers this layout, the metrics keys and the checkpoint contents. |
| `name` | `train.name`. |
| `created` | Local timestamp with UTC offset, to the second. |
| `git_sha` | Commit the run was trained from, or `null` outside a git checkout. |
| `agent` | `agent.name`. |
| `param_count` | Number of model parameters. |
| `device` | Device the run trained on, for example `cuda` or `cpu`. |

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

`uv run snakeppo eval <run folder or .pt file>` evaluates a checkpoint (`--checkpoint best`
or `last`, default `best`); `--render` watches it play.

## Comparing runs

Print a table of runs sorted by best eval score:

```sh
uv run snakeppo compare            # every run under runs/
uv run snakeppo compare runs/a runs/b
```

Each path is a run folder or a folder of run folders. Plot the curves with:

```sh
uv run tensorboard --logdir runs
```

TensorBoard's x-axis is env steps, so runs with different rollout sizes line up.

For a fair comparison, run both sides with the same seeds and the same
`train.total_env_steps`, and change one switch at a time against `configs/baseline.yaml`.
