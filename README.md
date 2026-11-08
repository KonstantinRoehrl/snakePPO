# snakePPO

A research bench for training PPO agents that play Snake from a grid observation. Every
learning-altering improvement is a config switch, and the defaults are the plain baseline, so
experiments compare one change at a time.

## Quickstart

```bash
uv sync                                                  # CPU-only: uv sync --no-group cuda --group cpu
uv run snakeppo train --config configs/smoke.yaml        # tiny run, finishes in seconds
uv run snakeppo train --config configs/baseline.yaml     # a real run
uv run snakeppo eval runs/<run> --render                 # watch the trained agent play
```

## Commands

- `uv run snakeppo train` — train an agent and write a run folder under `runs/`.
- `uv run snakeppo eval <run>` — evaluate a run folder or checkpoint; `--render` opens a window.
- `uv run snakeppo agents` — list the available architectures with their parameter counts.
- `uv run snakeppo compare [runs/…]` — summarise finished runs as a table.
- `uv run snakeppo train --resume runs/<run>` — continue an interrupted run exactly from its last checkpoint.
- `uv run snakeppo sweep --seeds 1,2,3 --config <file>` — train one run per seed as one seed group.
- `uv run snakeppo results [runs/…]` — judge seed groups against their baseline group, with a verdict per switch.
- `uv run snakeppo serve [--open]` — open the read-only local dashboard (live runs, curves, results).

Run `uv run snakeppo <command> --help` for every flag.

## Verification

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## Documentation

- [Architecture](docs/architecture.md) — module map and how the pieces fit.
- [Experiments](docs/experiments.md) — run folders, the metrics schema, resuming, sweeps, comparing runs and the dashboard.
- [Extending](docs/extending.md) — adding architectures, switches and observation variants.
