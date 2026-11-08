# snakePPO

PPO research bench for Snake on a grid observation. Every learning-altering improvement is a
switch, and the code defaults are the plain baseline. Start with `docs/architecture.md`.

## Commands

- Setup: `uv sync` (CUDA torch on Windows/Linux; CPU-only: `uv sync --no-group cuda --group cpu`)
- Verify (run before every commit): `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- Train: `uv run snakeppo train --config configs/baseline.yaml [--set section.key=value]`
- Evaluate: `uv run snakeppo eval runs/<run> [--render]`; list agents: `uv run snakeppo agents`
- Resume an interrupted run: `uv run snakeppo train --resume runs/<run>`
- Seed sweep: `uv run snakeppo sweep --seeds 1,2,3 --config configs/baseline.yaml [--jobs 2]`
- Verdicts across seed groups: `uv run snakeppo results [--baseline GROUP]`
- Dashboard: `uv run snakeppo serve [--open]`

## Layout

- `src/snakeppo/` — the package (module map: `docs/architecture.md`)
- `configs/` — presets; `smoke.yaml` is the tiny run the tests use
- `tests/` — fast CPU tests, including an end-to-end smoke train
- `runs/` — training output, never committed
- `docs/` — architecture, experiments (run folder + metrics schema), extending

## Invariants

- New switches default to the baseline behaviour; add them per `docs/extending.md`.
- No import-time side effects (pygame, argument parsing, device probing happen in functions).
- `metrics.jsonl` and `run.json` are a versioned contract (`docs/experiments.md`); changing a key
  means bumping `runs.SCHEMA_VERSION` and updating that doc.
- Code tests run on CPU; never require a GPU in tests or CI.
- After structural changes (modules added, moved or removed) run `graphify update .` and commit
  `graphify-out/`.

## Skills

`.claude/skills/`: `run-experiment`, `add-architecture`, `add-toggle`, `compare-runs`.
