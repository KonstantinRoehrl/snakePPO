# snakePPO

PPO research bench for Snake on a grid observation. Every learning-altering improvement is a
switch, and the code defaults are the plain baseline. Start with `docs/architecture.md`.

## Commands

- Setup: `uv sync` (CUDA torch on Windows/Linux; CPU-only: `uv sync --no-group cuda --group cpu`)
- Verify (run before every commit): `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- Train: `uv run snakeppo train --config configs/baseline.yaml [--set section.key=value]`
  (`--description TEXT` says in plain words what the run tries)
- Evaluate: `uv run snakeppo eval runs/<run> [--board-size N]`; watch it play in the browser:
  `uv run snakeppo eval runs/<run> --render [--port P]`; list agents (with whether each plays any board
  size): `uv run snakeppo agents`; list reward strategies: `uv run snakeppo rewards`
- Resume an interrupted run: `uv run snakeppo train --resume runs/<run>`
- Seed sweep: `uv run snakeppo sweep --seeds 1,2,3 --config configs/baseline.yaml [--jobs 2]`;
  `--group GROUP` joins an existing group
- Verdicts across seed groups: `uv run snakeppo results [--baseline GROUP]`
- Dashboard, the main UI (launch, stop, resume, replay): `uv run snakeppo serve [--open]`

## CI

`main` takes PRs only; the ruleset requires the `verify` (Ubuntu and Windows), secret-scan,
workflow-lint and PR-title checks. PR titles must be Conventional Commits. Workflows pin actions to
commit SHAs and run `actionlint` and `zizmor`. Optional local secret scan: `uvx pre-commit install`.

## Layout

- `src/snakeppo/` — the package (module map: `docs/architecture.md`)
- `configs/` — presets; `smoke.yaml` is the tiny run the tests use
- `tests/` — fast CPU tests, including an end-to-end smoke train
- `runs/` — training output and dashboard launch records (`runs/.launches/`), never committed
- `docs/` — architecture, experiments (run folder + metrics schema), extending

## Invariants

- New switches default to the baseline behaviour; add them per `docs/extending.md`.
- No import-time side effects (argument parsing, device probing, hostname lookups and torch
  thread settings happen in functions).
- `metrics.jsonl` and `run.json` are a versioned contract (`docs/experiments.md`); changing a key
  means bumping `runs.SCHEMA_VERSION` and updating that doc.
- Code tests run on CPU; never require a GPU, a real browser or the network in tests or CI.
- The dashboard never changes how a run learns: it launches the same CLI commands a terminal runs
  and stops runs only through a `stop-request` file; every POST passes its origin check
  (`docs/architecture.md` § Invariants).
- After structural changes (modules added, moved or removed) run `graphify update .` and commit
  `graphify-out/`.

## Skills

`.claude/skills/`: `run-experiment`, `add-architecture`, `add-toggle`, `compare-runs`.
