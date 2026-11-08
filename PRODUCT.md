# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Charts: Plotly (the user's choice, for modern-looking graphs). The dashboard is a stdlib HTTP
server inside the `snakeppo` package, serving a plain ES-module frontend with no build step and
the plotly.js that the `plotly` Python package bundles. Everything else is the existing Python package: uv, torch, the `snakeppo` CLI.

## Users

One person: the researcher who owns this repo, working alone in a browser on the same machine
that trains (localhost only). Claude agents run experiments too, but they work through the CLI
and the run files on disk, not the web UI.

## Product Purpose

snakePPO is a research bench for PPO on Snake. Every learning-altering improvement is a config
switch whose default is the plain baseline, so an experiment changes one thing and compares
against `configs/baseline.yaml`. The dashboard serves two jobs, equally important:

- **Watching a live run:** is it learning, has it diverged, should it be stopped?
- **Judging a switch:** did the change help across several seeds, not just on one lucky run?

Success means the researcher trusts what the dashboard shows enough to keep or drop a switch
on it.

## Positioning

The dashboard is built around this bench's own contract, not around generic experiment tracking:
the versioned run folder (`run.json`, `config.yaml`, `metrics.jsonl`, checkpoints) and the
one-switch-at-a-time, multi-seed comparison protocol. It only reads run files and never
changes how training behaves.

## Operating Context

- Training runs from the terminal (`uv run snakeppo train ...`) and writes `runs/<run>/`.
  `metrics.jsonl` gets one line per PPO update, and a run can take minutes or hours.
- The researcher keeps the dashboard open next to the terminal while a run trains, and comes
  back to it after a batch of seeded runs finishes.
- TensorBoard event files are still written to `runs/<run>/tb/`; the dashboard is meant to
  replace day-to-day use of TensorBoard for this bench.

## Capabilities and Constraints

- Reads the run folder contract in `docs/experiments.md`; `run.json` and `metrics.jsonl` are
  versioned by `runs.SCHEMA_VERSION`.
- Local only: binds to localhost, no authentication, no remote access.
- No import-time side effects; tests run on CPU only and never need a GPU.
- Out of scope for now: launching runs from the browser and in-browser replay of a trained
  agent (planned for a later cycle).

## Evidence on Hand

Real run folders under `runs/` (not committed). No screenshots, users or benchmarks beyond those
exist; future work must not invent any.

## Product Principles

1. **Read-only toward training.** The dashboard observes run files; it never changes a run's
   learning.
2. **Seeds over anecdotes.** A switch's effect is shown across seeds, with the spread visible,
   before it is shown as one number.
3. **The baseline is the reference.** Comparisons are framed against the baseline config, one
   switch at a time.
4. **The contract is the source of truth.** Everything shown comes from the documented run
   folder format, so the CLI, agents and the dashboard see the same facts.
