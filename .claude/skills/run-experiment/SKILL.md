---
name: run-experiment
description: Use when asked to train an agent or run an experiment.
---

# Run an experiment

1. Pick a preset from `configs/` (`baseline.yaml`, `tuned.yaml`), or copy one to a new file when
   the experiment needs several changed values. For a quick check use `configs/smoke.yaml`.
2. Choose a descriptive `--name` and a `--seed`; change single values with
   `--set section.key=value` (repeatable).
3. Run `uv run snakeppo train --config configs/<preset>.yaml --name <name> --seed <seed>`.
   Anything longer than the smoke run goes in the background.
4. When the result will be compared against a baseline, train a seed group instead:
   `uv run snakeppo sweep --seeds 1,2,3 --config configs/<preset>.yaml --name <name>`, and the
   baseline the same way. A single run cannot get a verdict from `snakeppo results`.
5. If a run was interrupted, continue it with `uv run snakeppo train --resume runs/<run>`
   instead of starting over.
6. Report the run folder (or the sweep's group id and folders) the command prints. The folder
   layout and metrics are described in `docs/experiments.md`.
