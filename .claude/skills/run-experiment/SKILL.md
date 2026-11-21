---
name: run-experiment
description: Use when asked to train an agent or run an experiment.
---

# Run an experiment

1. Pick a preset from `configs/` (`baseline.yaml`, `tuned.yaml`, one reward strategy from
   `configs/rewards/` or one curriculum from `configs/curriculum/`), or copy one to a new file
   when the experiment needs several changed values. For a quick check use `configs/smoke.yaml`.
2. Choose a descriptive `--name` and a `--seed`; change single values with
   `--set section.key=value` (repeatable). The run's description then becomes the preset's
   text plus `Changed: <the overrides>`; pass `--description TEXT` to write your own.
3. Run `uv run snakeppo train --config configs/<preset>.yaml --name <name> --seed <seed>`.
   Anything longer than the smoke run goes in the background. The dashboard's Launch tab
   (`uv run snakeppo serve`) starts exactly this command when the user prefers the browser.
4. When the result will be compared against a baseline, train a seed group instead:
   `uv run snakeppo sweep --seeds 1,2,3 --config configs/<preset>.yaml --name <name>`, and the
   baseline the same way. A single run cannot get a verdict from `snakeppo results`. To
   continue a stopped sweep as the same group, run the same command with the original seeds
   and `--group <group id>`: it resumes interrupted and stalled seeds, starts missing ones and
   leaves the others alone.
5. To stop a run cleanly, use the dashboard's Stop or Ctrl+C, or from another shell
   `uv run python -c "from pathlib import Path; from snakeppo.processes import request_stop; request_stop(Path('runs/<run>'))"`;
   it ends `interrupted` after the current update. Continue an interrupted, stalled or failed
   run with `uv run snakeppo train --resume runs/<run>` instead of starting over.
6. Report the run folder (or the sweep's group id and folders) the command prints. The folder
   layout and metrics are described in `docs/experiments.md`. To watch a trained run play,
   `uv run snakeppo eval runs/<run> --render` opens it in the dashboard's Replay tab.
