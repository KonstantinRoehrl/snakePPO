---
name: compare-runs
description: Use when asked to compare finished runs or judge whether a switch helped.
---

# Compare runs

1. To judge whether a switch helped, run `uv run snakeppo results [runs/…]` (defaults to `runs/`).
   It compares seed groups (from `snakeppo sweep` or `train --group`) against their baseline
   group on paired seeds and prints `better`, `worse` or `inconclusive` with a bootstrap CI. Pass
   `--baseline GROUP` to pick the reference group yourself.
2. A verdict is withheld when the comparison would not be fair, and the table says why: for
   example mixed configs in a group, no baseline group with the same budget, more than one
   learning switch changed, or fewer than 3 paired finished seeds. Report the reason; never turn a
   withheld or inconclusive result into a claim. The full rules are in `docs/experiments.md`
   § Comparing runs.
3. For training curves, point to `uv run snakeppo serve` (the read-only dashboard;
   `uv run tensorboard --logdir runs` still works).
4. `uv run snakeppo compare [runs/…]` stays the quick per-run table of best and last eval scores.
