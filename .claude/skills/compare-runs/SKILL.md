---
name: compare-runs
description: Use when asked to compare finished runs or judge whether a switch helped.
---

# Compare runs

1. Run `uv run snakeppo compare [runs/…]` (defaults to `runs/`) for a table of best and last eval
   scores.
2. Before judging whether a switch helped, read `docs/experiments.md` § Comparing runs for the
   fair-comparison rule.
3. For training curves, point to `uv run tensorboard --logdir runs`.
