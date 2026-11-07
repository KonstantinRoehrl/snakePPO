---
name: add-toggle
description: Use when adding a new training switch or improvement.
---

# Add a switch

1. Follow `docs/extending.md` § Add a switch.
2. The default value must reproduce today's behaviour; add a test that proves it.
3. Run the verification gate:
   `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
