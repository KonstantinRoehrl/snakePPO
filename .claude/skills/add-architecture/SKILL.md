---
name: add-architecture
description: Use when adding a new policy network.
---

# Add an architecture

1. Follow `docs/extending.md` § Add an architecture.
2. Run the verification gate:
   `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
3. Run `uv run snakeppo agents` to confirm the new network is listed and to see its parameter
   count.
