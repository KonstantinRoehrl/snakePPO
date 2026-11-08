---
name: add-architecture
description: Use when adding a new policy network.
---

# Add an architecture

1. Follow `docs/extending.md` § Add an architecture (and § A recurrent architecture for a network
   with a hidden state). Decide `size_agnostic` and `requires_even_board` from the layers: a
   network is size-agnostic only if a model built for one board runs on any board of at least 5.
2. Add `configs/agents/<name>.yaml` (the `agent.name` switch plus `train.name: agent-<name>`) and
   the row in `docs/experiments.md` § Architectures.
3. Run the verification gate:
   `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
4. Run `uv run snakeppo agents` to confirm the new network is listed with its parameter count and
   its `any size` value. On a CUDA machine, also run `uv run pytest tests/test_training.py -k cuda`
   (deterministic training for every architecture; CI has no GPU).
