# Extending snakePPO

## Add an architecture

1. In `src/snakeppo/agents/networks.py`, subclass `PolicyNetwork`. The constructor takes
   `board_size, in_channels, num_actions`, and `forward` returns `(logits, value)` with
   logits of shape `(B, num_actions)` and value of shape `(B,)` (squeeze the last
   dimension). `act` is inherited.
2. In `src/snakeppo/agents/registry.py`, call `register(AgentSpec(name, description,
   YourPolicy))`. Pass `requires_even_board=True` if layer sizes depend on the board, for
   example a flatten after a stride-2 conv; `build_agent` then rejects odd boards with a
   clear error.
3. In `tests/test_agents.py`, add the name to the parametrised name lists (and to the
   even-board list if it applies) and to the `available_agents()` expectation.
4. Check it with `uv run snakeppo agents` and train it with `--agent <name>`.

## Add a switch

1. Add a field to the config dataclass in `src/snakeppo/config.py` (usually `PPOConfig`)
   with the vanilla baseline value as its default, so existing runs do not change.
2. Reject meaningless values in `validate`.
3. Use it in `src/snakeppo/ppo/trainer.py`; put pure math in `src/snakeppo/ppo/functional.py`.
4. If it has a visible effect, log it as a metric (and add the key to the
   `metrics.jsonl` table in [experiments.md](experiments.md#metricsjsonl)).
5. Add a test proving that with the default value training behaves exactly as before.
6. Turn it on in `configs/tuned.yaml` and add it to the switch table in
   [experiments.md](experiments.md#switches).

## Add an observation variant (C4)

The observation is defined in `src/snakeppo/env.py` and consumed in a few places that must
change together:

1. `TorchSnakeEnv.render_grid` builds the unrotated grid and `TorchSnakeEnv.observe`
   rotates it for the policy; add or change channels there.
2. Update `NUM_CHANNELS`, which sets `obs_shape`.
3. Pass the new channel count as `in_channels` to `build_agent`. It defaults to 4, and
   both `train` in `src/snakeppo/ppo/trainer.py` and `_load_model` in
   `src/snakeppo/cli.py` currently rely on that default.
4. `Viewer.draw` in `src/snakeppo/viewer.py` unpacks the channels in the order body, head,
   food, wall; keep that order or update the viewer.
