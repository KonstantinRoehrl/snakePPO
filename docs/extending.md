# Extending snakePPO

## Add an architecture

1. In `src/snakeppo/agents/networks.py`, subclass `PolicyNetwork`. The constructor takes
   `board_size, in_channels, num_actions`, and `forward` returns `(logits, value)` with
   logits of shape `(B, num_actions)` and value of shape `(B,)` (squeeze the last
   dimension). `act` is inherited. `board_size` is the observation's side, which is
   `observation.view_size` with the local view, and `in_channels` its channel count. Reuse
   `conv_features` and `DeterministicAdaptiveAvgPool2d` rather than `nn.AdaptiveAvgPool2d`,
   which has no deterministic CUDA backward and breaks `train.deterministic: true`.
2. In `src/snakeppo/agents/registry.py`, call `register(AgentSpec(name, description,
   YourPolicy))`. Pass `requires_even_board=True` if layer sizes depend on an even board, for
   example a flatten after a stride-2 conv; `build_agent` then rejects odd boards with a
   clear error, and `make_policy` rejects the local view, whose side is odd. Pass
   `size_agnostic=True` only if a model built for one board size runs on any other size of
   at least 5 (a pool before every size-dependent layer); `snakeppo agents` shows it as
   `any size` and `snakeppo eval --board-size` relies on it.
3. In `tests/test_agents.py`, add the name to `SIZE_AGNOSTIC` or `BOARD_BOUND` (the shape and
   size tests run over both) and to the `available_agents()` expectation; the even-board
   test lists the networks with `requires_even_board`.
4. Check it with `uv run snakeppo agents`, train it with `--agent <name>`, and add
   `configs/agents/<name>.yaml`. On a CUDA machine, run
   `uv run pytest tests/test_training.py -k cuda`: it trains every registered architecture
   with `train.deterministic: true`.

### A recurrent architecture

Subclass `RecurrentPolicy` (`src/snakeppo/agents/recurrent.py`) instead of `PolicyNetwork`,
as `RecurrentCNNPolicy` does:

- **State layout.** One tensor `(B, K, H)` per batch of envs: set `state_slots` (K, 1 for a
  GRU's `h`, 2 for an LSTM's `h` and `c`) and `hidden_size` (H) in the constructor;
  `initial_state(batch_size, device)` returns zeros.
- **`unroll(obs, state, starts)`** runs T steps: `obs` `(T, B, C, S, S)`, `state`
  `(B, K, H)`, `starts` `(T, B)` bool. It zeroes the state rows where `starts[t]` is true
  before step `t` and returns `(logits (T, B, A), values (T, B), final_state (B, K, H))`.
  `forward(obs, state, starts)` (one step) and `act(obs, state, starts, greedy=False)`
  (returning `(actions, log_probs, values, new_state)`) are inherited.
- **What the trainer does with it.** The rollout zeroes the state of every env whose episode
  ended (and of every env after a curriculum promotion), and checkpoints store it as
  `policy_state`. The update splits the envs into `recurrent_minibatches(config)` groups of
  whole envs and replays each group with `unroll` from the rollout's start state, with
  `episode_starts(done)` as `starts`, so that count must divide `env.num_envs`
  (`make_policy` refuses the config otherwise).
- **Playing.** `evaluate` and `snakeppo eval --render` play through `PolicyPlayer`, which keeps
  one state per game and resets it when the game's episode ends.
- Add the new architecture's cases to `tests/test_recurrent.py` (shapes, `unroll` equals one
  `forward` per step, a replayed rollout reproduces its log-probs) and a smoke train to
  `tests/test_recurrent_training.py`.

## Add a switch

1. Add a field to the config dataclass in `src/snakeppo/config.py` (usually `PPOConfig`)
   with the vanilla baseline value as its default, so existing runs do not change.
2. Reject meaningless values in `validate`.
3. Use it in `src/snakeppo/ppo/trainer.py`; put pure math in `src/snakeppo/ppo/functional.py`.
4. If it has a visible effect, log it as a metric (and add the key to the
   `metrics.jsonl` table in [experiments.md](experiments.md#metricsjsonl)).
5. Add a test proving that with the default value training behaves exactly as before.
6. Turn a PPO switch on in `configs/tuned.yaml`; a reward, curriculum or observation switch
   gets its own preset under `configs/rewards/`, `configs/curriculum/` or
   `configs/observation/` instead. Add it to the switch table
   in [experiments.md](experiments.md#switches).

## Add a reward strategy

1. In `src/snakeppo/config.py`, add the name to `REWARD_STRATEGIES`, and any coefficient to
   `RewardConfig` and `STRATEGY_PARAMETERS` (which rejects negative values), with a validation
   rule in `validate` for anything stricter.
2. In `src/snakeppo/rewards.py`, write a frozen dataclass with
   `__call__(self, step: StepContext) -> torch.Tensor` and `register` a `RewardSpec` that
   builds it from the `RewardConfig` and the discount.
3. Add hand-computed tests to `tests/test_rewards.py`.
4. Add `configs/rewards/<name>.yaml` and the rows in
   [experiments.md](experiments.md#reward-strategies).

## Add an observation variant

The observation is described by `ObservationSpec` in `src/snakeppo/observation.py`, built from
the `observation` config section, and built by `TorchSnakeEnv.observe` in
`src/snakeppo/env.py`:

1. In `src/snakeppo/config.py`, add the key to `ObservationConfig` with a default that keeps
   today's observation, and reject meaningless values in `_validate_observation`.
2. In `src/snakeppo/observation.py`, add the field to `ObservationSpec`, set it in
   `from_config`, and validate it in `__post_init__`. Keep `channels` (the fixed order:
   body, `head`, `food`, `wall`, then the optional channels) and `is_binary` (false as soon as
   a value can be other than 0 or 1, so the rollout buffer stores `float32` and never
   truncates it) true to it; `record` then changes with it, which is what checkpoints are
   checked against. Put the tensor math in a pure helper next to `body_order_grid`,
   `tail_grid` and `local_view`.
3. In `TorchSnakeEnv.observe`, apply it before the four-way rotation. A spatial channel is
   built on the full board before the local view crops; a uniform one is appended after it,
   as `hunger` is.
4. Nothing else needs to change: `make_policy` and `load_policy` in `src/snakeppo/builders.py`
   take the network's input side and channel count from the spec, and `run.json` and every
   checkpoint record it. `render_grid` stays the 4-channel display grid, and
   `Viewer.draw(grids, scores, body_order)` in `src/snakeppo/viewer.py` reads it with the
   `render_body_order` grid, whatever the policy observes.
5. Test the values with hand-built snakes (`place_snake`) in `tests/test_env.py` and the
   helper in `tests/test_observation.py`, train it end to end in `tests/test_training.py`
   (`OBSERVATION_SWITCHES`), and add `configs/observation/<name>.yaml` and the switch-table
   row in [experiments.md](experiments.md#switches).
