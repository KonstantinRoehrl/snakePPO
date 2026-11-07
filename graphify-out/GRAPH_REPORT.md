# Graph Report - snakePPO  (2026-11-07)

## Corpus Check
- 38 files · ~10,917 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 3 file(s) not represented in the graph (top: (none) 2, .lock 1)

## Summary
- 408 nodes · 856 edges · 23 communities (16 shown, 7 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 97 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7d948a13`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cli.py
- trainer.py
- ppo
- config.py
- runs.py
- torch
- test_env.py
- Data flow of one update
- verify
- Viewer
- train
- updates
- ppo
- snakeppo
- cli.py
- test_env.py
- snakePPO
- add-architecture/SKILL.md
- add-toggle/SKILL.md
- compare-runs/SKILL.md
- run-experiment/SKILL.md

## God Nodes (most connected - your core abstractions)
1. `train()` - 35 edges
2. `Module map` - 28 edges
3. `build_agent()` - 19 edges
4. `Config` - 18 edges
5. `load_config()` - 18 edges
6. `ppo` - 17 edges
7. `PolicyNetwork` - 17 edges
8. `evaluate()` - 17 edges
9. `TorchSnakeEnv` - 15 edges
10. `ConfigError` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Add an observation variant (C4)` --references--> `build_agent()`  [INFERRED]
  docs/extending.md → src/snakeppo/agents/registry.py
- `Module map` --references--> `resolve_device()`  [INFERRED]
  docs/architecture.md → src/snakeppo/builders.py
- `Module map` --references--> `seed_everything()`  [INFERRED]
  docs/architecture.md → src/snakeppo/builders.py
- `Module map` --references--> `make_env()`  [INFERRED]
  docs/architecture.md → src/snakeppo/builders.py
- `Add an observation variant (C4)` --references--> `_load_model()`  [INFERRED]
  docs/extending.md → src/snakeppo/cli.py

## Import Cycles
- None detected.

## Communities (23 total, 7 thin omitted)

### Community 0 - "cli.py"
Cohesion: 0.14
Nodes (10): Module map, Add an architecture, CNNPolicy, PolicyNetwork, ResNetPolicy, SpatialCNNPolicy, AgentSpec, available_agents() (+2 more)

### Community 1 - "trainer.py"
Cohesion: 0.14
Nodes (8): make_env(), resolve_device(), seed_everything(), EnvConfig, RewardConfig, EvalStats, evaluate(), test_evaluate_counts_one_finished_episode_per_game_and_is_reproducible()

### Community 2 - "ppo"
Cohesion: 0.05
Nodes (38): agent, kwargs, name, env, board_size, num_envs, seed, starvation_steps (+30 more)

### Community 3 - "config.py"
Cohesion: 0.13
Nodes (23): AgentConfig, apply_overrides(), _coerce(), Config, config_from_dict(), config_to_dict(), ConfigError, load_config() (+15 more)

### Community 4 - "runs.py"
Cohesion: 0.10
Nodes (21): train(), create_run_dir(), find_runs(), git_sha(), load_checkpoint(), MetricsLogger, read_metrics(), save_checkpoint() (+13 more)

### Community 5 - "torch"
Cohesion: 0.13
Nodes (20): compute_gae(), entropy_coef_at(), explained_variance(), LossOutput, lr_at(), ppo_loss(), test_board_dependent_agents_reject_odd_boards(), test_cnn_accepts_odd_boards() (+12 more)

### Community 6 - "test_env.py"
Cohesion: 0.22
Nodes (3): Environment contract, Add an observation variant (C4), TorchSnakeEnv

### Community 7 - "Data flow of one update"
Cohesion: 0.08
Nodes (21): Architecture, Data flow of one update, Invariants, Checkpoints, Comparing runs, Install, `metrics.jsonl`, Presets and config load order (+13 more)

### Community 8 - "verify"
Cohesion: 0.07
Nodes (27): concurrency, cancel-in-progress, group, run, jobs, verify, os, name (+19 more)

### Community 9 - "Viewer"
Cohesion: 0.11
Nodes (6): Viewer, _grids(), test_closing_the_window_stops_drawing_cleanly(), test_draw_keeps_one_font_and_reports_open(), test_importing_the_viewer_does_not_import_pygame(), viewer()

### Community 10 - "train"
Cohesion: 0.12
Nodes (16): agent, name, env, board_size, num_envs, starvation_steps, ppo, minibatch_size (+8 more)

### Community 11 - "updates"
Cohesion: 0.17
Nodes (11): patterns, github-actions, python-dependencies, patterns, interval, updates, directory, groups (+3 more)

### Community 12 - "ppo"
Cohesion: 0.20
Nodes (9): ppo, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, lr_schedule, target_kl, train (+1 more)

### Community 17 - "cli.py"
Cohesion: 0.14
Nodes (15): count_parameters(), _agents(), _compare(), _eval(), _load_model(), main(), _parser(), _train() (+7 more)

### Community 18 - "test_env.py"
Cohesion: 0.34
Nodes (13): _act(), _env(), _food_after_resets(), test_eating_grows_scores_and_respawns_food_off_the_body(), test_observation_is_rotated_so_the_snake_faces_up(), test_only_finished_envs_reset(), test_relative_actions_turn_the_snake(), test_reset_starts_every_snake_in_the_centre_facing_up() (+5 more)

### Community 19 - "snakePPO"
Cohesion: 0.25
Nodes (6): Agent instructions, Commands, Invariants, Layout, Skills, snakePPO

## Knowledge Gaps
- **1 isolated node(s):** `snakeppo`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 175 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Module map` connect `cli.py` to `trainer.py`, `config.py`, `runs.py`, `torch`, `test_env.py`, `Data flow of one update`, `Viewer`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `train()` (e.g. with `Data flow of one update` and `Module map`) actually correct?**
  _`train()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `snakeppo` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `cli.py` be split into smaller, more focused modules?**
  _Cohesion score 0.135632183908046 - nodes in this community are weakly interconnected._
- **Why does `train()` connect `runs.py` to `cli.py`, `trainer.py`, `config.py`, `torch`, `test_env.py`, `Data flow of one update`, `cli.py`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `Module map` (e.g. with `CNNPolicy` and `PolicyNetwork`) actually correct?**
  _`Module map` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Should `trainer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._