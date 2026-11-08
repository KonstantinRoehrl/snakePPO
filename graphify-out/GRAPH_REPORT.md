# Graph Report - snakePPO  (2026-11-08)

## Corpus Check
- 59 files · ~38,896 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .css 1, .lock 1)

## Summary
- 931 nodes · 2221 edges · 50 communities (32 shown, 18 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 208 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1d65cf9d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cli.py
- train
- ppo
- Config
- runs.py
- trainer.py
- TorchSnakeEnv
- app.js
- verify
- server.py
- train
- updates
- ppo
- on-device-results.md
- snakeppo
- builders.py
- snakePPO
- add-architecture/SKILL.md
- add-toggle/SKILL.md
- compare-runs/SKILL.md
- run-experiment/SKILL.md
- api.py
- results.py
- Components
- test_server.py
- charts.js
- cli.py
- main
- On-device checklist — C2 dashboard
- test_runs.py
- Product
- ._route
- test_env.py
- build_agent
- Running experiments
- Design System: snakePPO dashboard
- MetricsLogger
- Colors
- snakePPO
- Typography
- c2-experiments-dashboard/on-device-results.md

## God Nodes (most connected - your core abstractions)
1. `Module map` - 42 edges
2. `Config` - 41 edges
3. `train()` - 37 edges
4. `resume()` - 31 edges
5. `main()` - 30 edges
6. `RunRecord` - 25 edges
7. `load_run()` - 25 edges
8. `apply_overrides()` - 21 edges
9. `load_config()` - 21 edges
10. `build_agent()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `Module map` --references--> `ResumeError`  [INFERRED]
  docs/architecture.md → src/snakeppo/ppo/trainer.py
- `Module map` --references--> `RunRecord`  [INFERRED]
  docs/architecture.md → src/snakeppo/runs.py
- `Data flow of one update` --references--> `MetricsLogger`  [INFERRED]
  docs/architecture.md → src/snakeppo/runs.py
- `Module map` --references--> `MetricsLogger`  [INFERRED]
  docs/architecture.md → src/snakeppo/runs.py
- `Module map` --references--> `EvalStats`  [INFERRED]
  docs/architecture.md → src/snakeppo/evaluation.py

## Import Cycles
- None detected.

## Communities (50 total, 18 thin omitted)

### Community 0 - "cli.py"
Cohesion: 0.11
Nodes (12): Module map, Add an architecture, CNNPolicy, DeterministicAdaptiveAvgPool2d, PolicyNetwork, _pooling_bins(), ResNetPolicy, SpatialCNNPolicy (+4 more)

### Community 1 - "train"
Cohesion: 0.07
Nodes (34): Progress, resume(), ResumeError, _run(), _Session, train(), load_checkpoint(), _interrupted_run() (+26 more)

### Community 2 - "ppo"
Cohesion: 0.05
Nodes (40): agent, kwargs, name, env, board_size, num_envs, seed, starvation_steps (+32 more)

### Community 3 - "Config"
Cohesion: 0.11
Nodes (32): Add a switch, Extending snakePPO, _experiment_config(), _train(), AgentConfig, apply_overrides(), _coerce(), Config (+24 more)

### Community 4 - "runs.py"
Cohesion: 0.14
Nodes (15): Invariants, derive_status(), git_sha(), last_write_time(), load_run(), read_metrics(), read_run_info(), replace_atomically() (+7 more)

### Community 5 - "trainer.py"
Cohesion: 0.12
Nodes (19): Architecture, Data flow of one update, compute_gae(), entropy_coef_at(), explained_variance(), LossOutput, lr_at(), ppo_loss() (+11 more)

### Community 6 - "TorchSnakeEnv"
Cohesion: 0.12
Nodes (4): Environment contract, Add an observation variant (C4), TorchSnakeEnv, Viewer

### Community 7 - "app.js"
Cohesion: 0.06
Nodes (86): ApiError, fetchDiff(), fetchMetrics(), fetchResults(), fetchRun(), fetchRuns(), getJson(), MetricsStore (+78 more)

### Community 8 - "verify"
Cohesion: 0.07
Nodes (27): concurrency, cancel-in-progress, group, run, jobs, verify, os, name (+19 more)

### Community 10 - "train"
Cohesion: 0.12
Nodes (16): agent, name, env, board_size, num_envs, starvation_steps, ppo, minibatch_size (+8 more)

### Community 11 - "updates"
Cohesion: 0.17
Nodes (11): patterns, github-actions, python-dependencies, patterns, interval, updates, directory, groups (+3 more)

### Community 12 - "ppo"
Cohesion: 0.20
Nodes (9): ppo, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, lr_schedule, target_kl, train (+1 more)

### Community 17 - "builders.py"
Cohesion: 0.14
Nodes (10): capture_rng_state(), make_env(), resolve_device(), restore_rng_state(), seed_everything(), EnvConfig, RewardConfig, EvalStats (+2 more)

### Community 19 - "snakePPO"
Cohesion: 0.25
Nodes (6): Agent instructions, Commands, Invariants, Layout, Skills, snakePPO

### Community 25 - "api.py"
Cohesion: 0.13
Nodes (12): diff_payload(), json_safe(), metrics_payload(), results_payload(), run_detail(), run_summary(), MetricsChunk, RunRecord (+4 more)

### Community 26 - "results.py"
Cohesion: 0.07
Nodes (33): Bootstrap, _cells(), config_differences(), _csv(), flatten_config(), _footer(), format_results(), _Group (+25 more)

### Community 27 - "Components"
Cohesion: 0.14
Nodes (14): Badges, Buttons, Cards / Containers, Charts (signature), Chips, Components, Empty and loading states, Inputs / Fields (+6 more)

### Community 28 - "test_server.py"
Cohesion: 0.08
Nodes (27): _serve(), DashboardServer, make_server(), serve(), static_dir(), server(), _shipped(), _status_and_body() (+19 more)

### Community 29 - "charts.js"
Cohesion: 0.14
Nodes (19): aggregate(), average(), card(), CARDS, ChartBoard, chartLayout(), DEFAULT_X, element() (+11 more)

### Community 30 - "cli.py"
Cohesion: 0.19
Nodes (9): _agents(), _eval(), _experiment_options(), _load_model(), _parser(), _seed_list(), _watch(), resolve_checkpoint() (+1 more)

### Community 31 - "main"
Cohesion: 0.05
Nodes (37): _compare(), main(), _results(), _sweep(), find_runs(), _Child, _folders_by_seed(), format_outcomes() (+29 more)

### Community 32 - "On-device checklist — C2 dashboard"
Cohesion: 0.17
Nodes (11): Compare tab, Keyboard and accessibility, Live tab, Loading and frame, On-device checklist — C2 dashboard, Responsive, Results tab, Sidebar (+3 more)

### Community 34 - "test_runs.py"
Cohesion: 0.17
Nodes (17): create_run_dir(), read_metrics_since(), truncate_metrics(), _fake_run(), _lines(), test_atomic_writes_leave_no_temp_files(), test_derive_status_reports_stored_status_and_stalls(), test_load_run_reads_a_v2_run_and_a_v1_run() (+9 more)

### Community 35 - "Product"
Cohesion: 0.18
Nodes (10): Capabilities and Constraints, Evidence on Hand, Operating Context, Platform, Positioning, Product, Product Principles, Product Purpose (+2 more)

### Community 39 - "._route"
Cohesion: 0.20
Nodes (3): _BadRequest, _Handler, _NotFound

### Community 40 - "test_env.py"
Cohesion: 0.25
Nodes (17): _act(), _env(), _food_after_resets(), _random_actions(), test_eating_grows_scores_and_respawns_food_off_the_body(), test_load_state_dict_rejects_a_different_env_shape(), test_observation_is_rotated_so_the_snake_faces_up(), test_only_finished_envs_reset() (+9 more)

### Community 41 - "build_agent"
Cohesion: 0.19
Nodes (10): build_agent(), _cnn_pool_input_size(), test_board_dependent_agents_reject_odd_boards(), test_cnn_accepts_odd_boards(), test_cnn_has_no_adaptive_pool_without_a_deterministic_backward(), test_deterministic_pool_matches_torch_adaptive_pool(), test_every_agent_maps_observations_to_logits_and_values(), test_registry_lists_the_c1_architectures() (+2 more)

### Community 42 - "Running experiments"
Cohesion: 0.15
Nodes (13): Bookkeeping keys, Checkpoints, Comparing runs, Dashboard, Install, `metrics.jsonl`, Presets and config load order, Resuming a run (+5 more)

### Community 43 - "Design System: snakePPO dashboard"
Cohesion: 0.18
Nodes (10): Design System: snakePPO dashboard, Do:, Do's and Don'ts, Don't:, Elevation & Depth, Layout, Named Rules, Overview (+2 more)

### Community 45 - "Colors"
Cohesion: 0.33
Nodes (6): Colors, Data series, Named Rules, Neutral, Primary, State

### Community 46 - "snakePPO"
Cohesion: 0.40
Nodes (5): Commands, Documentation, Quickstart, snakePPO, Verification

### Community 48 - "Typography"
Cohesion: 0.67
Nodes (3): Hierarchy, Named Rules, Typography

## Knowledge Gaps
- **1 isolated node(s):** `snakeppo`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 342 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Module map` connect `cli.py` to `train`, `test_runs.py`, `Config`, `runs.py`, `trainer.py`, `TorchSnakeEnv`, `build_agent`, `MetricsLogger`, `builders.py`, `api.py`, `results.py`, `test_server.py`, `main`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 41 inferred relationships involving `Module map` (e.g. with `CNNPolicy` and `PolicyNetwork`) actually correct?**
  _`Module map` has 41 INFERRED edges - model-reasoned connections that need verification._
- **What connects `snakeppo` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `cli.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10588235294117647 - nodes in this community are weakly interconnected._
- **Why does `Architecture` connect `trainer.py` to `cli.py`, `runs.py`, `TorchSnakeEnv`, `architecture.md`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 28 inferred relationships involving `Config` (e.g. with `_experiment_config()` and `_load_model()`) actually correct?**
  _`Config` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Should `train` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._