# Graph Report - snakePPO  (2026-11-19)

## Corpus Check
- 97 files · ~61,435 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .css 1, .lock 1)

## Summary
- 1574 nodes · 3685 edges · 104 communities (80 shown, 24 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 447 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `00fb8699`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_ppo_functional.py
- train
- ppo
- trainer.py
- torch
- TorchSnakeEnv
- app.js
- verify
- with
- train
- updates
- ppo
- on-device-results.md
- snakeppo
- test_cli_ctrl_c_exits_130
- analyze
- add-architecture/SKILL.md
- add-toggle/SKILL.md
- compare-runs/SKILL.md
- run-experiment/SKILL.md
- sweep.py
- server.py
- results.py
- Design System: snakePPO dashboard
- api.py
- charts.js
- cli.py
- main
- On-device checklist — C2 dashboard
- evaluate
- pr-title.yml
- Product
- runs.py
- build_agent
- Module map
- h
- networks.py
- Running experiments
- update
- repos
- RecurrentPolicy
- test_recurrent_training.py
- find_runs
- Module map
- c2-experiments-dashboard/on-device-results.md
- registry.py
- pytest
- ._route
- test_recurrent.py
- Components
- renderCompareCharts
- reward
- test_results.py
- train
- PolicyPlayer
- baseline.yaml
- curriculum
- On-device checklist — C3 reward strategies and curriculum (dashboard)
- board.yaml
- both.yaml
- length.yaml
- distance.yaml
- free_space.yaml
- hunger.yaml
- potential.yaml
- sparse.yaml
- api.js
- _Group
- snakePPO
- c3-reward-curriculum/on-device-results.md
- snakePPO
- agent
- observation
- Colors
- attention.yaml
- cnn-medium.yaml
- cnn-small.yaml
- cnn-tiny.yaml
- mlp.yaml
- recurrent.yaml
- body-order.yaml
- observation/hunger.yaml
- local-view.yaml
- tail.yaml
- Switches
- renderLive
- Run folders
- C4 sanity report
- json_safe
- capture_rng_state
- Data flow of one update
- c4-architecture-observation-zoo/on-device-results.md

## God Nodes (most connected - your core abstractions)
1. `Module map` - 79 edges
2. `Config` - 62 edges
3. `train()` - 44 edges
4. `main()` - 44 edges
5. `resume()` - 40 edges
6. `ObservationSpec` - 35 edges
7. `PolicyNetwork` - 32 edges
8. `apply_overrides()` - 32 edges
9. `load_config()` - 30 edges
10. `_env()` - 30 edges

## Surprising Connections (you probably didn't know these)
- `Curriculum` --references--> `validate()`  [INFERRED]
  docs/experiments.md → src/snakeppo/config.py
- `Architectures` --references--> `train()`  [INFERRED]
  docs/experiments.md → src/snakeppo/ppo/trainer.py
- `CI` --references--> `main()`  [INFERRED]
  CLAUDE.md → src/snakeppo/cli.py
- `Module map` --references--> `PolicyNetwork`  [INFERRED]
  docs/architecture.md → src/snakeppo/agents/networks.py
- `Module map` --references--> `DeterministicAdaptiveAvgPool2d`  [INFERRED]
  docs/architecture.md → src/snakeppo/agents/networks.py

## Import Cycles
- None detected.

## Communities (104 total, 24 thin omitted)

### Community 0 - "test_ppo_functional.py"
Cohesion: 0.15
Nodes (14): compute_gae(), entropy_coef_at(), explained_variance(), LossOutput, ppo_loss(), _loss(), _rollout(), test_explained_variance() (+6 more)

### Community 1 - "train"
Cohesion: 0.08
Nodes (43): _check_resumable(), resume(), ResumeError, train(), load_checkpoint(), read_run_info(), update_run_info(), _curve() (+35 more)

### Community 2 - "ppo"
Cohesion: 0.12
Nodes (17): ppo, clip_eps, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, gae_lambda, gamma (+9 more)

### Community 3 - "trainer.py"
Cohesion: 0.15
Nodes (9): lr_at(), _env_groups(), _loop(), _make_train_env(), Progress, RolloutBuffer, _run(), _Session (+1 more)

### Community 5 - "torch"
Cohesion: 0.06
Nodes (64): body_order_grid(), local_view(), tail_grid(), body_colour(), _act(), _env(), _food_after_resets(), _plane() (+56 more)

### Community 6 - "TorchSnakeEnv"
Cohesion: 0.08
Nodes (7): Environment contract, On-device checklist — C4 viewer body gradient, TorchSnakeEnv, Viewer, server(), _LegacySnakeEnv, viewer()

### Community 7 - "app.js"
Cohesion: 0.13
Nodes (29): badge(), byCreatedDesc(), clearFilters(), compact, CURVE_SWITCHES, distinctSwitches(), duration(), every() (+21 more)

### Community 8 - "verify"
Cohesion: 0.07
Nodes (29): concurrency, cancel-in-progress, group, run, jobs, verify, os, name (+21 more)

### Community 9 - "with"
Cohesion: 0.05
Nodes (43): concurrency, cancel-in-progress, group, if, name, runs-on, steps, timeout-minutes (+35 more)

### Community 10 - "train"
Cohesion: 0.12
Nodes (16): agent, name, env, board_size, num_envs, starvation_steps, ppo, minibatch_size (+8 more)

### Community 11 - "updates"
Cohesion: 0.14
Nodes (13): default-days, patterns, github-actions, python-dependencies, patterns, interval, updates, cooldown (+5 more)

### Community 12 - "ppo"
Cohesion: 0.20
Nodes (9): ppo, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, lr_schedule, target_kl, train (+1 more)

### Community 19 - "analyze"
Cohesion: 0.07
Nodes (27): name, permissions, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+19 more)

### Community 24 - "sweep.py"
Cohesion: 0.18
Nodes (8): _Child, _folders_by_seed(), format_outcomes(), group_id(), _outcome(), _report(), run_sweep(), SeedOutcome

### Community 25 - "server.py"
Cohesion: 0.12
Nodes (6): DashboardServer, make_server(), plotly_js(), serve(), static_dir(), test_a_busy_port_is_reported_with_the_port_flag()

### Community 26 - "results.py"
Cohesion: 0.12
Nodes (17): Bootstrap, _cells(), _csv(), _footer(), format_results(), GroupRow, _number(), paired_bootstrap() (+9 more)

### Community 27 - "Design System: snakePPO dashboard"
Cohesion: 0.14
Nodes (13): Design System: snakePPO dashboard, Do:, Do's and Don'ts, Don't:, Elevation & Depth, Hierarchy, Layout, Named Rules (+5 more)

### Community 28 - "api.py"
Cohesion: 0.14
Nodes (11): diff_payload(), metrics_payload(), results_payload(), run_detail(), run_summary(), config_differences(), flatten_config(), learning_config() (+3 more)

### Community 29 - "charts.js"
Cohesion: 0.12
Nodes (21): aggregate(), average(), card(), CARDS, ChartBoard, chartLayout(), DEFAULT_X, element() (+13 more)

### Community 30 - "cli.py"
Cohesion: 0.13
Nodes (17): resolve_device(), _compare(), _eval(), _experiment_config(), _experiment_options(), _load_model(), _parser(), _play_env_config() (+9 more)

### Community 31 - "main"
Cohesion: 0.12
Nodes (27): main(), _edit_saved_config(), _smoke_run(), test_a_local_view_with_an_even_board_network_fails_before_any_run_folder(), test_agent_flag_is_taken_as_text_not_parsed_as_yaml(), test_agents_lists_every_architecture_and_whether_it_plays_any_board_size(), test_bad_agent_kwargs_are_reported_without_a_traceback_or_run_folder(), test_compare_without_runs_says_so() (+19 more)

### Community 32 - "On-device checklist — C2 dashboard"
Cohesion: 0.17
Nodes (11): Compare tab, Keyboard and accessibility, Live tab, Loading and frame, On-device checklist — C2 dashboard, Responsive, Results tab, Sidebar (+3 more)

### Community 33 - "evaluate"
Cohesion: 0.13
Nodes (13): make_play_env(), EnvConfig, EvalStats, evaluate(), SparseReward, _Scripted, test_a_snake_circling_in_place_starves(), test_a_snake_that_only_goes_straight_always_dies_on_the_wall() (+5 more)

### Community 34 - "pr-title.yml"
Cohesion: 0.08
Nodes (23): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, GITHUB_TOKEN (+15 more)

### Community 35 - "Product"
Cohesion: 0.18
Nodes (10): Capabilities and Constraints, Evidence on Hand, Operating Context, Platform, Positioning, Product, Product Principles, Product Purpose (+2 more)

### Community 37 - "runs.py"
Cohesion: 0.05
Nodes (47): create_run_dir(), derive_status(), git_sha(), last_write_time(), load_run(), MetricsChunk, MetricsLogger, read_metrics() (+39 more)

### Community 38 - "build_agent"
Cohesion: 0.12
Nodes (19): available_agents(), build_agent(), count_parameters(), _agents(), _cnn_pool_input_size(), _shape_cases(), test_board_bound_agents_are_flagged_and_reject_another_size(), test_board_dependent_agents_reject_odd_boards() (+11 more)

### Community 39 - "Module map"
Cohesion: 0.14
Nodes (15): Module map, Add an observation variant, load_policy(), make_env(), make_policy(), observation_record(), ObservationConfig, describe_record() (+7 more)

### Community 40 - "h"
Cohesion: 0.20
Nodes (20): append(), code(), h(), icon(), incomparableSwitches(), protocolKey(), referenceCandidates(), referenceGuide() (+12 more)

### Community 41 - "networks.py"
Cohesion: 0.07
Nodes (10): AttentionPolicy, CNNPolicy, DeterministicAdaptiveAvgPool2d, MLPPolicy, _pooling_bins(), ResNetPolicy, sinusoidal_positions_2d(), SpatialCNNPolicy (+2 more)

### Community 42 - "Running experiments"
Cohesion: 0.22
Nodes (9): Bookkeeping keys, Comparing runs, Dashboard, Install, Presets and config load order, Resuming a run, Running experiments, Seed sweeps (+1 more)

### Community 43 - "update"
Cohesion: 0.17
Nodes (20): applyTheme(), buildXControl(), chooseReference(), clearProblem(), metricsLoop, noteProblem(), readHash(), redrawCharts() (+12 more)

### Community 44 - "repos"
Cohesion: 0.33
Nodes (5): id, repos, hooks, repo, rev

### Community 45 - "RecurrentPolicy"
Cohesion: 0.10
Nodes (7): A recurrent architecture, Add an architecture, conv_features(), PolicyNetwork, RecurrentCNNPolicy, RecurrentPolicy, test_conv_features_pool_to_a_fixed_feature_count()

### Community 46 - "test_recurrent_training.py"
Cohesion: 0.07
Nodes (30): _curriculum(), _curriculum_run(), _curve(), test_a_curriculum_run_logs_its_stages_and_ends_on_the_full_board(), test_a_resumed_curriculum_run_matches_an_uninterrupted_one_on_cpu(), test_gate_never_fires_before_min_updates(), test_gate_promotes_once_the_window_mean_reaches_its_share_of_capacity(), test_gate_waits_below_the_threshold_and_for_a_full_window() (+22 more)

### Community 47 - "find_runs"
Cohesion: 0.21
Nodes (4): _args(), test_a_failing_seed_does_not_stop_the_others(), test_bad_sweep_options_fail_before_any_training(), test_sweep_trains_every_seed_into_one_group()

### Community 48 - "Module map"
Cohesion: 0.08
Nodes (33): Add a reward strategy, RewardConfig, available_rewards(), build_reward(), _classic(), ClassicReward, _dilate(), DistanceReward (+25 more)

### Community 51 - "registry.py"
Cohesion: 0.24
Nodes (5): ScalableCNNPolicy, AgentSpec, _cnn_of_width(), build(), register()

### Community 52 - "pytest"
Cohesion: 0.05
Nodes (60): Add a switch, Extending snakePPO, AgentConfig, apply_overrides(), _coerce(), Config, config_from_dict(), config_to_dict() (+52 more)

### Community 53 - "._route"
Cohesion: 0.20
Nodes (3): _BadRequest, _Handler, _NotFound

### Community 54 - "test_recurrent.py"
Cohesion: 0.25
Nodes (11): episode_starts(), _model(), _random_state(), test_a_recurrent_model_runs_on_any_board_size(), test_a_start_runs_the_step_from_a_zero_state(), test_cells_are_gru_and_lstm(), test_episode_starts_marks_the_step_after_each_done(), test_recurrent_networks_avoid_torchs_adaptive_pool() (+3 more)

### Community 55 - "Components"
Cohesion: 0.14
Nodes (14): Badges, Buttons, Cards / Containers, Charts (signature), Chips, Components, Empty and loading states, Inputs / Fields (+6 more)

### Community 56 - "renderCompareCharts"
Cohesion: 0.24
Nodes (8): MetricsStore, movesForward(), compareGroups(), renderCompareCharts(), renderLiveCharts(), runLabel(), readTheme(), seriesColor()

### Community 58 - "reward"
Cohesion: 0.18
Nodes (11): reward, apple, approach, death, distance_scale, free_space_penalty, hunger_penalty, potential_scale (+3 more)

### Community 59 - "test_results.py"
Cohesion: 0.25
Nodes (14): results_table(), switch_unit(), _group(), _row(), _run(), test_a_reduced_budget_group_needs_a_baseline_at_its_budget(), test_a_selector_and_its_own_keys_count_as_one_switch(), test_a_switch_group_is_judged_against_the_newest_baseline_at_its_budget() (+6 more)

### Community 60 - "train"
Cohesion: 0.20
Nodes (10): train, checkpoint_interval, deterministic, device, eval_episodes, eval_interval, group, name (+2 more)

### Community 61 - "PolicyPlayer"
Cohesion: 0.26
Nodes (4): PolicyPlayer, test_policy_player_accepts_any_object_with_a_feed_forward_act(), test_policy_player_carries_state_and_resets_only_finished_games(), test_policy_player_plays_a_feed_forward_model_statelessly()

### Community 62 - "baseline.yaml"
Cohesion: 0.22
Nodes (8): agent, kwargs, name, env, board_size, num_envs, seed, starvation_steps

### Community 63 - "curriculum"
Cohesion: 0.22
Nodes (9): curriculum, grow, max_updates, min_updates, promote_fraction, promote_window, stages, start_board (+1 more)

### Community 64 - "On-device checklist — C3 reward strategies and curriculum (dashboard)"
Cohesion: 0.33
Nodes (5): Compare notice, Evaluation and Curriculum cards, On-device checklist — C3 reward strategies and curriculum (dashboard), Score meaning, Themes, widths, accessibility

### Community 65 - "board.yaml"
Cohesion: 0.40
Nodes (4): curriculum, grow, train, name

### Community 66 - "both.yaml"
Cohesion: 0.40
Nodes (4): curriculum, grow, train, name

### Community 67 - "length.yaml"
Cohesion: 0.40
Nodes (4): curriculum, grow, train, name

### Community 68 - "distance.yaml"
Cohesion: 0.40
Nodes (4): reward, strategy, train, name

### Community 69 - "free_space.yaml"
Cohesion: 0.40
Nodes (4): reward, strategy, train, name

### Community 70 - "hunger.yaml"
Cohesion: 0.40
Nodes (4): reward, strategy, train, name

### Community 71 - "potential.yaml"
Cohesion: 0.40
Nodes (4): reward, strategy, train, name

### Community 72 - "sparse.yaml"
Cohesion: 0.40
Nodes (4): reward, strategy, train, name

### Community 73 - "api.js"
Cohesion: 0.33
Nodes (7): ApiError, fetchDiff(), fetchMetrics(), fetchResults(), fetchRun(), fetchRuns(), getJson()

### Community 76 - "snakePPO"
Cohesion: 0.22
Nodes (7): Agent instructions, CI, Commands, Invariants, Layout, Skills, snakePPO

### Community 78 - "snakePPO"
Cohesion: 0.28
Nodes (5): Commands, Documentation, Quickstart, snakePPO, Verification

### Community 79 - "agent"
Cohesion: 0.29
Nodes (6): agent, kwargs, name, cell, train, name

### Community 80 - "observation"
Cohesion: 0.33
Nodes (6): observation, body, hunger, tail, view, view_size

### Community 81 - "Colors"
Cohesion: 0.33
Nodes (6): Colors, Data series, Named Rules, Neutral, Primary, State

### Community 82 - "attention.yaml"
Cohesion: 0.40
Nodes (4): agent, name, train, name

### Community 83 - "cnn-medium.yaml"
Cohesion: 0.40
Nodes (4): agent, name, train, name

### Community 84 - "cnn-small.yaml"
Cohesion: 0.40
Nodes (4): agent, name, train, name

### Community 85 - "cnn-tiny.yaml"
Cohesion: 0.40
Nodes (4): agent, name, train, name

### Community 86 - "mlp.yaml"
Cohesion: 0.40
Nodes (4): agent, name, train, name

### Community 87 - "recurrent.yaml"
Cohesion: 0.40
Nodes (4): agent, name, train, name

### Community 88 - "body-order.yaml"
Cohesion: 0.40
Nodes (4): observation, body, train, name

### Community 89 - "observation/hunger.yaml"
Cohesion: 0.40
Nodes (4): observation, hunger, train, name

### Community 90 - "local-view.yaml"
Cohesion: 0.40
Nodes (4): observation, view, train, name

### Community 91 - "tail.yaml"
Cohesion: 0.40
Nodes (4): observation, tail, train, name

### Community 92 - "Switches"
Cohesion: 0.40
Nodes (5): Architectures, Curriculum, Observation, Reward strategies, Switches

### Community 93 - "renderLive"
Cohesion: 0.60
Nodes (5): compareRuns(), displayedRunIds(), knownRuns(), liveRun(), renderLive()

### Community 94 - "Run folders"
Cohesion: 0.50
Nodes (4): Checkpoints, Evaluating on another board size, `metrics.jsonl`, Run folders

### Community 97 - "C4 sanity report"
Cohesion: 0.29
Nodes (6): Body order (3 seeds against a 3-seed baseline), C4 sanity report, Caveats, Result, Training runs, Transfer (trained on 16x16, evaluated greedy over 64 episodes, best checkpoint)

### Community 101 - "capture_rng_state"
Cohesion: 0.25
Nodes (3): capture_rng_state(), restore_rng_state(), seed_everything()

### Community 105 - "Data flow of one update"
Cohesion: 0.40
Nodes (4): Architecture, Data flow of one update, Invariants, recurrent_minibatches()

## Knowledge Gaps
- **1 isolated node(s):** `snakeppo`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 585 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Module map` connect `Module map` to `test_ppo_functional.py`, `train`, `trainer.py`, `torch`, `TorchSnakeEnv`, `sweep.py`, `server.py`, `results.py`, `api.py`, `cli.py`, `evaluate`, `runs.py`, `build_agent`, `networks.py`, `RecurrentPolicy`, `Module map`, `registry.py`, `pytest`, `test_recurrent.py`, `test_results.py`, `PolicyPlayer`, `capture_rng_state`, `Data flow of one update`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Are the 78 inferred relationships involving `Module map` (e.g. with `AttentionPolicy` and `CNNPolicy`) actually correct?**
  _`Module map` has 78 INFERRED edges - model-reasoned connections that need verification._
- **What connects `snakeppo` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train` be split into smaller, more focused modules?**
  _Cohesion score 0.0773892773892774 - nodes in this community are weakly interconnected._
- **Why does `A recurrent architecture` connect `RecurrentPolicy` to `h`, `evaluate`, `PolicyPlayer`, `Module map`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `Config` (e.g. with `load_policy()` and `make_policy()`) actually correct?**
  _`Config` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Should `ppo` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._