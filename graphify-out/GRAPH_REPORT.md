# Graph Report - snakePPO  (2026-11-08)

## Corpus Check
- 78 files · ~48,211 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .css 1, .lock 1)

## Summary
- 1266 nodes · 2884 edges · 78 communities (59 shown, 19 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 287 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8a14c709`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cli.py
- train
- ppo
- config.py
- Path
- trainer.py
- TorchSnakeEnv
- app.js
- verify
- with
- train
- updates
- ppo
- on-device-results.md
- snakeppo
- Module map
- analyze
- add-architecture/SKILL.md
- add-toggle/SKILL.md
- compare-runs/SKILL.md
- run-experiment/SKILL.md
- sweep.py
- server.py
- results.py
- Components
- test_server.py
- charts.js
- cli.py
- main
- On-device checklist — C2 dashboard
- evaluate
- pr-title.yml
- Product
- DeterministicAdaptiveAvgPool2d
- resume
- build_agent
- Running experiments
- Design System: snakePPO dashboard
- repos
- Colors
- snakePPO
- find_runs
- rewards.py
- c2-experiments-dashboard/on-device-results.md
- test_curriculum.py
- Config
- runs.py
- snakePPO
- trainer.py
- test_results.py
- Viewer
- reward
- test_viewer.py
- train
- Typography
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
- test_curriculum.py
- Switches
- c3-reward-curriculum/on-device-results.md

## God Nodes (most connected - your core abstractions)
1. `Module map` - 54 edges
2. `Config` - 52 edges
3. `train()` - 41 edges
4. `resume()` - 36 edges
5. `main()` - 33 edges
6. `TorchSnakeEnv` - 29 edges
7. `apply_overrides()` - 28 edges
8. `RunRecord` - 25 edges
9. `load_run()` - 25 edges
10. `load_config()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Curriculum` --references--> `validate()`  [INFERRED]
  docs/experiments.md → src/snakeppo/config.py
- `CI` --references--> `main()`  [INFERRED]
  CLAUDE.md → src/snakeppo/cli.py
- `Module map` --references--> `CNNPolicy`  [INFERRED]
  docs/architecture.md → src/snakeppo/agents/networks.py
- `Module map` --references--> `PolicyNetwork`  [INFERRED]
  docs/architecture.md → src/snakeppo/agents/networks.py
- `Add an architecture` --references--> `PolicyNetwork`  [INFERRED]
  docs/extending.md → src/snakeppo/agents/networks.py

## Import Cycles
- None detected.

## Communities (78 total, 19 thin omitted)

### Community 0 - "cli.py"
Cohesion: 0.21
Nodes (6): CNNPolicy, PolicyNetwork, ResNetPolicy, SpatialCNNPolicy, AgentSpec, register()

### Community 1 - "train"
Cohesion: 0.17
Nodes (18): train(), _Boom, _metric_lines(), _raise_at_update(), _smoke_config(), test_an_aborted_run_records_why_and_re_raises(), test_checkpoint_interval_writes_last_between_evals(), test_constant_schedules_never_move() (+10 more)

### Community 2 - "ppo"
Cohesion: 0.12
Nodes (17): ppo, clip_eps, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, gae_lambda, gamma (+9 more)

### Community 3 - "config.py"
Cohesion: 0.13
Nodes (13): Add a switch, AgentConfig, _coerce(), config_from_dict(), _merge(), _merge_section(), PPOConfig, total_updates() (+5 more)

### Community 4 - "Path"
Cohesion: 0.12
Nodes (13): Invariants, git_sha(), load_checkpoint(), read_metrics(), replace_atomically(), resolve_checkpoint(), save_checkpoint(), truncate_metrics() (+5 more)

### Community 5 - "trainer.py"
Cohesion: 0.06
Nodes (46): Architecture, Data flow of one update, compute_gae(), entropy_coef_at(), explained_variance(), LossOutput, lr_at(), ppo_loss() (+38 more)

### Community 6 - "TorchSnakeEnv"
Cohesion: 0.13
Nodes (4): Environment contract, Add an observation variant (C4), TorchSnakeEnv, _LegacySnakeEnv

### Community 7 - "app.js"
Cohesion: 0.06
Nodes (89): ApiError, fetchDiff(), fetchMetrics(), fetchResults(), fetchRun(), fetchRuns(), getJson(), MetricsStore (+81 more)

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

### Community 17 - "Module map"
Cohesion: 0.17
Nodes (8): Module map, capture_rng_state(), make_env(), make_play_env(), resolve_device(), restore_rng_state(), EnvConfig, EvalStats

### Community 19 - "analyze"
Cohesion: 0.07
Nodes (27): name, permissions, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+19 more)

### Community 24 - "sweep.py"
Cohesion: 0.16
Nodes (9): load_run(), _Child, _folders_by_seed(), format_outcomes(), group_id(), _outcome(), _report(), run_sweep() (+1 more)

### Community 25 - "server.py"
Cohesion: 0.06
Nodes (17): _serve(), diff_payload(), json_safe(), results_payload(), run_detail(), run_summary(), _BadRequest, DashboardServer (+9 more)

### Community 26 - "results.py"
Cohesion: 0.08
Nodes (25): Bootstrap, _cells(), config_differences(), _csv(), flatten_config(), _footer(), format_results(), _Group (+17 more)

### Community 27 - "Components"
Cohesion: 0.14
Nodes (14): Badges, Buttons, Cards / Containers, Charts (signature), Chips, Components, Empty and loading states, Inputs / Fields (+6 more)

### Community 28 - "test_server.py"
Cohesion: 0.10
Nodes (21): server(), _shipped(), _status_and_body(), test_index_and_every_asset_it_references_are_served(), test_modules_import_only_sibling_files(), test_nothing_is_loaded_from_the_internet(), test_the_frontend_files_ship_inside_the_package(), dashboard() (+13 more)

### Community 29 - "charts.js"
Cohesion: 0.12
Nodes (21): aggregate(), average(), card(), CARDS, ChartBoard, chartLayout(), DEFAULT_X, element() (+13 more)

### Community 30 - "cli.py"
Cohesion: 0.18
Nodes (13): _compare(), _eval(), _experiment_config(), _experiment_options(), _load_model(), _parser(), _rewards(), _seed_list() (+5 more)

### Community 31 - "main"
Cohesion: 0.17
Nodes (17): main(), test_agent_flag_is_taken_as_text_not_parsed_as_yaml(), test_agents_lists_every_architecture(), test_bad_agent_kwargs_are_reported_without_a_traceback_or_run_folder(), test_compare_without_runs_says_so(), test_config_errors_are_reported_without_a_traceback(), test_help_exits_cleanly(), test_name_flag_is_taken_as_text_not_parsed_as_yaml() (+9 more)

### Community 32 - "On-device checklist — C2 dashboard"
Cohesion: 0.17
Nodes (11): Compare tab, Keyboard and accessibility, Live tab, Loading and frame, On-device checklist — C2 dashboard, Responsive, Results tab, Sidebar (+3 more)

### Community 33 - "evaluate"
Cohesion: 0.22
Nodes (6): evaluate(), _Scripted, test_a_snake_circling_in_place_starves(), test_a_snake_that_only_goes_straight_always_dies_on_the_wall(), test_end_causes_sum_to_one_and_steps_per_apple_is_steps_over_apples(), test_evaluate_counts_one_finished_episode_per_game_and_is_reproducible()

### Community 34 - "pr-title.yml"
Cohesion: 0.08
Nodes (23): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, GITHUB_TOKEN (+15 more)

### Community 35 - "Product"
Cohesion: 0.18
Nodes (10): Capabilities and Constraints, Evidence on Hand, Operating Context, Platform, Positioning, Product, Product Principles, Product Purpose (+2 more)

### Community 40 - "resume"
Cohesion: 0.22
Nodes (15): ResumeError, read_run_info(), _curve(), _interrupted_run(), _smoke_config(), test_cli_group_flag_is_recorded_as_text(), test_cli_resume_takes_no_other_option_and_reports_refusals(), test_resume_continues_a_schema_2_run_exactly_and_raises_it_to_schema_3() (+7 more)

### Community 41 - "build_agent"
Cohesion: 0.15
Nodes (14): Add an architecture, available_agents(), build_agent(), count_parameters(), _agents(), _cnn_pool_input_size(), test_board_dependent_agents_reject_odd_boards(), test_cnn_accepts_odd_boards() (+6 more)

### Community 42 - "Running experiments"
Cohesion: 0.17
Nodes (12): Bookkeeping keys, Checkpoints, Comparing runs, Dashboard, Install, `metrics.jsonl`, Presets and config load order, Resuming a run (+4 more)

### Community 43 - "Design System: snakePPO dashboard"
Cohesion: 0.18
Nodes (10): Design System: snakePPO dashboard, Do:, Do's and Don'ts, Don't:, Elevation & Depth, Layout, Named Rules, Overview (+2 more)

### Community 44 - "repos"
Cohesion: 0.33
Nodes (5): id, repos, hooks, repo, rev

### Community 45 - "Colors"
Cohesion: 0.33
Nodes (6): Colors, Data series, Named Rules, Neutral, Primary, State

### Community 46 - "snakePPO"
Cohesion: 0.28
Nodes (5): Commands, Documentation, Quickstart, snakePPO, Verification

### Community 47 - "find_runs"
Cohesion: 0.17
Nodes (6): _results(), find_runs(), _args(), test_a_failing_seed_does_not_stop_the_others(), test_bad_sweep_options_fail_before_any_training(), test_sweep_trains_every_seed_into_one_group()

### Community 48 - "rewards.py"
Cohesion: 0.07
Nodes (34): Add a reward strategy, Extending snakePPO, RewardConfig, build_reward(), _classic(), ClassicReward, _dilate(), DistanceReward (+26 more)

### Community 51 - "test_curriculum.py"
Cohesion: 0.11
Nodes (11): curriculum_ladder(), CurriculumConfig, CurriculumStage, Curriculum, UpdateResult, _make_curriculum(), _curriculum(), test_capacity_counts_playable_cells_or_apples_to_the_goal() (+3 more)

### Community 52 - "Config"
Cohesion: 0.15
Nodes (29): apply_overrides(), Config, config_to_dict(), ConfigError, load_config(), save_config(), test_bad_override_raises(), test_baseline_preset_names_every_config_key() (+21 more)

### Community 53 - "runs.py"
Cohesion: 0.11
Nodes (20): metrics_payload(), create_run_dir(), derive_status(), MetricsChunk, read_metrics_since(), seconds_per_update(), stall_window(), _fake_run() (+12 more)

### Community 54 - "snakePPO"
Cohesion: 0.22
Nodes (7): Agent instructions, CI, Commands, Invariants, Layout, Skills, snakePPO

### Community 55 - "trainer.py"
Cohesion: 0.17
Nodes (9): seed_everything(), _check_resumable(), _make_train_env(), Progress, resume(), _run(), _Session, last_write_time() (+1 more)

### Community 56 - "test_results.py"
Cohesion: 0.33
Nodes (11): results_table(), _group(), _row(), _run(), test_a_reduced_budget_group_needs_a_baseline_at_its_budget(), test_a_selector_and_its_own_keys_count_as_one_switch(), test_a_switch_group_is_judged_against_the_newest_baseline_at_its_budget(), test_forced_baseline_is_every_row_s_reference() (+3 more)

### Community 58 - "reward"
Cohesion: 0.18
Nodes (11): reward, apple, approach, death, distance_scale, free_space_penalty, hunger_penalty, potential_scale (+3 more)

### Community 59 - "test_viewer.py"
Cohesion: 0.36
Nodes (5): _grids(), test_closing_the_window_stops_drawing_cleanly(), test_draw_keeps_one_font_and_reports_open(), test_importing_the_viewer_does_not_import_pygame(), viewer()

### Community 60 - "train"
Cohesion: 0.20
Nodes (10): train, checkpoint_interval, deterministic, device, eval_episodes, eval_interval, group, name (+2 more)

### Community 61 - "Typography"
Cohesion: 0.67
Nodes (3): Hierarchy, Named Rules, Typography

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

### Community 73 - "test_curriculum.py"
Cohesion: 0.20
Nodes (13): _curriculum(), _curriculum_run(), _curve(), test_a_curriculum_run_logs_its_stages_and_ends_on_the_full_board(), test_a_resumed_curriculum_run_matches_an_uninterrupted_one_on_cpu(), test_gate_never_fires_before_min_updates(), test_gate_promotes_once_the_window_mean_reaches_its_share_of_capacity(), test_gate_waits_below_the_threshold_and_for_a_full_window() (+5 more)

### Community 76 - "Switches"
Cohesion: 0.67
Nodes (3): Curriculum, Reward strategies, Switches

## Knowledge Gaps
- **1 isolated node(s):** `snakeppo`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 477 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Module map` connect `Module map` to `cli.py`, `train`, `config.py`, `Path`, `trainer.py`, `TorchSnakeEnv`, `sweep.py`, `server.py`, `results.py`, `cli.py`, `evaluate`, `resume`, `build_agent`, `find_runs`, `rewards.py`, `test_curriculum.py`, `runs.py`, `trainer.py`, `test_results.py`, `test_viewer.py`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 53 inferred relationships involving `Module map` (e.g. with `CNNPolicy` and `PolicyNetwork`) actually correct?**
  _`Module map` has 53 INFERRED edges - model-reasoned connections that need verification._
- **What connects `snakeppo` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ppo` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Why does `Architecture` connect `trainer.py` to `Module map`, `Path`, `snakePPO`, `TorchSnakeEnv`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 36 inferred relationships involving `Config` (e.g. with `_experiment_config()` and `_load_model()`) actually correct?**
  _`Config` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Should `config.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12615384615384614 - nodes in this community are weakly interconnected._