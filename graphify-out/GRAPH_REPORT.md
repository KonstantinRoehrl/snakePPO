# Graph Report - snakePPO  (2026-11-08)

## Corpus Check
- 96 files · ~60,704 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 2, .css 1, .lock 1)

## Summary
- 1566 nodes · 3691 edges · 108 communities (86 shown, 22 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 445 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2a9089f8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_ppo_functional.py
- train
- ppo
- trainer.py
- test_env.py
- TorchSnakeEnv
- app.js
- verify
- with
- train
- updates
- ppo
- on-device-results.md
- snakeppo
- test_resume.py
- analyze
- add-architecture/SKILL.md
- add-toggle/SKILL.md
- compare-runs/SKILL.md
- run-experiment/SKILL.md
- sweep.py
- server.py
- results.py
- Design System: snakePPO dashboard
- test_server.py
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
- test_curriculum.py
- find_runs
- Module map
- c2-experiments-dashboard/on-device-results.md
- registry.py
- Curriculum
- test_runs.py
- test_recurrent.py
- Components
- renderCompareCharts
- test_viewer.py
- reward
- PolicyNetwork
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
- config.py
- MetricsLogger
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
- pytest
- test_observation.py
- torch
- builders.py
- Config
- env.py
- Any
- Data flow of one update
- total_updates
- c4-architecture-observation-zoo/on-device-results.md

## God Nodes (most connected - your core abstractions)
1. `Module map` - 79 edges
2. `Config` - 62 edges
3. `train()` - 50 edges
4. `main()` - 44 edges
5. `resume()` - 40 edges
6. `ObservationSpec` - 35 edges
7. `PolicyNetwork` - 32 edges
8. `apply_overrides()` - 32 edges
9. `TorchSnakeEnv` - 30 edges
10. `load_config()` - 30 edges

## Surprising Connections (you probably didn't know these)
- `Curriculum` --references--> `validate()`  [INFERRED]
  docs/experiments.md → src/snakeppo/config.py
- `Architectures` --references--> `train()`  [INFERRED]
  docs/experiments.md → src/snakeppo/ppo/trainer.py
- `Module map` --references--> `ResumeError`  [INFERRED]
  docs/architecture.md → src/snakeppo/ppo/trainer.py
- `Module map` --references--> `RunRecord`  [INFERRED]
  docs/architecture.md → src/snakeppo/runs.py
- `Data flow of one update` --references--> `RolloutBuffer`  [INFERRED]
  docs/architecture.md → src/snakeppo/ppo/trainer.py

## Import Cycles
- None detected.

## Communities (108 total, 22 thin omitted)

### Community 0 - "test_ppo_functional.py"
Cohesion: 0.14
Nodes (15): compute_gae(), entropy_coef_at(), explained_variance(), LossOutput, lr_at(), ppo_loss(), _loss(), _rollout() (+7 more)

### Community 1 - "train"
Cohesion: 0.08
Nodes (33): train(), load_checkpoint(), _config(), _curve(), _interrupted_run(), _record_training(), test_a_batching_that_splits_an_env_is_refused_before_any_run_folder(), test_a_recurrent_run_trains_and_checkpoints_its_state() (+25 more)

### Community 2 - "ppo"
Cohesion: 0.12
Nodes (17): ppo, clip_eps, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, gae_lambda, gamma (+9 more)

### Community 3 - "trainer.py"
Cohesion: 0.14
Nodes (11): episode_starts(), _check_resumable(), _env_groups(), _loop(), _make_train_env(), Progress, resume(), RolloutBuffer (+3 more)

### Community 5 - "test_env.py"
Cohesion: 0.21
Nodes (23): _act(), _env(), test_arena_border_kills_like_the_outer_wall(), test_arena_walls_off_everything_outside_a_centred_square(), test_default_observation_names_todays_four_binary_channels(), test_eating_grows_scores_and_respawns_food_off_the_body(), test_end_reasons_name_wall_body_and_starvation(), test_env_uses_the_reward_strategy_it_is_given() (+15 more)

### Community 6 - "TorchSnakeEnv"
Cohesion: 0.12
Nodes (3): Environment contract, TorchSnakeEnv, _LegacySnakeEnv

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

### Community 17 - "test_resume.py"
Cohesion: 0.14
Nodes (21): ResumeError, read_run_info(), update_run_info(), _curve(), _interrupted_run(), _smoke_config(), _strip_observation(), test_a_schema_3_run_still_evaluates_and_resumes_exactly() (+13 more)

### Community 19 - "analyze"
Cohesion: 0.07
Nodes (27): name, permissions, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+19 more)

### Community 24 - "sweep.py"
Cohesion: 0.18
Nodes (9): _sweep(), _Child, _folders_by_seed(), format_outcomes(), group_id(), _outcome(), _report(), run_sweep() (+1 more)

### Community 25 - "server.py"
Cohesion: 0.06
Nodes (18): _serve(), diff_payload(), json_safe(), metrics_payload(), results_payload(), run_detail(), run_summary(), _BadRequest (+10 more)

### Community 26 - "results.py"
Cohesion: 0.07
Nodes (37): Bootstrap, _cells(), config_differences(), _csv(), flatten_config(), _footer(), format_results(), _Group (+29 more)

### Community 27 - "Design System: snakePPO dashboard"
Cohesion: 0.14
Nodes (13): Design System: snakePPO dashboard, Do:, Do's and Don'ts, Don't:, Elevation & Depth, Hierarchy, Layout, Named Rules (+5 more)

### Community 28 - "test_server.py"
Cohesion: 0.11
Nodes (20): server(), _shipped(), _status_and_body(), test_index_and_every_asset_it_references_are_served(), test_modules_import_only_sibling_files(), test_nothing_is_loaded_from_the_internet(), test_the_frontend_files_ship_inside_the_package(), dashboard() (+12 more)

### Community 29 - "charts.js"
Cohesion: 0.12
Nodes (21): aggregate(), average(), card(), CARDS, ChartBoard, chartLayout(), DEFAULT_X, element() (+13 more)

### Community 30 - "cli.py"
Cohesion: 0.19
Nodes (9): _eval(), _experiment_config(), _experiment_options(), _load_model(), _parser(), _play_env_config(), _rewards(), _seed_list() (+1 more)

### Community 31 - "main"
Cohesion: 0.08
Nodes (34): Agent instructions, CI, Commands, Invariants, Layout, Skills, snakePPO, main() (+26 more)

### Community 32 - "On-device checklist — C2 dashboard"
Cohesion: 0.17
Nodes (11): Compare tab, Keyboard and accessibility, Live tab, Loading and frame, On-device checklist — C2 dashboard, Responsive, Results tab, Sidebar (+3 more)

### Community 33 - "evaluate"
Cohesion: 0.17
Nodes (11): EnvConfig, EvalStats, evaluate(), _Scripted, test_a_snake_circling_in_place_starves(), test_a_snake_that_only_goes_straight_always_dies_on_the_wall(), test_end_causes_sum_to_one_and_steps_per_apple_is_steps_over_apples(), test_eval_games_show_the_policy_the_runs_observation() (+3 more)

### Community 34 - "pr-title.yml"
Cohesion: 0.08
Nodes (23): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, GITHUB_TOKEN (+15 more)

### Community 35 - "Product"
Cohesion: 0.18
Nodes (10): Capabilities and Constraints, Evidence on Hand, Operating Context, Platform, Positioning, Product, Product Principles, Product Purpose (+2 more)

### Community 37 - "runs.py"
Cohesion: 0.14
Nodes (11): create_run_dir(), git_sha(), last_write_time(), MetricsChunk, replace_atomically(), resolve_checkpoint(), save_checkpoint(), write_json_atomic() (+3 more)

### Community 38 - "build_agent"
Cohesion: 0.12
Nodes (19): available_agents(), build_agent(), count_parameters(), _agents(), _cnn_pool_input_size(), _shape_cases(), test_board_bound_agents_are_flagged_and_reject_another_size(), test_board_dependent_agents_reject_odd_boards() (+11 more)

### Community 39 - "Module map"
Cohesion: 0.18
Nodes (11): Module map, Add an observation variant, make_env(), make_play_env(), make_policy(), resolve_device(), ObservationConfig, ObservationSpec (+3 more)

### Community 40 - "h"
Cohesion: 0.20
Nodes (20): append(), code(), h(), icon(), incomparableSwitches(), protocolKey(), referenceCandidates(), referenceGuide() (+12 more)

### Community 41 - "networks.py"
Cohesion: 0.07
Nodes (11): AttentionPolicy, CNNPolicy, conv_features(), DeterministicAdaptiveAvgPool2d, MLPPolicy, _pooling_bins(), ResNetPolicy, sinusoidal_positions_2d() (+3 more)

### Community 42 - "Running experiments"
Cohesion: 0.22
Nodes (9): Bookkeeping keys, Comparing runs, Dashboard, Install, Presets and config load order, Resuming a run, Running experiments, Seed sweeps (+1 more)

### Community 43 - "update"
Cohesion: 0.17
Nodes (20): applyTheme(), buildXControl(), chooseReference(), clearProblem(), metricsLoop, noteProblem(), readHash(), redrawCharts() (+12 more)

### Community 44 - "repos"
Cohesion: 0.33
Nodes (5): id, repos, hooks, repo, rev

### Community 46 - "test_curriculum.py"
Cohesion: 0.18
Nodes (13): _curriculum(), _curriculum_run(), _curve(), test_a_curriculum_run_logs_its_stages_and_ends_on_the_full_board(), test_a_resumed_curriculum_run_matches_an_uninterrupted_one_on_cpu(), test_gate_never_fires_before_min_updates(), test_gate_promotes_once_the_window_mean_reaches_its_share_of_capacity(), test_gate_waits_below_the_threshold_and_for_a_full_window() (+5 more)

### Community 47 - "find_runs"
Cohesion: 0.13
Nodes (10): _compare(), _results(), find_runs(), load_run(), summarize_run(), test_summaries_find_runs_and_report_best_and_last_eval(), _args(), test_a_failing_seed_does_not_stop_the_others() (+2 more)

### Community 48 - "Module map"
Cohesion: 0.08
Nodes (35): Add a reward strategy, Extending snakePPO, RewardConfig, available_rewards(), build_reward(), _classic(), ClassicReward, _dilate() (+27 more)

### Community 51 - "registry.py"
Cohesion: 0.22
Nodes (5): ScalableCNNPolicy, AgentSpec, _cnn_of_width(), build(), register()

### Community 52 - "Curriculum"
Cohesion: 0.11
Nodes (11): curriculum_ladder(), CurriculumConfig, CurriculumStage, Curriculum, UpdateResult, _make_curriculum(), _curriculum(), test_capacity_counts_playable_cells_or_apples_to_the_goal() (+3 more)

### Community 53 - "test_runs.py"
Cohesion: 0.20
Nodes (13): read_metrics(), read_metrics_since(), truncate_metrics(), _fake_run(), _lines(), test_atomic_writes_leave_no_temp_files(), test_derive_status_reports_stored_status_and_stalls(), test_metrics_lines_follow_the_contract_and_tensorboard_is_written() (+5 more)

### Community 54 - "test_recurrent.py"
Cohesion: 0.30
Nodes (10): _model(), _random_state(), test_a_recurrent_model_runs_on_any_board_size(), test_a_start_runs_the_step_from_a_zero_state(), test_cells_are_gru_and_lstm(), test_episode_starts_marks_the_step_after_each_done(), test_recurrent_networks_avoid_torchs_adaptive_pool(), test_replaying_a_rollout_reproduces_its_log_probs_and_values() (+2 more)

### Community 55 - "Components"
Cohesion: 0.14
Nodes (14): Badges, Buttons, Cards / Containers, Charts (signature), Chips, Components, Empty and loading states, Inputs / Fields (+6 more)

### Community 56 - "renderCompareCharts"
Cohesion: 0.24
Nodes (8): MetricsStore, movesForward(), compareGroups(), renderCompareCharts(), renderLiveCharts(), runLabel(), readTheme(), seriesColor()

### Community 57 - "test_viewer.py"
Cohesion: 0.14
Nodes (12): On-device checklist — C4 viewer body gradient, body_colour(), Viewer, _grids(), _orders(), _pixel(), test_body_colour_runs_from_bright_at_the_head_to_dark_at_the_tail(), test_body_colours_of_a_snake_are_distinct_and_darken_towards_the_tail() (+4 more)

### Community 58 - "reward"
Cohesion: 0.18
Nodes (11): reward, apple, approach, death, distance_scale, free_space_penalty, hunger_penalty, potential_scale (+3 more)

### Community 59 - "PolicyNetwork"
Cohesion: 0.18
Nodes (4): A recurrent architecture, Add an architecture, PolicyNetwork, SpatialCNNPolicy

### Community 60 - "train"
Cohesion: 0.20
Nodes (10): train, checkpoint_interval, deterministic, device, eval_episodes, eval_interval, group, name (+2 more)

### Community 61 - "PolicyPlayer"
Cohesion: 0.33
Nodes (5): PolicyPlayer, _watch(), test_policy_player_accepts_any_object_with_a_feed_forward_act(), test_policy_player_carries_state_and_resets_only_finished_games(), test_policy_player_plays_a_feed_forward_model_statelessly()

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

### Community 75 - "config.py"
Cohesion: 0.15
Nodes (14): Add a switch, AgentConfig, _coerce(), config_from_dict(), _merge(), _merge_section(), PPOConfig, TrainConfig (+6 more)

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

### Community 98 - "pytest"
Cohesion: 0.20
Nodes (16): ConfigError, test_bad_override_raises(), test_bookkeeping_keys_default_to_today_and_are_validated(), test_curriculum_settings_are_validated_when_it_is_on(), test_defaults_are_the_vanilla_baseline(), test_every_folder_preset_turns_on_one_switch(), test_every_meaningful_observation_is_accepted(), test_every_preset_loads() (+8 more)

### Community 99 - "test_observation.py"
Cohesion: 0.16
Nodes (14): local_view(), _display_grid(), _plane(), _snakes(), test_a_local_view_as_large_as_the_board_still_records_a_different_observation(), test_a_spec_built_directly_rejects_meaningless_values(), test_body_order_runs_evenly_from_the_head_to_the_tail(), test_every_switch_names_its_channels_in_the_fixed_order_and_is_not_binary() (+6 more)

### Community 100 - "torch"
Cohesion: 0.17
Nodes (16): _food_after_resets(), _plane(), _random_actions(), _RecordingEnv, _reference_observation(), _rotated(), test_body_order_replaces_the_body_channel_and_render_body_order_ignores_the_switch(), test_classic_default_env_is_bit_identical_to_the_legacy_env() (+8 more)

### Community 101 - "builders.py"
Cohesion: 0.14
Nodes (8): capture_rng_state(), load_policy(), observation_record(), restore_rng_state(), seed_everything(), describe_record(), test_describe_record_names_the_channels_the_size_and_the_view(), test_the_default_policy_is_built_exactly_as_before()

### Community 102 - "Config"
Cohesion: 0.23
Nodes (17): apply_overrides(), Config, config_to_dict(), load_config(), save_config(), test_baseline_preset_names_every_config_key(), test_baseline_preset_states_exactly_the_code_defaults(), test_curriculum_section_round_trips_through_a_saved_config() (+9 more)

### Community 104 - "Any"
Cohesion: 0.36
Nodes (4): derive_status(), seconds_per_update(), stall_window(), test_stall_window_is_five_median_gaps_but_at_least_a_minute()

### Community 105 - "Data flow of one update"
Cohesion: 0.40
Nodes (4): Architecture, Data flow of one update, Invariants, recurrent_minibatches()

### Community 106 - "total_updates"
Cohesion: 0.40
Nodes (3): total_updates(), _validate_curriculum(), test_total_updates_is_importable_from_config_and_runs()

## Knowledge Gaps
- **1 isolated node(s):** `snakeppo`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 580 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A recurrent architecture` connect `PolicyNetwork` to `evaluate`, `Module map`, `h`, `RecurrentPolicy`, `PolicyPlayer`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Are the 78 inferred relationships involving `Module map` (e.g. with `AttentionPolicy` and `CNNPolicy`) actually correct?**
  _`Module map` has 78 INFERRED edges - model-reasoned connections that need verification._
- **What connects `snakeppo` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_ppo_functional.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._
- **Why does `h()` connect `h` to `update`, `PolicyNetwork`, `renderLive`, `app.js`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `Config` (e.g. with `load_policy()` and `make_policy()`) actually correct?**
  _`Config` has 45 INFERRED edges - model-reasoned connections that need verification._
- **Should `train` be split into smaller, more focused modules?**
  _Cohesion score 0.08176100628930817 - nodes in this community are weakly interconnected._