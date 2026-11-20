# Graph Report - snakePPO  (2026-11-20)

## Corpus Check
- 110 files · ~90,472 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: .css 3, (none) 2, .lock 1)

## Summary
- 2156 nodes · 5430 edges · 130 communities (105 shown, 25 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 607 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e710507b`
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
- launches.py
- on-device-results.md
- snakeppo
- test_cli_ctrl_c_exits_130
- analyze
- add-architecture/SKILL.md
- add-toggle/SKILL.md
- compare-runs/SKILL.md
- run-experiment/SKILL.md
- load_run
- server.py
- results.py
- Design System: snakePPO dashboard
- runs.py
- charts.js
- cli.py
- main
- On-device checklist — C2 dashboard
- evaluate
- pr-title.yml
- Product
- test_runs.py
- build_agent
- ObservationSpec
- h
- PolicyNetwork
- Running experiments
- test_replay.py
- repos
- PolicyPlayer
- load_checkpoint
- test_sweep.py
- Module map
- c2-experiments-dashboard/on-device-results.md
- server
- Module map
- Config
- ._route
- test_recurrent.py
- Components
- renderCompareCharts
- reward
- test_results.py
- train
- mountLaunch
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
- test_launches.py
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
- mountReplay
- C4 sanity report
- config.py
- test_server.py
- Curriculum
- test_stop.py
- load_config
- processes.py
- resume
- test_server_replay.py
- json
- c4-architecture-observation-zoo/on-device-results.md
- read_metrics
- test_observation.py
- pytest
- LaunchRequest
- replay.js
- launch_dir
- play_env_config
- _smoke_run
- parametrize
- _FakeSpawn
- _run
- served_runs_dir
- test_eval_render_serves_the_runs_folder_when_no_dashboard_answers
- bench
- restore_deterministic_settings
- c5-web-app-main-ui/on-device-checklist.md

## God Nodes (most connected - your core abstractions)
1. `Module map` - 105 edges
2. `Config` - 68 edges
3. `main()` - 56 edges
4. `train()` - 54 edges
5. `resume()` - 49 edges
6. `load_config()` - 40 edges
7. `mountReplay()` - 37 edges
8. `_Handler` - 36 edges
9. `apply_overrides()` - 34 edges
10. `load_run()` - 32 edges

## Surprising Connections (you probably didn't know these)
- `CI` --references--> `main()`  [INFERRED]
  CLAUDE.md → src/snakeppo/cli.py
- `Curriculum` --references--> `validate()`  [INFERRED]
  docs/experiments.md → src/snakeppo/config.py
- `Architectures` --references--> `train()`  [INFERRED]
  docs/experiments.md → src/snakeppo/ppo/trainer.py
- `Dashboard routes` --references--> `resume()`  [INFERRED]
  docs/architecture.md → src/snakeppo/ppo/trainer.py
- `Module map` --references--> `resolve_device()`  [INFERRED]
  docs/architecture.md → src/snakeppo/builders.py

## Import Cycles
- None detected.

## Communities (130 total, 25 thin omitted)

### Community 0 - "test_ppo_functional.py"
Cohesion: 0.15
Nodes (14): compute_gae(), entropy_coef_at(), explained_variance(), LossOutput, ppo_loss(), _loss(), _rollout(), test_explained_variance() (+6 more)

### Community 1 - "train"
Cohesion: 0.16
Nodes (20): train(), _Boom, _metric_lines(), _raise_at_update(), _smoke_config(), test_an_aborted_run_records_why_and_re_raises(), test_checkpoint_interval_writes_last_between_evals(), test_constant_schedules_never_move() (+12 more)

### Community 2 - "ppo"
Cohesion: 0.12
Nodes (17): ppo, clip_eps, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, gae_lambda, gamma (+9 more)

### Community 3 - "trainer.py"
Cohesion: 0.07
Nodes (21): Architecture, Dashboard routes, Data flow of one update, Invariants, capture_rng_state(), load_policy(), observation_record(), recurrent_minibatches() (+13 more)

### Community 5 - "test_env.py"
Cohesion: 0.15
Nodes (31): _act(), _env(), _food_after_resets(), _random_actions(), _rotated(), test_arena_border_kills_like_the_outer_wall(), test_arena_walls_off_everything_outside_a_centred_square(), test_body_order_replaces_the_body_channel_and_render_body_order_ignores_the_switch() (+23 more)

### Community 6 - "TorchSnakeEnv"
Cohesion: 0.11
Nodes (4): Environment contract, On-device checklist — C4 viewer body gradient, TorchSnakeEnv, _LegacySnakeEnv

### Community 7 - "app.js"
Cohesion: 0.07
Nodes (51): APP_HASH_KEYS, append(), buildXControl(), byCreatedDesc(), CHART_TABS, chooseReference(), clearFilters(), clearProblem() (+43 more)

### Community 8 - "verify"
Cohesion: 0.07
Nodes (29): concurrency, cancel-in-progress, group, run, jobs, verify, os, name (+21 more)

### Community 9 - "with"
Cohesion: 0.05
Nodes (43): concurrency, cancel-in-progress, group, if, name, runs-on, steps, timeout-minutes (+35 more)

### Community 10 - "train"
Cohesion: 0.11
Nodes (17): agent, name, env, board_size, num_envs, starvation_steps, ppo, minibatch_size (+9 more)

### Community 11 - "updates"
Cohesion: 0.14
Nodes (13): default-days, patterns, github-actions, python-dependencies, patterns, interval, updates, cooldown (+5 more)

### Community 12 - "ppo"
Cohesion: 0.18
Nodes (10): ppo, clip_value_loss, entropy_coef, entropy_coef_final, entropy_schedule, lr_schedule, target_kl, train (+2 more)

### Community 13 - "launches.py"
Cohesion: 0.11
Nodes (17): _command(), _command_path(), _entry(), _is_whole(), _live_launch(), _load_launch(), _load_runs(), _log_tail() (+9 more)

### Community 19 - "analyze"
Cohesion: 0.07
Nodes (27): name, permissions, runs-on, steps, timeout-minutes, concurrency, cancel-in-progress, group (+19 more)

### Community 24 - "load_run"
Cohesion: 0.11
Nodes (15): _results(), _sweep(), child_creationflags(), load_run(), _Child, _folders_by_seed(), format_outcomes(), group_id() (+7 more)

### Community 25 - "server.py"
Cohesion: 0.08
Nodes (12): DashboardServer, _Forbidden, _HttpError, make_server(), plotly_js(), serve(), static_dir(), _TooLarge (+4 more)

### Community 26 - "results.py"
Cohesion: 0.12
Nodes (17): preset_description(), run_description(), Bootstrap, _cells(), _csv(), _footer(), format_results(), GroupRow (+9 more)

### Community 27 - "Design System: snakePPO dashboard"
Cohesion: 0.15
Nodes (12): Design System: snakePPO dashboard, Do's and Don'ts, Don't:, Elevation & Depth, Hierarchy, Layout, Named Rules, Named Rules (+4 more)

### Community 28 - "runs.py"
Cohesion: 0.11
Nodes (13): plays_any_size(), Preset, diff_payload(), json_safe(), metrics_payload(), presets_payload(), results_payload(), run_detail() (+5 more)

### Community 29 - "charts.js"
Cohesion: 0.12
Nodes (21): aggregate(), average(), card(), CARDS, ChartBoard, chartLayout(), DEFAULT_X, element() (+13 more)

### Community 30 - "cli.py"
Cohesion: 0.11
Nodes (13): _description(), _eval(), _experiment_options(), _load_model(), _parser(), _render(), _rewards(), _seed_list() (+5 more)

### Community 31 - "main"
Cohesion: 0.18
Nodes (17): main(), test_a_local_view_with_an_even_board_network_fails_before_any_run_folder(), test_agent_flag_is_taken_as_text_not_parsed_as_yaml(), test_agents_lists_every_architecture_and_whether_it_plays_any_board_size(), test_bad_agent_kwargs_are_reported_without_a_traceback_or_run_folder(), test_compare_without_runs_says_so(), test_config_errors_are_reported_without_a_traceback(), test_help_exits_cleanly() (+9 more)

### Community 32 - "On-device checklist — C2 dashboard"
Cohesion: 0.17
Nodes (11): Compare tab, Keyboard and accessibility, Live tab, Loading and frame, On-device checklist — C2 dashboard, Responsive, Results tab, Sidebar (+3 more)

### Community 33 - "evaluate"
Cohesion: 0.14
Nodes (15): make_env(), make_play_env(), EnvConfig, ObservationConfig, EvalStats, evaluate(), _Scripted, test_a_snake_circling_in_place_starves() (+7 more)

### Community 34 - "pr-title.yml"
Cohesion: 0.08
Nodes (23): concurrency, cancel-in-progress, group, name, runs-on, steps, timeout-minutes, GITHUB_TOKEN (+15 more)

### Community 35 - "Product"
Cohesion: 0.18
Nodes (10): Capabilities and Constraints, Evidence on Hand, Operating Context, Platform, Positioning, Product, Product Principles, Product Purpose (+2 more)

### Community 37 - "test_runs.py"
Cohesion: 0.07
Nodes (30): create_run_dir(), derive_status(), MetricsLogger, replace_atomically(), save_checkpoint(), seconds_per_update(), stall_window(), summarize_run() (+22 more)

### Community 38 - "build_agent"
Cohesion: 0.11
Nodes (20): sinusoidal_positions_2d(), available_agents(), build_agent(), count_parameters(), _agents(), _cnn_pool_input_size(), _shape_cases(), test_board_bound_agents_are_flagged_and_reject_another_size() (+12 more)

### Community 39 - "ObservationSpec"
Cohesion: 0.14
Nodes (11): ObservationSpec, _plane(), _RecordingEnv, _reference_observation(), test_default_observation_is_exactly_todays_formula(), test_local_view_is_head_centred_and_rotated_so_the_wall_ahead_is_on_top(), test_local_view_projects_far_food_onto_the_crop_edge(), test_local_view_shows_food_straight_ahead_above_the_head() (+3 more)

### Community 40 - "h"
Cohesion: 0.18
Nodes (27): actionError(), badge(), code(), distinctSwitches(), fillSelect(), fixed(), h(), icon() (+19 more)

### Community 41 - "PolicyNetwork"
Cohesion: 0.06
Nodes (20): A recurrent architecture, Add an architecture, AttentionPolicy, CNNPolicy, conv_features(), DeterministicAdaptiveAvgPool2d, MLPPolicy, PolicyNetwork (+12 more)

### Community 42 - "Running experiments"
Cohesion: 0.20
Nodes (10): Bookkeeping keys, Comparing runs, Descriptions, Install, Presets and config load order, Resuming a run, Running experiments, Seed sweeps (+2 more)

### Community 43 - "test_replay.py"
Cohesion: 0.06
Nodes (30): _Board, BoardSpec, _check_speed(), _parse_board_spec(), parse_board_specs(), ReplayError, ReplayManager, ReplaySession (+22 more)

### Community 44 - "repos"
Cohesion: 0.33
Nodes (5): id, repos, hooks, repo, rev

### Community 45 - "PolicyPlayer"
Cohesion: 0.09
Nodes (5): episode_starts(), PolicyPlayer, test_policy_player_accepts_any_object_with_a_feed_forward_act(), test_policy_player_plays_a_feed_forward_model_statelessly(), counting_player()

### Community 46 - "load_checkpoint"
Cohesion: 0.15
Nodes (12): load_checkpoint(), _config(), _curve(), _interrupted_run(), _record_training(), test_a_batching_that_splits_an_env_is_refused_before_any_run_folder(), test_a_recurrent_run_trains_and_checkpoints_its_state(), test_a_resumed_recurrent_run_matches_an_uninterrupted_one_on_cpu() (+4 more)

### Community 47 - "test_sweep.py"
Cohesion: 0.13
Nodes (11): find_runs(), _args(), _FinishedProcess, test_a_detached_sweep_starts_seeds_without_a_console_into_its_own_output(), popen(), test_a_failing_seed_does_not_stop_the_others(), test_a_launched_sweep_with_a_stop_request_exits_130(), test_a_stop_request_starts_no_further_seeds() (+3 more)

### Community 48 - "Module map"
Cohesion: 0.12
Nodes (18): _dilate(), manhattan(), reachable_fraction(), _shaped(), StepContext, _terminal_rewards(), _bfs_fraction(), _cells() (+10 more)

### Community 50 - "server"
Cohesion: 0.13
Nodes (8): dashboard_on(), no_dashboard(), opened(), render_run(), test_eval_render_opens_the_replay_on_the_dashboard_that_serves_its_runs_folder(), test_eval_render_starts_its_own_dashboard_when_the_port_serves_another_folder(), test_serve_passes_the_configs_folder_and_says_it_is_the_main_ui(), server()

### Community 51 - "Module map"
Cohesion: 0.09
Nodes (23): Module map, Add a preset, Add a reward strategy, Add an observation variant, Extending snakePPO, RewardConfig, body_order_grid(), local_view() (+15 more)

### Community 52 - "Config"
Cohesion: 0.16
Nodes (30): apply_overrides(), Config, save_config(), test_a_run_shows_its_stored_description_else_its_preset_s(), test_bad_override_raises(), test_baseline_preset_names_every_config_key(), test_baseline_preset_states_exactly_the_code_defaults(), test_bookkeeping_keys_default_to_today_and_are_validated() (+22 more)

### Community 53 - "._route"
Cohesion: 0.12
Nodes (3): _BadRequest, _Handler, _NotFound

### Community 54 - "test_recurrent.py"
Cohesion: 0.27
Nodes (13): _model(), _random_state(), test_a_recurrent_model_runs_on_any_board_size(), test_a_start_runs_the_step_from_a_zero_state(), test_an_unknown_cell_names_the_valid_ones(), test_cells_are_gru_and_lstm(), test_episode_starts_marks_the_step_after_each_done(), test_policy_player_carries_state_and_resets_only_finished_games() (+5 more)

### Community 55 - "Components"
Cohesion: 0.12
Nodes (16): Badges, Buttons, Cards / Containers, Charts (signature), Chips, Components, Empty and loading states, Inputs / Fields (+8 more)

### Community 56 - "renderCompareCharts"
Cohesion: 0.16
Nodes (15): Do:, MetricsStore, applyTheme(), compareEntries(), compareGroups(), metricsLoop, redrawCharts(), renderCompareCharts() (+7 more)

### Community 58 - "reward"
Cohesion: 0.18
Nodes (11): reward, apple, approach, death, distance_scale, free_space_penalty, hunger_penalty, potential_scale (+3 more)

### Community 59 - "test_results.py"
Cohesion: 0.19
Nodes (18): results_table(), switch_unit(), truncate(), _group(), _row(), _run(), test_a_group_shows_its_first_run_s_description_or_its_preset_s(), test_a_reduced_budget_group_needs_a_baseline_at_its_budget() (+10 more)

### Community 60 - "train"
Cohesion: 0.18
Nodes (11): train, checkpoint_interval, description, deterministic, device, eval_episodes, eval_interval, group (+3 more)

### Community 61 - "mountLaunch"
Cohesion: 0.13
Nodes (36): startLaunch(), code(), dot(), field(), h(), launchRequest(), launchRunId(), launchSummary() (+28 more)

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
Cohesion: 0.33
Nodes (5): curriculum, grow, train, description, name

### Community 66 - "both.yaml"
Cohesion: 0.33
Nodes (5): curriculum, grow, train, description, name

### Community 67 - "length.yaml"
Cohesion: 0.33
Nodes (5): curriculum, grow, train, description, name

### Community 68 - "distance.yaml"
Cohesion: 0.33
Nodes (5): reward, strategy, train, description, name

### Community 69 - "free_space.yaml"
Cohesion: 0.33
Nodes (5): reward, strategy, train, description, name

### Community 70 - "hunger.yaml"
Cohesion: 0.33
Nodes (5): reward, strategy, train, description, name

### Community 71 - "potential.yaml"
Cohesion: 0.33
Nodes (5): reward, strategy, train, description, name

### Community 72 - "sparse.yaml"
Cohesion: 0.33
Nodes (5): reward, strategy, train, description, name

### Community 73 - "api.js"
Cohesion: 0.16
Nodes (19): ApiError, controlReplay(), createReplay(), fetchDiff(), fetchLaunches(), fetchMetrics(), fetchPresets(), fetchResults() (+11 more)

### Community 74 - "test_launches.py"
Cohesion: 0.15
Nodes (26): list_launches(), FakeSpawn, _gone(), _launch(), _run(), _sleeper(), test_a_resume_launch_follows_the_run_it_resumes(), test_a_running_sweep_lists_every_seed() (+18 more)

### Community 75 - "_Group"
Cohesion: 0.14
Nodes (8): config_differences(), flatten_config(), _Group, learning_config(), protocol(), _reference_for(), switches(), test_learning_config_drops_bookkeeping_and_switches_skip_protocol_keys()

### Community 76 - "snakePPO"
Cohesion: 0.22
Nodes (7): Agent instructions, CI, Commands, Invariants, Layout, Skills, snakePPO

### Community 78 - "snakePPO"
Cohesion: 0.28
Nodes (5): Commands, Documentation, Quickstart, snakePPO, Verification

### Community 79 - "agent"
Cohesion: 0.25
Nodes (7): agent, kwargs, name, cell, train, description, name

### Community 80 - "observation"
Cohesion: 0.33
Nodes (6): observation, body, hunger, tail, view, view_size

### Community 81 - "Colors"
Cohesion: 0.29
Nodes (7): Board, Colors, Named Rules, Neutral, Primary, Run palette, State

### Community 82 - "attention.yaml"
Cohesion: 0.33
Nodes (5): agent, name, train, description, name

### Community 83 - "cnn-medium.yaml"
Cohesion: 0.33
Nodes (5): agent, name, train, description, name

### Community 84 - "cnn-small.yaml"
Cohesion: 0.33
Nodes (5): agent, name, train, description, name

### Community 85 - "cnn-tiny.yaml"
Cohesion: 0.33
Nodes (5): agent, name, train, description, name

### Community 86 - "mlp.yaml"
Cohesion: 0.33
Nodes (5): agent, name, train, description, name

### Community 87 - "recurrent.yaml"
Cohesion: 0.33
Nodes (5): agent, name, train, description, name

### Community 88 - "body-order.yaml"
Cohesion: 0.33
Nodes (5): observation, body, train, description, name

### Community 89 - "observation/hunger.yaml"
Cohesion: 0.33
Nodes (5): observation, hunger, train, description, name

### Community 90 - "local-view.yaml"
Cohesion: 0.33
Nodes (5): observation, view, train, description, name

### Community 91 - "tail.yaml"
Cohesion: 0.33
Nodes (5): observation, tail, train, description, name

### Community 92 - "Switches"
Cohesion: 0.40
Nodes (5): Architectures, Curriculum, Observation, Reward strategies, Switches

### Community 93 - "renderLive"
Cohesion: 0.29
Nodes (12): compareRuns(), displayedRunIds(), focusedKey(), knownRuns(), liveRun(), refreshDiff(), renderCompare(), renderCompareLegend() (+4 more)

### Community 94 - "Run folders"
Cohesion: 0.20
Nodes (10): Checkpoints, Dashboard, Evaluating on another board size, Launch records, Launching from the dashboard, `metrics.jsonl`, Run folders, Watching a run play (+2 more)

### Community 95 - "mountReplay"
Cohesion: 0.18
Nodes (32): replayStreamUrl(), readBoardTheme(), h(), mountReplay(), addBoard(), addCard(), boardCard(), boardsChanged() (+24 more)

### Community 97 - "C4 sanity report"
Cohesion: 0.29
Nodes (6): Body order (3 seeds against a 3-seed baseline), C4 sanity report, Caveats, Result, Training runs, Transfer (trained on 16x16, evaluated greedy over 64 episodes, best checkpoint)

### Community 98 - "config.py"
Cohesion: 0.13
Nodes (19): Add a switch, _experiment_config(), _train(), AgentConfig, _coerce(), config_from_dict(), config_to_dict(), ConfigError (+11 more)

### Community 99 - "test_server.py"
Cohesion: 0.13
Nodes (33): _get(), _json(), _post(), _run(), _status_of(), test_a_broken_preset_only_loses_the_description_fallback(), test_a_post_from_another_origin_is_forbidden_before_any_work(), test_a_post_that_is_not_json_is_forbidden() (+25 more)

### Community 100 - "Curriculum"
Cohesion: 0.11
Nodes (11): curriculum_ladder(), CurriculumConfig, CurriculumStage, Curriculum, UpdateResult, _make_curriculum(), _curriculum(), test_capacity_counts_playable_cells_or_apples_to_the_goal() (+3 more)

### Community 101 - "test_stop.py"
Cohesion: 0.14
Nodes (20): request_stop(), read_run_info(), test_cli_group_flag_is_recorded_as_text(), _after_update(), _Boom, _curve(), _interrupted_run(), _smoke_config() (+12 more)

### Community 102 - "load_config"
Cohesion: 0.16
Nodes (11): _compare(), _run_description(), list_presets(), load_config(), _preset_id(), test_a_preset_that_does_not_load_is_named(), test_list_presets_reads_every_preset_sorted_by_path(), test_partial_yaml_file_merges_onto_defaults() (+3 more)

### Community 103 - "processes.py"
Cohesion: 0.15
Nodes (13): is_alive(), _probe(), process_record(), _record_pid(), terminate(), test_a_process_that_exited_is_not_alive(), test_a_record_from_another_host_cannot_be_judged(), test_a_recycled_pid_is_not_the_recorded_process() (+5 more)

### Community 104 - "resume"
Cohesion: 0.17
Nodes (20): _check_resumable(), resume(), ResumeError, update_run_info(), _curve(), _interrupted_run(), _smoke_config(), _strip_observation() (+12 more)

### Community 105 - "test_server_replay.py"
Cohesion: 0.16
Nodes (17): _connect(), _create(), dashboard(), _get_status(), _keep_torch_threads(), _next_event(), _next_frame(), _open_stream() (+9 more)

### Community 106 - "json"
Cohesion: 0.15
Nodes (14): _get(), _launch(), live_dashboard(), _post(), _run_of(), test_a_launched_run_stops_as_interrupted_and_resumes_to_the_end(), _wait_for(), _shipped() (+6 more)

### Community 108 - "read_metrics"
Cohesion: 0.17
Nodes (14): read_metrics(), _curriculum(), _curriculum_run(), _curve(), test_a_curriculum_run_logs_its_stages_and_ends_on_the_full_board(), test_a_resumed_curriculum_run_matches_an_uninterrupted_one_on_cpu(), test_gate_never_fires_before_min_updates(), test_gate_promotes_once_the_window_mean_reaches_its_share_of_capacity() (+6 more)

### Community 109 - "test_observation.py"
Cohesion: 0.18
Nodes (13): _display_grid(), _plane(), _snakes(), test_a_local_view_as_large_as_the_board_still_records_a_different_observation(), test_a_spec_built_directly_rejects_meaningless_values(), test_body_order_runs_evenly_from_the_head_to_the_tail(), test_describe_record_names_the_channels_the_size_and_the_view(), test_every_switch_names_its_channels_in_the_fixed_order_and_is_not_binary() (+5 more)

### Community 110 - "pytest"
Cohesion: 0.16
Nodes (10): _escalate(), launch_folder(), LaunchConflict, LaunchError, stop_run(), stop_sweep(), test_launch_folder_takes_only_a_launch_of_the_runs_folder(), test_resume_sweep_of_an_unknown_launch_is_refused() (+2 more)

### Community 111 - "LaunchRequest"
Cohesion: 0.18
Nodes (12): launch_argv(), LaunchRequest, parse_launch_request(), test_launch_argv_is_the_cli_sweep_command(), test_launch_argv_is_the_cli_train_command(), test_launch_argv_sends_a_run_with_another_runs_folder_to_the_served_one(), test_launch_argv_takes_only_a_yaml_preset_inside_the_configs_folder(), test_parse_launch_request_names_the_bad_field() (+4 more)

### Community 112 - "replay.js"
Cohesion: 0.17
Nodes (13): bodyShade(), cellRect(), channels(), drawBoard(), BOARD_SIZES, CHECKPOINTS, DEFAULT_SPEED, MAX_BOARDS (+5 more)

### Community 113 - "launch_dir"
Cohesion: 0.15
Nodes (7): clear_stop_request(), launch_dir(), spawn_detached(), test_launch_dir_is_under_the_launches_folder(), test_spawn_detached_appends_output_and_stderr_to_the_log(), test_spawn_detached_uses_the_platform_detach_options(), test_stop_request_round_trip()

### Community 114 - "play_env_config"
Cohesion: 0.36
Nodes (8): play_env_config(), _config(), test_play_env_config_keeps_the_trained_board_when_no_size_is_given(), test_play_env_config_lets_a_board_bound_run_play_its_own_board_and_a_local_view_any(), test_play_env_config_moves_a_size_agnostic_run_to_another_board(), test_play_env_config_refuses_a_board_smaller_than_5(), test_play_env_config_refuses_another_board_for_a_board_bound_run(), test_plays_any_size_needs_a_size_agnostic_network_or_a_local_view()

### Community 115 - "_smoke_run"
Cohesion: 0.24
Nodes (8): _edit_saved_config(), _smoke_run(), test_eval_on_another_board_does_not_revalidate_the_run_config(), test_eval_plays_a_body_order_run_and_refuses_it_under_another_observation(), test_eval_plays_a_local_view_run_of_a_board_bound_network_on_any_board(), test_eval_plays_a_size_agnostic_run_on_another_board(), test_eval_refuses_a_checkpoint_that_does_not_fit_its_configured_network(), test_eval_refuses_another_board_for_a_board_bound_run()

### Community 116 - "parametrize"
Cohesion: 0.22
Nodes (8): _saved_description(), test_an_explicit_description_wins_without_a_suffix(), test_eval_render_refuses_a_checkpoint_that_is_not_a_runs_best_or_last(), test_eval_render_refuses_options_the_replay_cannot_honour(), test_name_flag_is_taken_as_text_not_parsed_as_yaml(), test_speed_for_sleep_picks_the_nearest_replay_speed(), test_sweep_builds_the_network_once_before_any_seed_starts(), test_train_saves_the_preset_description_with_its_changes()

### Community 117 - "_FakeSpawn"
Cohesion: 0.20
Nodes (3): dashboard(), _FakeSpawn, sleeper()

### Community 118 - "_run"
Cohesion: 0.29
Nodes (3): Progress, _run(), _Session

### Community 119 - "served_runs_dir"
Cohesion: 0.40
Nodes (3): served_runs_dir(), test_served_runs_dir_finds_the_folder_a_dashboard_serves(), test_served_runs_dir_is_none_when_no_dashboard_answers()

### Community 120 - "test_eval_render_serves_the_runs_folder_when_no_dashboard_answers"
Cohesion: 0.40
Nodes (4): _boards(), test_eval_render_serves_the_runs_folder_when_no_dashboard_answers(), test_serve_binds_localhost_opens_the_browser_and_stops_on_ctrl_c(), interrupted()

## Knowledge Gaps
- **1 isolated node(s):** `snakeppo`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 752 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Module map` connect `Module map` to `test_ppo_functional.py`, `train`, `trainer.py`, `TorchSnakeEnv`, `launches.py`, `load_run`, `server.py`, `results.py`, `runs.py`, `cli.py`, `evaluate`, `test_runs.py`, `build_agent`, `ObservationSpec`, `PolicyNetwork`, `test_replay.py`, `PolicyPlayer`, `test_sweep.py`, `Module map`, `test_results.py`, `test_launches.py`, `config.py`, `Curriculum`, `test_stop.py`, `load_config`, `processes.py`, `resume`, `pytest`, `LaunchRequest`, `launch_dir`, `play_env_config`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Are the 104 inferred relationships involving `Module map` (e.g. with `AttentionPolicy` and `CNNPolicy`) actually correct?**
  _`Module map` has 104 INFERRED edges - model-reasoned connections that need verification._
- **What connects `snakeppo` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `ppo` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Why does `Environment contract` connect `TorchSnakeEnv` to `replay.js`, `trainer.py`, `ObservationSpec`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Are the 50 inferred relationships involving `Config` (e.g. with `load_policy()` and `make_policy()`) actually correct?**
  _`Config` has 50 INFERRED edges - model-reasoned connections that need verification._
- **Should `trainer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0700354609929078 - nodes in this community are weakly interconnected._