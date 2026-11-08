import csv
import io
from datetime import datetime

import pytest

from snakeppo.config import Config, apply_overrides, save_config
from snakeppo.results import (
    BOOKKEEPING_KEYS,
    PROTOCOL_KEYS,
    Bootstrap,
    config_differences,
    flatten_config,
    format_results,
    learning_config,
    paired_bootstrap,
    results_table,
    run_score,
    switch_unit,
    switches,
    verdict,
)
from snakeppo.runs import (
    MetricsLogger,
    create_run_dir,
    load_run,
    update_run_info,
    write_run_info,
)


def _run(tmp_path, group, seed, evals, *overrides, status="finished", created=None):
    """A synthetic run folder: config, run.json and one eval line per score in ``evals``."""
    config = apply_overrides(Config(), [f"env.seed={seed}", *overrides])
    stamp = datetime(2026, 11, 7, 9, 0, 0)
    run_dir = create_run_dir(tmp_path, group or "solo", seed, now=stamp)
    save_config(config, run_dir / "config.yaml")
    write_run_info(
        run_dir, name="x", agent="cnn", param_count=1, device="cpu", total_updates=9, group=group
    )
    update_run_info(run_dir, status=status, created=created or "2026-11-07T09:00:00+01:00")
    with MetricsLogger(run_dir, tensorboard=False) as logger:
        for update, score in enumerate(evals, start=1):
            logger.log(update, update * 10, {"eval/score_mean": score})
    return load_run(run_dir)


def _group(tmp_path, group, scores, *overrides, created="2026-11-07T09:00:00+01:00"):
    return [
        _run(tmp_path, group, seed, [score] * 3, *overrides, created=created)
        for seed, score in scores.items()
    ]


def _row(table, group):
    (row,) = [row for row in table.rows if row.group == group]
    return row


def test_run_score_is_the_mean_of_the_last_three_evals():
    lines = [{"eval/score_mean": v} for v in (1.0, 2.0, 4.0, 6.0)] + [{"loss/total": 1.0}]
    assert run_score(lines).value == 4.0 and run_score(lines).evals == 3
    assert run_score([{"eval/score_mean": 3.0}]).evals == 1
    assert run_score([{"loss/total": 1.0}]).value is None


def test_learning_config_drops_bookkeeping_and_switches_skip_protocol_keys():
    assert BOOKKEEPING_KEYS.isdisjoint(learning_config(Config()))
    assert set(PROTOCOL_KEYS) <= learning_config(Config()).keys()
    config = apply_overrides(
        Config(), ["ppo.target_kl=0.02", "train.total_env_steps=1000", "env.seed=7"]
    )
    assert switches(config) == {"ppo.target_kl": 0.02}
    assert switches(Config()) == {}
    assert flatten_config(Config())["agent.kwargs"] == {}
    assert config_differences({"a": 1, "b": 2}, {"a": 1, "c": 3}) == {
        "b": (2, None),
        "c": (None, 3),
    }


def test_paired_bootstrap_is_deterministic_and_drives_the_verdict():
    first = paired_bootstrap([0.5, 1.0, 2.0, 0.2])
    assert first == paired_bootstrap([0.5, 1.0, 2.0, 0.2])
    assert first.low <= first.mean <= first.high
    assert paired_bootstrap([1.0, 1.0, 1.0]) == Bootstrap(1.0, 1.0, 1.0)
    assert verdict(Bootstrap(1.0, 0.1, 2.0)) == "better"
    assert verdict(Bootstrap(-1.0, -2.0, -0.1)) == "worse"
    assert verdict(Bootstrap(0.0, -1.0, 1.0)) == "inconclusive"


def test_a_switch_group_is_judged_against_the_newest_baseline_at_its_budget(tmp_path):
    runs = [
        *_group(
            tmp_path, "base-old", {1: 1.0, 2: 1.0, 3: 1.0}, created="2026-11-01T09:00:00+01:00"
        ),
        *_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5}),
        *_group(tmp_path, "kl", {1: 4.0, 2: 5.0, 3: 4.5}, "ppo.target_kl=0.02"),
        *_group(tmp_path, "worse", {1: 0.0, 2: 1.0, 3: 0.5}, "ppo.clip_value_loss=true"),
        *_group(tmp_path, "same", {1: 2.0, 2: 3.5, 3: 2.0}, "ppo.lr_schedule=linear"),
        _run(tmp_path, None, 9, [1.0]),
    ]
    table = results_table(runs)
    assert table.left_out == 1
    assert _row(table, "base").verdict == "reference" and _row(table, "base").reference == "base"
    assert _row(table, "base-old").verdict == "reference"
    kl = _row(table, "kl")
    assert kl.reference == "base" and kl.verdict == "better"
    assert kl.differences == {"ppo.target_kl": (None, 0.02)}
    assert kl.mean_difference == pytest.approx(2.0) and kl.paired_seeds == [1, 2, 3]
    assert kl.finished == 3 and kl.seeds == 3 and kl.score_mean == pytest.approx(4.5)
    assert kl.score_std == pytest.approx(0.5)
    assert _row(table, "worse").verdict == "worse"
    assert _row(table, "same").verdict == "inconclusive"


def test_forced_baseline_is_every_row_s_reference(tmp_path):
    runs = [
        *_group(
            tmp_path, "base-old", {1: 1.0, 2: 1.0, 3: 1.0}, created="2026-11-01T09:00:00+01:00"
        ),
        *_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5}),
        *_group(tmp_path, "kl", {1: 4.0, 2: 5.0, 3: 4.5}, "ppo.target_kl=0.02"),
    ]
    table = results_table(runs, baseline="base-old")
    assert table.baseline == "base-old"
    assert _row(table, "base-old").verdict == "reference"
    assert _row(table, "base").reference == "base-old" and _row(table, "base").verdict == "better"
    assert _row(table, "kl").reference == "base-old"
    with pytest.raises(ValueError, match="unknown baseline group 'nope'"):
        results_table(runs, baseline="nope")


@pytest.mark.parametrize(
    ("group_runs", "reason"),
    [
        (
            lambda p: [
                _run(p, "g", 1, [1.0], "ppo.target_kl=0.02"),
                _run(p, "g", 2, [1.0], "ppo.target_kl=0.03"),
            ],
            "mixed configs",
        ),
        (
            lambda p: _group(p, "g", {1: 1.0, 2: 1.0}, "ppo.target_kl=0.02"),
            "only 2 paired finished seeds (need 3)",
        ),
        (
            lambda p: [
                *_group(p, "g", {1: 1.0, 2: 1.0, 3: 1.0}, "ppo.target_kl=0.02"),
                _run(p, "g", 3, [2.0], "ppo.target_kl=0.02"),
            ],
            "duplicate seed 3",
        ),
        (
            lambda p: _group(
                p, "g", {1: 1.0, 2: 1.0, 3: 1.0}, "ppo.target_kl=0.02", "ppo.clip_value_loss=true"
            ),
            "2 learning switches differ from the reference (at most 1)",
        ),
        (
            lambda p: _group(
                p, "g", {1: 1.0, 2: 1.0, 3: 1.0}, "reward.strategy=hunger", "ppo.lr_schedule=linear"
            ),
            "2 learning switches differ from the reference (at most 1)",
        ),
    ],
)
def test_verdicts_are_withheld_for_unfair_comparisons(tmp_path, group_runs, reason):
    runs = [*_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5}), *group_runs(tmp_path)]
    row = _row(results_table(runs), "g")
    assert row.verdict is None and row.reason == reason


def test_a_reduced_budget_group_needs_a_baseline_at_its_budget(tmp_path):
    runs = [
        *_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5}),
        *_group(
            tmp_path,
            "short",
            {1: 1.0, 2: 1.0, 3: 1.0},
            "ppo.target_kl=0.02",
            "train.total_env_steps=1000",
        ),
    ]
    short = _row(results_table(runs), "short")
    assert short.verdict is None and short.reference is None
    assert short.reason == "no baseline group with the same budget and eval settings"
    forced = _row(results_table(runs, baseline="base"), "short")
    assert (
        forced.reason == "budget or eval settings differ from the reference (train.total_env_steps)"
    )


def test_unfinished_and_unpaired_seeds_are_listed_but_do_not_block(tmp_path):
    runs = [
        *_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5, 4: 2.0}),
        *_group(tmp_path, "kl", {1: 4.0, 2: 5.0, 3: 4.5}, "ppo.target_kl=0.02"),
        _run(tmp_path, "kl", 5, [9.0], "ppo.target_kl=0.02", status="failed"),
    ]
    kl = _row(results_table(runs), "kl")
    assert kl.finished == 3 and kl.seeds == 4
    assert kl.paired_seeds == [1, 2, 3] and kl.unpaired_seeds == [4]
    assert kl.verdict == "better"


def test_formats_render_every_row(tmp_path):
    runs = [
        *_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5}),
        *_group(tmp_path, "kl", {1: 4.0, 2: 5.0, 3: 4.5}, "ppo.target_kl=0.02"),
        _run(tmp_path, "kl", 4, [4.0], "ppo.target_kl=0.02"),
        _run(tmp_path, None, 9, [1.0]),
    ]
    table = results_table(runs)
    text = format_results(table)
    assert "kl" in text and "better: +2.00" in text and "ppo.target_kl=0.02" in text
    assert "1 ungrouped run(s) left out" in text
    assert "score from 1 eval(s)" in text
    markdown = format_results(table, "markdown")
    assert markdown.splitlines()[1].startswith("| --- |")
    rows = list(csv.DictReader(io.StringIO(format_results(table, "csv"))))
    assert {row["group"] for row in rows} == {"base", "kl"}
    assert rows[0]["train.total_env_steps"] == "10000000"
    with pytest.raises(ValueError, match="format must be one of"):
        format_results(table, "html")


def test_a_selector_and_its_own_keys_count_as_one_switch(tmp_path):
    runs = [
        *_group(tmp_path, "base", {1: 2.0, 2: 3.0, 3: 2.5}),
        *_group(
            tmp_path,
            "hunger",
            {1: 3.0, 2: 4.0, 3: 3.5},
            "reward.strategy=hunger",
            "reward.hunger_penalty=0.05",
            "reward.timeout_penalty=2",
        ),
        *_group(
            tmp_path,
            "grow",
            {1: 2.0, 2: 3.0, 3: 2.5},
            "curriculum.grow=board",
            "curriculum.max_updates=50",
        ),
    ]
    table = results_table(runs)
    hunger = _row(table, "hunger")
    assert hunger.verdict == "better" and len(hunger.differences) == 3
    grow = _row(table, "grow")
    assert grow.verdict == "inconclusive"
    assert grow.differences == {
        "curriculum.grow": ("off", "board"),
        "curriculum.max_updates": (100, 50),
    }


def test_switch_units_fold_only_keys_a_selector_owns():
    assert switch_unit("reward.potential_scale") == "reward.strategy"
    assert switch_unit("reward.strategy") == "reward.strategy"
    assert switch_unit("curriculum.promote_window") == "curriculum.grow"
    for shared in ("reward.apple", "reward.approach", "ppo.gamma", "ppo.target_kl"):
        assert switch_unit(shared) == shared


def test_an_architecture_and_a_view_own_their_settings():
    assert switch_unit("agent.kwargs") == "agent.name"
    assert switch_unit("agent.name") == "agent.name"
    assert switch_unit("observation.view_size") == "observation.view"
    for own in ("observation.tail", "observation.body", "observation.hunger"):
        assert switch_unit(own) == own
