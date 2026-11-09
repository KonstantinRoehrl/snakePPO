"""The multi-seed comparison protocol: run scores, switches, seed groups and verdicts.

Pure functions over :class:`~snakeppo.runs.RunRecord`; ``snakeppo results`` and the dashboard
both build their table here, so they always agree. The protocol is documented in
``docs/experiments.md`` (Comparing runs).
"""

from __future__ import annotations

import csv
import io
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

import numpy as np

from snakeppo.config import STRATEGY_PARAMETERS, Config, CurriculumConfig, config_to_dict
from snakeppo.runs import RunRecord

# Keys that name, place or schedule a run without changing what it learns.
BOOKKEEPING_KEYS = frozenset(
    {
        "env.seed",
        "train.name",
        "train.group",
        "train.runs_dir",
        "train.device",
        "train.deterministic",
        "train.checkpoint_interval",
    }
)
# Keys that set the budget and how runs are measured: shown as the budget, never a switch.
PROTOCOL_KEYS = ("train.total_env_steps", "train.eval_interval", "train.eval_episodes")
# A selector and the keys only it reads are one switch: changing a strategy together with its
# own coefficient, an architecture with its kwargs or a local view with its size is still one
# experiment.
SELECTOR_KEYS = {
    "reward.strategy": tuple(f"reward.{name}" for name in STRATEGY_PARAMETERS),
    "curriculum.grow": tuple(
        f"curriculum.{f.name}" for f in fields(CurriculumConfig) if f.name != "grow"
    ),
    "agent.name": ("agent.kwargs",),
    "observation.view": ("observation.view_size",),
}
SCORE_EVALS = 3
MIN_PAIRED_SEEDS = 3
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
FORMATS = ("text", "markdown", "csv")


# ------------------------------------------------------------------ configs
def flatten_config(config: Config) -> dict[str, Any]:
    """``{"section.key": value}`` for every config value (``agent.kwargs`` stays one dict)."""
    return {
        f"{section}.{key}": value
        for section, values in config_to_dict(config).items()
        for key, value in values.items()
    }


def learning_config(config: Config) -> dict[str, Any]:
    """The flattened config without :data:`BOOKKEEPING_KEYS`."""
    return {k: v for k, v in flatten_config(config).items() if k not in BOOKKEEPING_KEYS}


def config_differences(
    base: Mapping[str, Any], other: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """``{key: (base value, other value)}`` for every dotted key whose values differ."""
    keys = sorted(base.keys() | other.keys())
    return {k: (base.get(k), other.get(k)) for k in keys if base.get(k) != other.get(k)}


def protocol(config: Config) -> dict[str, Any]:
    """The budget and eval settings a run was measured with."""
    flat = flatten_config(config)
    return {key: flat[key] for key in PROTOCOL_KEYS}


def switches(config: Config) -> dict[str, Any]:
    """A config's learning differences from the code defaults, protocol keys excluded."""
    differences = config_differences(learning_config(Config()), learning_config(config))
    return {k: new for k, (_, new) in differences.items() if k not in PROTOCOL_KEYS}


def switch_unit(key: str) -> str:
    """The switch a differing key belongs to: its selector, or the key itself."""
    for selector, owned in SELECTOR_KEYS.items():
        if key == selector or key in owned:
            return selector
    return key


# ------------------------------------------------------------------ scores
@dataclass(frozen=True)
class RunScore:
    value: float | None  # mean of the last SCORE_EVALS eval scores; None without any eval
    evals: int  # how many evals the mean used


def run_score(metrics: Sequence[Mapping[str, Any]]) -> RunScore:
    """A run's score: the mean of its last :data:`SCORE_EVALS` ``eval/score_mean`` values."""
    evals = [line["eval/score_mean"] for line in metrics if "eval/score_mean" in line]
    last = evals[-SCORE_EVALS:]
    return RunScore(statistics.fmean(last) if last else None, len(last))


@dataclass(frozen=True)
class Bootstrap:
    mean: float
    low: float
    high: float


def paired_bootstrap(
    differences: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Bootstrap:
    """Mean of the paired differences and its 95% percentile bootstrap CI (deterministic)."""
    values = np.asarray(differences, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Bootstrap(float(values.mean()), float(low), float(high))


def verdict(ci: Bootstrap) -> str:
    """``better`` or ``worse`` when the CI excludes zero, else ``inconclusive``."""
    if ci.low > 0:
        return "better"
    if ci.high < 0:
        return "worse"
    return "inconclusive"


# ------------------------------------------------------------------ groups
@dataclass(frozen=True)
class _Group:
    id: str
    runs: list[RunRecord]

    @property
    def finished(self) -> list[RunRecord]:
        return [run for run in self.runs if run.status == "finished"]

    @property
    def mixed(self) -> bool:
        first = learning_config(self.runs[0].config)
        return any(learning_config(run.config) != first for run in self.runs[1:])

    @property
    def config(self) -> Config:
        return self.runs[0].config

    @property
    def is_baseline(self) -> bool:
        return not self.mixed and not switches(self.config)

    @property
    def newest(self) -> datetime:
        return max(datetime.fromisoformat(run.info["created"]) for run in self.runs)

    def finished_scores(self) -> dict[int, float]:
        """Seed -> run score of each finished run that has one."""
        scores = {}
        for run in self.finished:
            score = run_score(run.metrics).value
            if score is not None:
                scores[run.seed] = score
        return scores

    def duplicate_seed(self) -> int | None:
        counts = Counter(run.seed for run in self.finished)
        duplicates = sorted(seed for seed, count in counts.items() if count > 1)
        return duplicates[0] if duplicates else None


@dataclass(frozen=True)
class GroupRow:
    """One line of the results table."""

    group: str
    protocol: dict[str, Any]
    finished: int
    seeds: int
    score_mean: float | None
    score_std: float | None  # sample std across finished seeds; None below two
    reference: str | None
    differences: dict[str, tuple[Any, Any]]  # learning keys: (reference value, group value)
    verdict: str | None  # better, worse, inconclusive, reference; None when withheld
    reason: str | None  # why the verdict is withheld
    mean_difference: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    paired_seeds: list[int] = field(default_factory=list)
    unpaired_seeds: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def budget(self) -> int:
        return self.protocol["train.total_env_steps"]


@dataclass(frozen=True)
class ResultsTable:
    rows: list[GroupRow]
    left_out: int  # ungrouped runs, which are not in the table
    baseline: str | None  # the forced reference group, if any


def results_table(runs: Sequence[RunRecord], baseline: str | None = None) -> ResultsTable:
    """Group runs by ``group`` and judge each group against its reference group."""
    by_id: dict[str, list[RunRecord]] = {}
    for run in runs:
        if run.group is not None:
            by_id.setdefault(run.group, []).append(run)
    groups = sorted(
        (_Group(gid, members) for gid, members in by_id.items()),
        key=lambda g: g.newest,
        reverse=True,
    )
    if baseline is not None and baseline not in by_id:
        known = ", ".join(sorted(by_id)) or "none"
        raise ValueError(f"unknown baseline group '{baseline}'; groups: {known}")
    forced = next((g for g in groups if g.id == baseline), None)
    rows = [_row(group, _reference_for(group, groups, forced)) for group in groups]
    left_out = sum(1 for run in runs if run.group is None)
    return ResultsTable(rows, left_out, baseline)


def _reference_for(group: _Group, groups: list[_Group], forced: _Group | None) -> _Group | None:
    if forced is not None:
        return forced
    if group.is_baseline:
        return group
    same_protocol = (
        g for g in groups if g.is_baseline and protocol(g.config) == protocol(group.config)
    )
    return next(same_protocol, None)  # groups are sorted newest first


def _row(group: _Group, reference: _Group | None) -> GroupRow:
    scores = group.finished_scores()
    values = list(scores.values())
    notes = [
        f"{run.id}: score from {score.evals} eval(s)"
        for run in group.finished
        if 0 < (score := run_score(run.metrics)).evals < SCORE_EVALS
    ]
    base = {
        "group": group.id,
        "protocol": protocol(group.config),
        "finished": len(group.finished),
        "seeds": len(group.runs),
        "score_mean": statistics.fmean(values) if values else None,
        "score_std": statistics.stdev(values) if len(values) > 1 else None,
        "reference": reference.id if reference is not None else None,
        "notes": notes,
    }
    if group.mixed:
        return GroupRow(**base, differences={}, verdict=None, reason="mixed configs")
    if reference is None:
        reason = "no baseline group with the same budget and eval settings"
        return GroupRow(**base, differences={}, verdict=None, reason=reason)
    if reference is group:
        return GroupRow(**base, differences={}, verdict="reference", reason=None)
    if reference.mixed:
        reason = "the reference group has mixed configs"
        return GroupRow(**base, differences={}, verdict=None, reason=reason)

    learning = config_differences(learning_config(reference.config), learning_config(group.config))
    differences = {k: v for k, v in learning.items() if k not in PROTOCOL_KEYS}
    ref_scores = reference.finished_scores()
    paired = sorted(scores.keys() & ref_scores.keys())
    unpaired = sorted(scores.keys() ^ ref_scores.keys())
    base |= {"differences": differences, "paired_seeds": paired, "unpaired_seeds": unpaired}

    reason = None
    protocol_diff = [k for k in PROTOCOL_KEYS if k in learning]
    if (seed := group.duplicate_seed()) is not None:
        reason = f"duplicate seed {seed}"
    elif (seed := reference.duplicate_seed()) is not None:
        reason = f"duplicate seed {seed} in the reference group"
    elif protocol_diff:
        reason = f"budget or eval settings differ from the reference ({', '.join(protocol_diff)})"
    elif len(units := {switch_unit(key) for key in differences}) > 1:
        reason = f"{len(units)} learning switches differ from the reference (at most 1)"
    elif len(paired) < MIN_PAIRED_SEEDS:
        reason = f"only {len(paired)} paired finished seeds (need {MIN_PAIRED_SEEDS})"
    if reason is not None:
        return GroupRow(**base, verdict=None, reason=reason)

    ci = paired_bootstrap([scores[seed] - ref_scores[seed] for seed in paired])
    return GroupRow(
        **base,
        verdict=verdict(ci),
        reason=None,
        mean_difference=ci.mean,
        ci_low=ci.low,
        ci_high=ci.high,
    )


# ------------------------------------------------------------------ formatting
COLUMNS = ("group", "budget", "seeds", "score", "reference", "differences", "vs reference")


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _cells(row: GroupRow) -> tuple[str, ...]:
    score = _number(row.score_mean)
    if row.score_std is not None:
        score += f" ± {row.score_std:.2f}"
    differences = ", ".join(f"{k}={new}" for k, (_, new) in row.differences.items()) or "-"
    if row.verdict is None:
        judged = f"withheld: {row.reason}"
    elif row.verdict == "reference":
        judged = "reference"
    else:
        judged = (
            f"{row.verdict}: {row.mean_difference:+.2f} [{row.ci_low:+.2f}, {row.ci_high:+.2f}]"
        )
    return (
        row.group,
        f"{row.budget:,}",
        f"{row.finished}/{row.seeds}",
        score,
        row.reference or "-",
        differences,
        judged,
    )


def _footer(table: ResultsTable) -> list[str]:
    lines = []
    for row in table.rows:
        if row.unpaired_seeds:
            seeds = ", ".join(map(str, row.unpaired_seeds))
            lines.append(f"{row.group}: unpaired seeds {seeds}")
        lines.extend(f"{row.group}: {note}" for note in row.notes)
    if table.left_out:
        lines.append(f"{table.left_out} ungrouped run(s) left out (sweeps and --group set one).")
    return lines


def format_results(table: ResultsTable, fmt: str = "text") -> str:
    """Render the table as aligned ``text``, a ``markdown`` table or ``csv``."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    if fmt == "csv":
        return _csv(table)
    cells = [COLUMNS, *(_cells(row) for row in table.rows)]
    if fmt == "markdown":
        body = ["| " + " | ".join(line) + " |" for line in cells]
        body.insert(1, "|" + "|".join(" --- " for _ in COLUMNS) + "|")
    else:
        widths = [max(len(line[i]) for line in cells) for i in range(len(COLUMNS))]
        body = [
            "  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)).rstrip()
            for line in cells
        ]
    footer = _footer(table)
    return "\n".join(body + ([""] + footer if footer else [])) + "\n"


def _csv(table: ResultsTable) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            "group",
            *PROTOCOL_KEYS,
            "finished",
            "seeds",
            "score_mean",
            "score_std",
            "reference",
            "differences",
            "verdict",
            "reason",
            "mean_difference",
            "ci_low",
            "ci_high",
            "paired_seeds",
            "unpaired_seeds",
        ]
    )
    for row in table.rows:
        writer.writerow(
            [
                row.group,
                *(row.protocol[k] for k in PROTOCOL_KEYS),
                row.finished,
                row.seeds,
                row.score_mean,
                row.score_std,
                row.reference,
                "; ".join(f"{k}={new}" for k, (_, new) in row.differences.items()),
                row.verdict,
                row.reason,
                row.mean_difference,
                row.ci_low,
                row.ci_high,
                " ".join(map(str, row.paired_seeds)),
                " ".join(map(str, row.unpaired_seeds)),
            ]
        )
    return out.getvalue()
