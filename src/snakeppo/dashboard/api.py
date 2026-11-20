"""JSON payloads of the dashboard API, built by the same code the CLI uses.

Every function returns plain JSON-ready data; :func:`json_safe` turns non-finite floats into
``null`` because browsers reject ``NaN`` in JSON.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from snakeppo.builders import plays_any_size
from snakeppo.config import Preset, config_to_dict, run_description
from snakeppo.processes import is_alive, stop_requested
from snakeppo.results import (
    BOOKKEEPING_KEYS,
    config_differences,
    learning_config,
    results_table,
    run_score,
    switches,
)
from snakeppo.runs import (
    MetricsChunk,
    RunRecord,
    last_write_time,
    seconds_per_update,
    stall_window,
)


def json_safe(value: Any) -> Any:
    """Copy of ``value`` with tuples as lists and NaN/inf as ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def run_summary(
    record: RunRecord, now: float | None = None, presets: Sequence[Preset] = ()
) -> dict[str, Any]:
    """One run-list entry: identity, status, progress, freshness, score, switches and process.

    ``quiet_seconds`` is measured here rather than in the browser so the page needs no clock
    of its own and sees the same silence the stall rule sees. ``alive`` is ``None`` when the
    run has no process record or one from another machine; ``pid`` is set only while alive.
    """
    score = run_score(record.metrics)
    info = record.info
    process = info.get("process")
    alive = is_alive(process)
    return {
        "id": record.id,
        "name": info.get("name"),
        "group": record.group,
        "seed": record.seed,
        "agent": info.get("agent"),
        "device": info.get("device"),
        "created": info.get("created"),
        "schema_version": info.get("schema_version", 1),
        "status": record.status,
        "update": record.update,
        "total_updates": record.total_updates,
        "env_steps": record.env_steps,
        "seconds_per_update": seconds_per_update(record.metrics),
        "quiet_seconds": max(
            0.0, (time.time() if now is None else now) - last_write_time(record.path)
        ),
        "stall_seconds": stall_window(record.metrics),
        "score": score.value,
        "score_evals": score.evals,
        "switches": switches(record.config),
        "description": run_description(record.config, presets),
        "alive": alive,
        "pid": process["pid"] if alive else None,
        "launch": info.get("launch"),
        "stop_requested": stop_requested(record.path),
        "size_agnostic": plays_any_size(record.config),
        "board_size": record.config.env.board_size,
    }


def run_detail(record: RunRecord, presets: Sequence[Preset] = ()) -> dict[str, Any]:
    return {
        "run": run_summary(record, presets=presets),
        "config": config_to_dict(record.config),
        "info": record.info,
    }


def metrics_payload(chunk: MetricsChunk) -> dict[str, Any]:
    return {"lines": chunk.lines, "offset": chunk.offset, "reset": chunk.reset}


def results_payload(
    records: Sequence[RunRecord], baseline: str | None, presets: Sequence[Preset] = ()
) -> dict[str, Any]:
    table = results_table(records, baseline, presets)
    rows = [asdict(row) | {"budget": row.budget} for row in table.rows]
    return {"rows": rows, "left_out": table.left_out, "baseline": table.baseline}


def presets_payload(presets: Sequence[Preset], configs_dir: Path) -> dict[str, Any]:
    """The Launch view's presets, plus the keys its description suffix leaves out."""
    return {
        "presets": [asdict(preset) for preset in presets],
        "bookkeeping_keys": sorted(BOOKKEEPING_KEYS),
        "configs_dir": configs_dir.as_posix(),
    }


def diff_payload(base: RunRecord, others: Sequence[RunRecord]) -> dict[str, Any]:
    """Each run's learning-config differences from ``base``: ``{id: {key: [base, run]}}``."""
    reference = learning_config(base.config)
    return {
        "base": base.id,
        "differences": {
            run.id: config_differences(reference, learning_config(run.config)) for run in others
        },
    }
