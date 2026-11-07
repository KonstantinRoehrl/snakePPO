"""JSON payloads of the dashboard API, built by the same code the CLI uses.

Every function returns plain JSON-ready data; :func:`json_safe` turns non-finite floats into
``null`` because browsers reject ``NaN`` in JSON.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from snakeppo.config import config_to_dict
from snakeppo.results import (
    config_differences,
    learning_config,
    results_table,
    run_score,
    switches,
)
from snakeppo.runs import MetricsChunk, RunRecord, seconds_per_update


def json_safe(value: Any) -> Any:
    """Copy of ``value`` with tuples as lists and NaN/inf as ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def run_summary(record: RunRecord) -> dict[str, Any]:
    """One run-list entry: identity, status, progress, score and switches."""
    score = run_score(record.metrics)
    info = record.info
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
        "score": score.value,
        "score_evals": score.evals,
        "switches": switches(record.config),
    }


def run_detail(record: RunRecord) -> dict[str, Any]:
    return {
        "run": run_summary(record),
        "config": config_to_dict(record.config),
        "info": record.info,
    }


def metrics_payload(chunk: MetricsChunk) -> dict[str, Any]:
    return {"lines": chunk.lines, "offset": chunk.offset, "reset": chunk.reset}


def results_payload(records: Sequence[RunRecord], baseline: str | None) -> dict[str, Any]:
    table = results_table(records, baseline)
    rows = [asdict(row) | {"budget": row.budget} for row in table.rows]
    return {"rows": rows, "left_out": table.left_out, "baseline": table.baseline}


def diff_payload(base: RunRecord, others: Sequence[RunRecord]) -> dict[str, Any]:
    """Each run's learning-config differences from ``base``: ``{id: {key: [base, run]}}``."""
    reference = learning_config(base.config)
    return {
        "base": base.id,
        "differences": {
            run.id: config_differences(reference, learning_config(run.config)) for run in others
        },
    }
