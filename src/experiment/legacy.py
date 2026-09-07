"""Folding the old per-component databases into the per-experiment one.

The store used to keep one ``<results_dir>/<component>.db`` per component, with
all runs in a ``runs`` table and the curves in a ``curves`` blob. This reads
those files and writes their rows into the experiment's single database, one
table per component.

Run ids are unchanged by the move, so a migrated run still counts as done and
nothing recomputes. The old files are left where they are: migration is cheap to
repeat and the results are expensive to lose, so deleting them is a decision for
whoever checks the numbers afterwards.

This module is one-shot. Delete it once every results directory has been
migrated, here and on the cluster.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiment.design import Experiment
from experiment.results import (
    _COLUMNS,
    _connect_write,
    _ensure_table,
    _query_ro,
    database_path,
)

__all__ = ["legacy_paths", "migrate"]


_IDENTITY = ("run_id", "config_id", "seed")


def legacy_paths(experiment: Experiment, component: str) -> list[Path]:
    """Find the old databases holding one component's runs.

    Args:
        experiment: The experiment being migrated.
        component: The component to look for.

    Returns:
        The component's old database and any per-worker parts beside it, in a
        stable order. Empty if the component was never run under the old store.
    """
    database = experiment.results_dir / f"{component}.db"
    parts = experiment.results_dir / f"{component}.parts"
    return ([database] if database.exists() else []) + (
        sorted(parts.glob("part-*.db")) if parts.is_dir() else []
    )


def migrate(experiment: Experiment) -> dict[str, int]:
    """Copy an experiment's legacy results into its database.

    Idempotent: a run already present is left alone, so migrating twice moves
    nothing the second time and never overwrites a newer result.

    Args:
        experiment: The experiment to migrate.

    Returns:
        The number of runs moved, per component. Components with no old
        database are absent.
    """
    moved: dict[str, int] = {}
    conn = None
    try:
        for component in experiment.components:
            sources = legacy_paths(experiment, component.name)
            if not sources:
                continue
            if conn is None:
                conn = _connect_write(database_path(experiment))
            _ensure_table(conn, component.name)

            before = conn.total_changes
            for source in sources:
                rows = _query_ro(
                    source,
                    "SELECT run_id, config_id, seed, config_json, curves FROM runs",
                )
                for run_id, config_id, seed, config_json, curves in rows:
                    stored = json.loads(config_json)
                    config = {
                        k: v for k, v in stored.items() if k not in _IDENTITY
                    }
                    conn.execute(
                        f'INSERT OR IGNORE INTO "{component.name}" ({_COLUMNS}) '
                        "VALUES (?, ?, ?, ?, ?)",
                        (run_id, config_id, int(seed), json.dumps(config), curves),
                    )
            conn.commit()
            moved[component.name] = conn.total_changes - before
    finally:
        if conn is not None:
            conn.close()
    return moved
