"""Where an experiment's results live.

One SQLite database per component, holding one row per run - its id, the id of
its config, its seed and the config itself - and one row per metric a run
produced, its array stored as compressed ``.npy`` bytes. The harness never names
a metric: they are whatever the shard function returned.

A shared SQLite file is not safe under concurrent writes, and on a cluster
filesystem it is worse, so a worker never writes a component's database
directly. It writes its own part, and the parts are merged in at the end of a
sweep. Reads always union the database with any parts still present, so
results from a sweep that was interrupted before its merge are still visible and
still count as done. Reads also take in the other machine's results, mirrored
under ``cluster/`` locally or ``local/`` on the cluster.

    results/
        dqn.db                  # merged: one per component
        pinball.parts/
            dqn/
                part-0.db       # transient, one per worker
                part-1.db
        cluster/                # read-only mirror of the cluster's results/
            dqn.db
"""

from __future__ import annotations

import io
import json
import shutil
import sqlite3
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl

from experiment.design import Experiment
from experiment.identity import as_point, config_id
from experiment.plan import Shard

__all__ = [
    "ResultWriter",
    "clear_results",
    "completed",
    "database_path",
    "delete_runs",
    "load_metric",
    "load_runs",
    "merge_parts",
]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    config_id  TEXT NOT NULL,
    seed       INTEGER NOT NULL,
    config     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id     TEXT NOT NULL,
    name       TEXT NOT NULL,
    value      BLOB NOT NULL,
    PRIMARY KEY (run_id, name)
);
"""
_COLUMNS = "run_id, config_id, seed, config"

# Each machine keeps a read-only copy of the other's results: the cluster's
# under ``cluster/`` locally, and the local machine's under ``local/`` there.
_CLUSTER_MIRROR = "cluster"
_LOCAL_MIRROR = "local"


def database_path(experiment: Experiment, component: str) -> Path:
    """A component's merged database.

    Args:
        experiment: The experiment.
        component: The component.

    Returns:
        ``<results_dir>/<component>.db``.
    """
    return experiment.results_dir / f"{component}.db"


def _parts_dir(experiment: Experiment) -> Path:
    """The directory holding a sweep's per-worker parts."""
    return experiment.results_dir / f"{experiment.name}.parts"


def _part_path(experiment: Experiment, component: str, part: str) -> Path:
    """The database one worker, or one of its slots, writes during a sweep."""
    return _parts_dir(experiment) / component / f"part-{part}.db"


def _part_paths(experiment: Experiment, component: str) -> list[Path]:
    """Every per-worker part of a component currently on disk, in a stable order."""
    return sorted((_parts_dir(experiment) / component).glob("part-*.db"))


def _databases_in(root: Path, experiment: Experiment, component: str) -> list[Path]:
    """A component's merged database under ``root``, then its parts there."""
    parts = root / f"{experiment.name}.parts" / component
    return sorted(root.glob(f"{component}.db")) + sorted(parts.glob("part-*.db"))


def _mirrored_paths(experiment: Experiment, component: str) -> list[Path]:
    """Every database of a component mirrored from another machine."""
    roots = [experiment.results_dir / m for m in (_CLUSTER_MIRROR, _LOCAL_MIRROR)]
    return [p for root in roots for p in _databases_in(root, experiment, component)]


def _database_paths(experiment: Experiment, component: str) -> list[Path]:
    """Every database holding a component's runs: this machine's, then mirrors."""
    own = _databases_in(experiment.results_dir, experiment, component)
    return own + _mirrored_paths(experiment, component)


def _connect_write(path: str | Path) -> sqlite3.Connection:
    """Open a database for writing, creating its parent directory.

    Only ever one process writes a given file - a worker its own part, the merge
    step the merged database - so there is no cross-process write contention.

    Args:
        path: The database file.

    Returns:
        The open connection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60.0)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create the runs and metrics tables if they are not there yet."""
    conn.executescript(_SCHEMA)


def _query_ro(path: str | Path, sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read-only query, returning its rows.

    Returns ``[]`` if the database cannot be read cleanly - it may not exist
    yet, may lack the table, or may be a peer worker's part mid-write. A worker
    never needs a peer's rows, since the plan already gave them disjoint work,
    so a tolerated miss costs at most a recompute and never a wrong skip.
    """
    try:
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _to_blob(array: np.ndarray) -> bytes:
    """Serialise one metric as compressed ``.npy`` bytes."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return zlib.compress(buffer.getvalue())


def _from_blob(blob: bytes) -> np.ndarray:
    """Read back one stored metric."""
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


def completed(experiment: Experiment) -> dict[str, set[str]]:
    """List the runs an experiment has already stored, per component.

    The result is what a plan filters against, so it unions the merged database
    with every part still on disk: a run counts as done whether or not the sweep
    that produced it got as far as merging.

    Args:
        experiment: The experiment to look up.

    Returns:
        A ``{component name: {run id}}`` mapping, with an entry for every
        component the experiment declares even when it has no results yet.
    """
    return {
        component.name: {
            run
            for path in _database_paths(experiment, component.name)
            for (run,) in _query_ro(path, "SELECT run_id FROM runs")
        }
        for component in experiment.components
    }


class ResultWriter:
    """Collects one worker's results into its own part of the store.

    A component's part database is created on its first save, so a worker whose
    shards were all already stored leaves no file behind. Use it as a context
    manager, or close it when done.
    """

    def __init__(self, experiment: Experiment, part: str = "0") -> None:
        self._experiment = experiment
        self._part = part
        self._conns: dict[str, sqlite3.Connection] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def save(self, shard: Shard, results: Sequence[dict]) -> int:
        """Store one result per run of a shard.

        Storing a run that is already there is a no-op, so re-running a shard
        never overwrites or duplicates a result.

        Args:
            shard: The shard whose runs were computed.
            results: One ``{name: array}`` result per run, in shard order.

        Returns:
            The number of runs actually stored, excluding any already present.

        Raises:
            ValueError: If there is not exactly one result per run.
        """
        if len(results) != len(shard.runs):
            raise ValueError(
                f"shard of component {shard.component!r} has {len(shard.runs)} "
                f"run(s) but {len(results)} result(s) came back"
            )
        conn = self._connection(shard.component)
        stored = 0
        for run, result in zip(shard.runs, results, strict=True):
            inserted = conn.execute(
                f"INSERT OR IGNORE INTO runs ({_COLUMNS}) VALUES (?, ?, ?, ?)",
                (
                    run.id,
                    config_id(run.config),
                    int(run.seed),
                    json.dumps(as_point(run.config)),
                ),
            ).rowcount
            if not inserted:
                continue
            conn.executemany(
                "INSERT INTO metrics (run_id, name, value) VALUES (?, ?, ?)",
                [(run.id, name, _to_blob(value)) for name, value in result.items()],
            )
            stored += 1
        conn.commit()
        return stored

    def close(self) -> None:
        """Close every part database this writer opened."""
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()

    def _connection(self, component: str) -> sqlite3.Connection:
        """A component's part connection, opened on first use."""
        if component not in self._conns:
            conn = _connect_write(_part_path(self._experiment, component, self._part))
            _ensure_table(conn)
            self._conns[component] = conn
        return self._conns[component]


def merge_parts(experiment: Experiment) -> int:
    """Merge a sweep's per-worker parts into each component's database.

    Idempotent, and meant to run in a single process once a sweep's workers
    have finished. Parts are deleted only after they have been merged cleanly,
    so a failure part-way leaves the results where they are rather than losing
    them.

    Args:
        experiment: The experiment whose parts to merge.

    Returns:
        The number of runs newly merged in.
    """
    directory = _parts_dir(experiment)
    merged = 0
    for folder in sorted(p for p in directory.glob("*") if p.is_dir()):
        merged += _merge_component(experiment, folder.name)
        if not any(folder.iterdir()):
            folder.rmdir()
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
    return merged


def _merge_component(experiment: Experiment, component: str) -> int:
    """Merge one component's parts into its database, then delete them."""
    parts = _part_paths(experiment, component)
    if not parts:
        return 0

    conn = _connect_write(database_path(experiment, component))
    merged = 0
    try:
        _ensure_table(conn)
        for part in parts:
            conn.execute("ATTACH DATABASE ? AS part", (str(part),))
            merged += conn.execute(
                f"INSERT OR IGNORE INTO runs ({_COLUMNS}) "
                f"SELECT {_COLUMNS} FROM part.runs"
            ).rowcount
            conn.execute("INSERT OR IGNORE INTO metrics SELECT * FROM part.metrics")
            conn.commit()
            conn.execute("DETACH DATABASE part")
    finally:
        conn.close()

    for part in parts:
        part.unlink()
    return merged


# --- reading ----------------------------------------------------------------


def _flatten(config: dict, prefix: str = "") -> dict:
    """Flatten a nested config to dotted keys, one per hyperparameter.

    ``{"HYPERS": {"LR": 1e-3}}`` becomes ``{"HYPERS.LR": 1e-3}``.
    """
    flat: dict = {}
    for key, value in config.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def delete_runs(experiment: Experiment, component: str, run_ids: Sequence[str]) -> int:
    """Remove stored runs from the merged database and every part.

    Parts count towards what is done, so a run left in one would still be
    skipped by the next plan.

    Args:
        experiment: The experiment whose databases to modify.
        component: The component whose table to delete from.
        run_ids: The run ids to remove.

    Returns:
        The number of distinct runs actually deleted.
    """
    if not run_ids:
        return 0
    placeholders = ", ".join("?" for _ in run_ids)
    removed: set[str] = set()
    for path in _database_paths(experiment, component):
        conn = _connect_write(path)
        try:
            rows = conn.execute(
                f"DELETE FROM runs WHERE run_id IN ({placeholders}) RETURNING run_id",
                tuple(run_ids),
            ).fetchall()
            conn.execute(
                f"DELETE FROM metrics WHERE run_id IN ({placeholders})", tuple(run_ids)
            )
            conn.commit()
        finally:
            conn.close()
        removed.update(run for (run,) in rows)
    return len(removed)


def _mirrored_runs(experiment: Experiment, component: str) -> set[str]:
    """The runs of a component that another machine stored, as mirrored here."""
    return {
        run
        for path in _mirrored_paths(experiment, component)
        for (run,) in _query_ro(path, "SELECT run_id FROM runs")
    }


def clear_results(experiment: Experiment) -> None:
    """Remove every component's merged database and every part.

    Args:
        experiment: The experiment whose results to remove.
    """
    for path in experiment.results_dir.glob("*.db"):
        path.unlink()
    parts = _parts_dir(experiment)
    if parts.is_dir():
        shutil.rmtree(parts)


def load_runs(experiment: Experiment, component: str) -> pl.DataFrame:
    """Load one component's runs as a table.

    Args:
        experiment: The experiment to read.
        component: The component whose table to read.

    Returns:
        One row per run, identified by ``run_id``, ``config_id`` and ``seed``,
        followed by the config flattened to dotted columns such as
        ``AGENT_HYPERS.LR``. Empty if nothing is stored yet.
    """
    keys = ["run_id", "config_id", "seed"]
    seen: set[str] = set()
    rows: list[dict] = []
    for path in _database_paths(experiment, component):
        query = "SELECT run_id, config_id, seed, config FROM runs"
        for run, config, seed, point in _query_ro(path, query):
            if run in seen:
                continue
            seen.add(run)
            record = _flatten(json.loads(point))
            for key in keys:
                record.pop(key, None)
            rows.append(
                {"run_id": run, "config_id": config, "seed": seed, **record}
            )

    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    return frame.select([*keys, *[c for c in frame.columns if c not in keys]])


def load_metric(
    experiment: Experiment, component: str, name: str, run_ids: Sequence[str]
) -> list[np.ndarray | None]:
    """Load one metric of several runs.

    Args:
        experiment: The experiment to read.
        component: The component the runs belong to.
        name: The metric to load.
        run_ids: The runs to load it for.

    Returns:
        One array per run, in ``run_ids`` order, ``None`` where a run stored no
        such metric.

    Raises:
        KeyError: If the component has no stored run with one of the ids.
    """
    placeholders = ", ".join("?" for _ in run_ids)
    runs_query = f"SELECT run_id FROM runs WHERE run_id IN ({placeholders})"
    metric_query = (
        "SELECT run_id, value FROM metrics "
        f"WHERE name = ? AND run_id IN ({placeholders})"
    )
    known: set[str] = set()
    blobs: dict[str, bytes] = {}
    for path in _database_paths(experiment, component):
        known.update(run for (run,) in _query_ro(path, runs_query, tuple(run_ids)))
        for run, blob in _query_ro(path, metric_query, (name, *run_ids)):
            blobs.setdefault(run, blob)

    missing = [run for run in run_ids if run not in known]
    if missing:
        raise KeyError(f"{component!r} has no stored run(s) {missing}")
    return [_from_blob(blobs[run]) if run in blobs else None for run in run_ids]
