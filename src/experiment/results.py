"""Where an experiment's results live.

One SQLite database per experiment, holding one table per component and one row
per run: the run's id, the id of its config, its seed, the config itself, and
whatever the run produced as a binary blob.

A shared SQLite file is not safe under concurrent writes, and on a cluster
filesystem it is worse, so a worker never writes the experiment's database
directly. It writes its own part beside it, and the parts are merged in at the
end of a sweep. Reads always union the database with any parts still present, so
results from a sweep that was interrupted before its merge are still visible and
still count as done.

    results/
        pinball.db              # merged: one table per component
        pinball.parts/
            part-0.db           # transient, one per worker
            part-1.db
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from experiment.design import Experiment
from experiment.identity import as_point, config_id
from experiment.plan import Shard

__all__ = [
    "database_path",
    "completed",
    "ResultWriter",
    "merge_parts",
    "load_runs",
    "load_result",
    "load_array",
]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS "{component}" (
    run_id     TEXT PRIMARY KEY,
    config_id  TEXT NOT NULL,
    seed       INTEGER NOT NULL,
    config     TEXT NOT NULL,
    result     BLOB
);
"""
_COLUMNS = "run_id, config_id, seed, config, result"


def database_path(experiment: Experiment) -> Path:
    """The experiment's merged database.

    Args:
        experiment: The experiment.

    Returns:
        ``<results_dir>/<name>.db``.
    """
    return experiment.results_dir / f"{experiment.name}.db"


def _parts_dir(experiment: Experiment) -> Path:
    """The directory holding a sweep's per-worker parts."""
    return experiment.results_dir / f"{experiment.name}.parts"


def _part_path(experiment: Experiment, worker: int) -> Path:
    """The database one worker writes during a sweep."""
    return _parts_dir(experiment) / f"part-{int(worker)}.db"


def _part_paths(experiment: Experiment) -> list[Path]:
    """Every per-worker part currently on disk, in a stable order."""
    parts = _parts_dir(experiment)
    return sorted(parts.glob("part-*.db")) if parts.is_dir() else []


def _database_paths(experiment: Experiment) -> list[Path]:
    """Every database holding results: the merged one, then any parts."""
    merged = database_path(experiment)
    return ([merged] if merged.exists() else []) + _part_paths(experiment)


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


def _ensure_table(conn: sqlite3.Connection, component: str) -> None:
    """Create a component's table if it is not there yet.

    Component names are validated as identifiers where they are declared, so
    interpolating one into the schema cannot inject SQL.
    """
    conn.executescript(_SCHEMA.format(component=component))


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


def _to_blob(result: dict[str, np.ndarray]) -> bytes | None:
    """Serialise one run's result arrays.

    Args:
        result: The run's named arrays.

    Returns:
        The arrays as npz bytes, or ``None`` if the run produced none.
    """
    if not result:
        return None
    buffer = io.BytesIO()
    np.savez(buffer, **{k: np.asarray(v) for k, v in result.items()})
    return buffer.getvalue()


def _from_blob(blob: bytes | None) -> dict[str, np.ndarray]:
    """Read back a result blob.

    Args:
        blob: The stored bytes, or ``None``.

    Returns:
        The run's named arrays, empty if there were none.
    """
    if not blob:
        return {}
    with np.load(io.BytesIO(blob)) as data:
        return {name: data[name] for name in data.files}


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
    done: dict[str, set[str]] = {c.name: set() for c in experiment.components}
    for path in _database_paths(experiment):
        for component, ids in done.items():
            ids.update(
                row[0]
                for row in _query_ro(path, f'SELECT run_id FROM "{component}"')
            )
    return done


class ResultWriter:
    """Collects one worker's results into its own part of the store.

    The part database is created on the first save, so a worker whose shards
    were all already stored leaves no file behind. Use it as a context manager,
    or close it when done.

    Attributes:
        path: The part database this writer owns.
    """

    def __init__(self, experiment: Experiment, worker: int = 0) -> None:
        self.path = _part_path(experiment, worker)
        self._conn: sqlite3.Connection | None = None
        self._tables: set[str] = set()

    def __enter__(self) -> "ResultWriter":
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
        conn = self._connection()
        if shard.component not in self._tables:
            _ensure_table(conn, shard.component)
            self._tables.add(shard.component)

        before = conn.total_changes
        for run, result in zip(shard.runs, results):
            conn.execute(
                f'INSERT OR IGNORE INTO "{shard.component}" ({_COLUMNS}) '
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run.id,
                    config_id(run.config),
                    int(run.seed),
                    json.dumps(as_point(run.config)),
                    _to_blob(result),
                ),
            )
        conn.commit()
        return conn.total_changes - before

    def close(self) -> None:
        """Close the part database, if one was ever opened."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> sqlite3.Connection:
        """The part's connection, opened on first use."""
        if self._conn is None:
            self._conn = _connect_write(self.path)
        return self._conn


def _tables(path: str | Path) -> list[str]:
    """The component tables a database holds, ignoring SQLite's own."""
    rows = _query_ro(path, "SELECT name FROM sqlite_master WHERE type = 'table'")
    return [name for (name,) in rows if not name.startswith("sqlite_")]


def merge_parts(experiment: Experiment) -> int:
    """Merge a sweep's per-worker parts into the experiment's database.

    Idempotent, and meant to run in a single process once a sweep's workers
    have finished. Parts are deleted only after they have been merged cleanly,
    so a failure part-way leaves the results where they are rather than losing
    them.

    Args:
        experiment: The experiment whose parts to merge.

    Returns:
        The number of runs newly merged in.
    """
    parts = _part_paths(experiment)
    if not parts:
        return 0

    conn = _connect_write(database_path(experiment))
    merged = 0
    try:
        for part in parts:
            tables = _tables(part)
            for table in tables:
                _ensure_table(conn, table)
            conn.execute("ATTACH DATABASE ? AS part", (str(part),))
            before = conn.total_changes
            for table in tables:
                conn.execute(
                    f'INSERT OR IGNORE INTO "{table}" ({_COLUMNS}) '
                    f'SELECT {_COLUMNS} FROM part."{table}"'
                )
            conn.commit()
            conn.execute("DETACH DATABASE part")
            merged += conn.total_changes - before
    finally:
        conn.close()

    for part in parts:
        part.unlink()
    directory = _parts_dir(experiment)
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
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
    for path in _database_paths(experiment):
        query = f'SELECT run_id, config_id, seed, config FROM "{component}"'
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


def load_result(
    experiment: Experiment, component: str, run_id: str
) -> dict[str, np.ndarray]:
    """Load everything one run produced.

    Args:
        experiment: The experiment to read.
        component: The component the run belongs to.
        run_id: The run to load.

    Returns:
        The run's named arrays, empty if it stored none.
    """
    query = f'SELECT result FROM "{component}" WHERE run_id = ?'
    for path in _database_paths(experiment):
        rows = _query_ro(path, query, (run_id,))
        if rows and rows[0][0] is not None:
            return _from_blob(rows[0][0])
    return {}


def load_array(
    experiment: Experiment, component: str, run_id: str, name: str
) -> np.ndarray | None:
    """Load one named array from a run's result.

    Args:
        experiment: The experiment to read.
        component: The component the run belongs to.
        run_id: The run to load.
        name: The array to load.

    Returns:
        The array, or ``None`` if the run has no such array.
    """
    return load_result(experiment, component, run_id).get(name)
