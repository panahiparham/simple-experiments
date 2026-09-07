"""Tests for migrating the old per-component databases (``experiment.legacy``).

These build databases in the old shape by hand - a ``runs`` table with
``config_json``, ``metrics_json`` and ``curves`` - and check what comes out the
other side, since the point of the migration is that nothing has to be recomputed.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sqlite3

import numpy as np
import pytest

from experiment.design import Component, Experiment
from experiment.identity import config_id, run_id
from experiment.legacy import legacy_paths, migrate
from experiment.plan import component_runs, plan_experiment
from experiment.results import completed, load_result, load_runs

_LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    config_id    TEXT NOT NULL,
    seed         INTEGER NOT NULL,
    config_json  TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    curves       BLOB
);
"""


@dataclasses.dataclass(frozen=True)
class Hypers:
    LR: float = 3e-4


@dataclasses.dataclass(frozen=True)
class Cfg:
    NAME: str = "dqn"
    HYPERS: Hypers = dataclasses.field(default_factory=Hypers)


def _curves(seed: int) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, reward=np.arange(4.0) + seed)
    return buffer.getvalue()


def write_legacy(path, config, seeds) -> None:
    """Write a database in the shape the old store used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_SCHEMA)
    point = dataclasses.asdict(config)
    for seed in seeds:
        rid = run_id(config, seed)
        # The old store folded the run's identity into config_json.
        record = {**point, "seed": seed, "config_id": config_id(config),
                  "run_id": rid}
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            (rid, config_id(config), seed, json.dumps(record), "{}", _curves(seed)),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    return Experiment(
        name="toy",
        components=[
            Component(name="a", config=Cfg(), seeds=[0, 1, 2], shard_size=1),
            Component(name="b", config=Cfg(NAME="ddqn"), seeds=[0, 1], shard_size=1),
        ],
        results_dir=tmp_path,
    )


@pytest.fixture
def legacy(experiment) -> Experiment:
    for comp in experiment.components:
        write_legacy(
            experiment.results_dir / f"{comp.name}.db", comp.config, comp.seeds
        )
    return experiment


# --- finding the old files --------------------------------------------------


def test_nothing_to_find_in_a_fresh_directory(experiment):
    assert legacy_paths(experiment, "a") == []


def test_the_old_database_is_found(legacy):
    assert [p.name for p in legacy_paths(legacy, "a")] == ["a.db"]


def test_unmerged_old_parts_are_found_too(legacy):
    write_legacy(
        legacy.results_dir / "a.parts" / "part-0.db", Cfg(), [7]
    )
    assert [p.name for p in legacy_paths(legacy, "a")] == ["a.db", "part-0.db"]


# --- migrating --------------------------------------------------------------


def test_every_row_moves(legacy):
    assert migrate(legacy) == {"a": 3, "b": 2}


def test_a_component_with_nothing_to_move_is_absent(experiment):
    write_legacy(experiment.results_dir / "a.db", Cfg(), [0])
    assert migrate(experiment) == {"a": 1}


def test_migrating_nothing_moves_nothing(experiment):
    assert migrate(experiment) == {}


def test_migrating_twice_moves_nothing_the_second_time(legacy):
    migrate(legacy)
    assert migrate(legacy) == {"a": 0, "b": 0}


def test_old_parts_are_migrated_with_the_database(legacy):
    write_legacy(legacy.results_dir / "a.parts" / "part-0.db", Cfg(), [9])
    assert migrate(legacy)["a"] == 4


def test_the_old_files_are_left_alone(legacy):
    migrate(legacy)
    assert (legacy.results_dir / "a.db").exists()


# --- what the migrated results look like ------------------------------------


def test_migrated_runs_count_as_completed(legacy):
    migrate(legacy)
    assert {k: len(v) for k, v in completed(legacy).items()} == {"a": 3, "b": 2}


def test_nothing_recomputes_after_migrating(legacy):
    """The whole point: ids survive the move, so the work is still done."""
    migrate(legacy)
    assert plan_experiment(legacy, done=completed(legacy)) == []


def test_run_ids_are_unchanged_by_the_move(legacy):
    migrate(legacy)
    expected = {r.id for r in component_runs(legacy.component("a"))}
    assert set(load_runs(legacy, "a")["run_id"]) == expected


def test_curves_come_across_intact(legacy):
    migrate(legacy)
    frame = load_runs(legacy, "a").sort("seed")
    stored = load_result(legacy, "a", frame["run_id"][2])
    assert np.array_equal(stored["reward"], np.arange(4.0) + 2)


def test_the_config_loses_the_identity_the_old_store_folded_in(legacy):
    migrate(legacy)
    frame = load_runs(legacy, "a")
    assert set(frame.columns) == {
        "run_id", "config_id", "seed", "NAME", "HYPERS.LR"
    }


def test_components_land_in_their_own_tables(legacy):
    migrate(legacy)
    assert load_runs(legacy, "b")["NAME"].to_list() == ["ddqn", "ddqn"]
