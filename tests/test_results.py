"""Tests for the per-experiment result store (``experiment.results``).

A fake config and hand-made results, so these exercise storage alone - layout,
dedup, per-worker parts, merging and reading back.
"""

from __future__ import annotations

import dataclasses
import sqlite3

import numpy as np
import pytest

from experiment.design import Component, Experiment
from experiment.hypers import traced
from experiment.plan import assign_shards, plan_experiment
from experiment.results import (
    ResultWriter,
    _parts_dir,
    _part_path,
    _query_ro,
    completed,
    database_path,
    load_array,
    load_result,
    load_runs,
    merge_parts,
)


@dataclasses.dataclass(frozen=True)
class Hypers:
    LR: float = traced(3e-4)
    HIDDEN: int = 64


@dataclasses.dataclass(frozen=True)
class Cfg:
    NAME: str = "dqn"
    HYPERS: Hypers = dataclasses.field(default_factory=Hypers)


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


def result(seed: int) -> dict:
    return {"reward": np.arange(4.0) + seed}


def run_everything(experiment: Experiment, num_workers: int = 1) -> None:
    """Compute and store every outstanding run, as a sweep would."""
    plan = plan_experiment(experiment, done=completed(experiment))
    for worker, shards in enumerate(assign_shards(plan, num_workers)):
        with ResultWriter(experiment, worker) as writer:
            for shard in shards:
                writer.save(shard, [result(r.seed) for r in shard.runs])


# --- layout -----------------------------------------------------------------


def test_the_database_is_named_after_the_experiment(experiment):
    assert database_path(experiment).name == "toy.db"


def test_parts_sit_beside_the_database(experiment):
    assert _part_path(experiment, 0).parent == _parts_dir(experiment)
    assert _parts_dir(experiment).parent == experiment.results_dir


# --- writing ----------------------------------------------------------------


def test_an_empty_store_has_completed_nothing(experiment):
    assert completed(experiment) == {"a": set(), "b": set()}


def test_a_worker_with_no_work_writes_no_file(experiment):
    writer = ResultWriter(experiment, 0)
    writer.close()
    assert not writer.path.exists()


def test_saved_runs_are_reported_as_completed(experiment):
    run_everything(experiment)
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 3, "b": 2}


def test_results_count_before_they_are_merged(experiment):
    run_everything(experiment)
    assert _parts_dir(experiment).is_dir()
    assert len(completed(experiment)["a"]) == 3


def test_a_finished_experiment_plans_no_more_work(experiment):
    run_everything(experiment)
    assert plan_experiment(experiment, done=completed(experiment)) == []


def test_saving_a_run_twice_stores_it_once(experiment):
    plan = plan_experiment(experiment)
    with ResultWriter(experiment, 0) as writer:
        first = writer.save(plan[0], [result(r.seed) for r in plan[0].runs])
        again = writer.save(plan[0], [result(r.seed) for r in plan[0].runs])
    assert (first, again) == (len(plan[0].runs), 0)


def test_a_result_per_run_is_required(experiment):
    shard = plan_experiment(experiment)[0]
    with ResultWriter(experiment, 0) as writer:
        with pytest.raises(ValueError, match="result"):
            writer.save(shard, [])


def test_a_run_may_produce_nothing(experiment):
    shard = plan_experiment(experiment)[0]
    with ResultWriter(experiment, 0) as writer:
        writer.save(shard, [{} for _ in shard.runs])
    run = shard.runs[0]
    assert load_result(experiment, shard.component, run.id) == {}
    assert run.id in completed(experiment)[shard.component]


def test_a_redundant_result_is_stored_compressed(experiment):
    shard = plan_experiment(experiment)[0]
    arrays = {"reward": np.ones(100_000, dtype=np.float32)}
    with ResultWriter(experiment, 0) as writer:
        writer.save(shard, [arrays for _ in shard.runs])
    stored = _query_ro(
        writer.path,
        f'SELECT result FROM "{shard.component}" WHERE run_id = ?',
        (shard.runs[0].id,),
    )[0][0]
    assert len(stored) < arrays["reward"].nbytes / 10, (
        f"{len(stored)} bytes stored for {arrays['reward'].nbytes} bytes of array"
    )


# --- merging ----------------------------------------------------------------


def test_merging_folds_every_worker_in(experiment):
    run_everything(experiment, num_workers=3)
    assert merge_parts(experiment) == 5
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 3, "b": 2}


def test_merging_clears_the_parts_directory(experiment):
    run_everything(experiment, num_workers=3)
    merge_parts(experiment)
    assert not _parts_dir(experiment).exists()


def test_merging_again_is_a_no_op(experiment):
    run_everything(experiment, num_workers=2)
    merge_parts(experiment)
    assert merge_parts(experiment) == 0


def test_merging_nothing_is_a_no_op(experiment):
    assert merge_parts(experiment) == 0


def test_a_worker_that_touched_one_component_merges_cleanly(experiment):
    plan = [s for s in plan_experiment(experiment) if s.component == "b"]
    with ResultWriter(experiment, 0) as writer:
        for shard in plan:
            writer.save(shard, [result(r.seed) for r in shard.runs])
    assert merge_parts(experiment) == 2
    assert completed(experiment)["a"] == set()


def test_results_survive_a_merge(experiment):
    run_everything(experiment, num_workers=2)
    before = load_runs(experiment, "a").height
    merge_parts(experiment)
    assert load_runs(experiment, "a").height == before == 3


# --- reading ----------------------------------------------------------------


def test_reading_an_empty_component_gives_an_empty_frame(experiment):
    assert load_runs(experiment, "a").is_empty()


def test_runs_load_with_the_config_flattened_to_columns(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    frame = load_runs(experiment, "a")
    assert frame.height == 3
    assert frame.columns[:3] == ["run_id", "config_id", "seed"]
    assert set(frame.columns) == {
        "run_id", "config_id", "seed", "NAME", "HYPERS.LR", "HYPERS.HIDDEN"
    }


def test_runs_can_be_ordered_by_seed(experiment):
    """Analysis stacks a component's seeds in order, so the column must be real."""
    run_everything(experiment)
    assert load_runs(experiment, "a").sort("seed")["seed"].to_list() == [0, 1, 2]


def test_components_are_read_separately(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    assert load_runs(experiment, "b")["NAME"].to_list() == ["ddqn", "ddqn"]


def test_a_runs_arrays_load_back_unchanged(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    for run_id in load_runs(experiment, "a")["run_id"]:
        stored = load_result(experiment, "a", run_id)
        assert set(stored) == {"reward"}
        assert stored["reward"].shape == (4,)


def test_one_array_can_be_loaded_by_name(experiment):
    run_everything(experiment)
    run_id = load_runs(experiment, "a")["run_id"][0]
    assert load_array(experiment, "a", run_id, "reward") is not None
    assert load_array(experiment, "a", run_id, "absent") is None


def test_an_unknown_run_reads_as_nothing(experiment):
    run_everything(experiment)
    assert load_result(experiment, "a", "nosuchrun") == {}


def test_a_database_that_is_not_there_reads_as_empty(experiment):
    assert _query_ro(database_path(experiment), 'SELECT run_id FROM "a"') == []


def test_a_query_with_the_wrong_binding_count_is_not_swallowed(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    with pytest.raises(sqlite3.ProgrammingError):
        _query_ro(database_path(experiment), 'SELECT * FROM "a" WHERE run_id = ?')
