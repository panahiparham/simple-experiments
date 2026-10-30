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
from experiment.plan import plan_experiment
from experiment.results import (
    ResultWriter,
    _part_path,
    _parts_dir,
    _query_ro,
    clear_results,
    completed,
    database_path,
    delete_runs,
    load_metric,
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
    return {"reward": (np.arange(1, 5), np.arange(4.0) + seed)}


def run_everything(experiment: Experiment, num_workers: int = 1) -> None:
    """Compute and store every outstanding run, as a sweep would."""
    plan = plan_experiment(experiment, done=completed(experiment))
    for worker in range(num_workers):
        with ResultWriter(experiment, str(worker)) as writer:
            for shard in plan[worker::num_workers]:
                writer.save(shard, [result(r.seed) for r in shard.runs])


# --- layout -----------------------------------------------------------------


def test_each_database_is_named_after_its_component(experiment):
    assert database_path(experiment, "a").name == "a.db"


def test_parts_sit_in_the_experiments_parts_directory(experiment):
    part = _part_path(experiment, "a", "0")
    assert part.is_relative_to(_parts_dir(experiment))
    assert _parts_dir(experiment).parent == experiment.results_dir


def test_each_component_is_merged_into_its_own_database(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    stored = sorted(p.name for p in experiment.results_dir.glob("*.db"))
    assert stored == ["a.db", "b.db"]


# --- writing ----------------------------------------------------------------


def test_an_empty_store_has_completed_nothing(experiment):
    assert completed(experiment) == {"a": set(), "b": set()}


def test_a_worker_with_no_work_writes_no_file(experiment):
    writer = ResultWriter(experiment, "0")
    writer.close()
    assert not _parts_dir(experiment).exists()


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
    with ResultWriter(experiment, "0") as writer:
        first = writer.save(plan[0], [result(r.seed) for r in plan[0].runs])
        again = writer.save(plan[0], [result(r.seed) for r in plan[0].runs])
    assert (first, again) == (len(plan[0].runs), 0)


def test_a_result_per_run_is_required(experiment):
    shard = plan_experiment(experiment)[0]
    with (
        ResultWriter(experiment, "0") as writer,
        pytest.raises(ValueError, match="result"),
    ):
        writer.save(shard, [])


def test_a_run_may_produce_nothing(experiment):
    shard = plan_experiment(experiment)[0]
    with ResultWriter(experiment, "0") as writer:
        writer.save(shard, [{} for _ in shard.runs])
    run = shard.runs[0]
    assert load_metric(experiment, shard.component, "reward", [run.id]) == [None]
    assert run.id in completed(experiment)[shard.component]


def test_a_redundant_result_is_stored_compressed(experiment):
    shard = plan_experiment(experiment)[0]
    values = np.ones(100_000, dtype=np.float32)
    arrays = {"reward": (np.arange(values.size), values)}
    with ResultWriter(experiment, "0") as writer:
        writer.save(shard, [arrays for _ in shard.runs])
    stored = _query_ro(
        _part_path(experiment, shard.component, "0"),
        "SELECT value FROM metrics WHERE run_id = ?",
        (shard.runs[0].id,),
    )[0][0]
    assert len(stored) < values.nbytes / 10, (
        f"{len(stored)} bytes stored for {values.nbytes} bytes of array"
    )


@pytest.mark.parametrize(
    "series",
    [
        (np.arange(3), np.zeros(4)),
        (np.arange(3.0), np.zeros(3)),
        (np.arange(6).reshape(2, 3), np.zeros(2)),
        (np.arange(1), np.float64(0.0)),
    ],
    ids=["count mismatch", "float steps", "2-d steps", "scalar values"],
)
def test_steps_must_pair_up_with_values(experiment, series):
    """A metric whose steps do not give one integer step per value is refused,
    and nothing of the shard is stored.
    """
    shard = plan_experiment(experiment)[0]

    with (
        ResultWriter(experiment, "0") as writer,
        pytest.raises(ValueError, match="reward"),
    ):
        writer.save(shard, [{"reward": series} for _ in shard.runs])

    assert completed(experiment)[shard.component] == set()


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
    with ResultWriter(experiment, "0") as writer:
        for shard in plan:
            writer.save(shard, [result(r.seed) for r in shard.runs])
    assert merge_parts(experiment) == 2
    assert completed(experiment)["a"] == set()


def test_results_survive_a_merge(experiment):
    run_everything(experiment, num_workers=2)
    before = load_runs(experiment, "a").height
    merge_parts(experiment)
    assert load_runs(experiment, "a").height == before == 3


# --- deleting -----------------------------------------------------------------


def test_deleting_a_run_removes_it_from_completed(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    run_id = next(iter(completed(experiment)["a"]))
    assert delete_runs(experiment, "a", [run_id]) == 1
    assert run_id not in completed(experiment)["a"]


def test_deleting_a_run_removes_its_arrays(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    run_id = next(iter(completed(experiment)["a"]))
    delete_runs(experiment, "a", [run_id])
    query = "SELECT name FROM metrics WHERE run_id = ?"
    assert _query_ro(database_path(experiment, "a"), query, (run_id,)) == []


def test_deleting_leaves_other_runs_and_components_untouched(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    run_id = next(iter(completed(experiment)["a"]))
    delete_runs(experiment, "a", [run_id])
    assert len(completed(experiment)["a"]) == 2
    assert len(completed(experiment)["b"]) == 2


def test_deleting_an_unknown_run_id_is_a_no_op(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    assert delete_runs(experiment, "a", ["nosuchrun"]) == 0
    assert len(completed(experiment)["a"]) == 3


def test_deleting_from_an_empty_store_is_a_no_op(experiment):
    assert delete_runs(experiment, "a", ["nosuchrun"]) == 0


def test_deleting_nothing_is_a_no_op(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    assert delete_runs(experiment, "a", []) == 0


def test_deleting_before_a_merge_removes_the_run_from_its_part(experiment):
    run_everything(experiment)
    run_id = next(iter(completed(experiment)["a"]))
    assert delete_runs(experiment, "a", [run_id]) == 1
    assert run_id not in completed(experiment)["a"]


def test_a_run_in_both_the_database_and_a_part_counts_once(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    shard = plan_experiment(experiment)[0]
    with ResultWriter(experiment, "late") as writer:
        writer.save(shard, [result(r.seed) for r in shard.runs])
    run_id = shard.runs[0].id
    assert delete_runs(experiment, shard.component, [run_id]) == 1
    assert run_id not in completed(experiment)[shard.component]


def test_clearing_removes_merged_and_unmerged_runs(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    with ResultWriter(experiment, "late") as writer:
        shard = plan_experiment(experiment)[0]
        writer.save(shard, [result(r.seed) for r in shard.runs])
    clear_results(experiment)
    assert completed(experiment) == {"a": set(), "b": set()}


def test_clearing_removes_components_no_longer_declared(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    clear_results(dataclasses.replace(experiment, components=experiment.components[:1]))
    assert completed(experiment) == {"a": set(), "b": set()}


def test_clearing_an_empty_store_is_a_no_op(experiment):
    clear_results(experiment)
    assert completed(experiment) == {"a": set(), "b": set()}


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


def test_a_database_that_is_not_there_reads_as_empty(experiment):
    assert _query_ro(database_path(experiment, "a"), "SELECT run_id FROM runs") == []


def test_a_query_with_the_wrong_binding_count_is_not_swallowed(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    with pytest.raises(sqlite3.ProgrammingError):
        _query_ro(database_path(experiment, "a"), "SELECT * FROM runs WHERE run_id = ?")


# --- mirrors ----------------------------------------------------------------


def mirror(experiment: Experiment, name: str) -> Experiment:
    """The same experiment as the other machine stores it, mirrored here."""
    return dataclasses.replace(experiment, results_dir=experiment.results_dir / name)


@pytest.mark.parametrize("name", ["cluster", "local"])
def test_mirrored_runs_count_as_completed(experiment, name):
    run_everything(mirror(experiment, name))
    merge_parts(mirror(experiment, name))
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 3, "b": 2}


def test_a_mirrored_sweep_in_flight_counts_as_completed(experiment):
    run_everything(mirror(experiment, "cluster"), num_workers=2)
    assert len(completed(experiment)["a"]) == 3


def test_mirrored_runs_load_back(experiment):
    run_everything(mirror(experiment, "cluster"))
    run_id = load_runs(experiment, "a")["run_id"][0]
    [loaded] = load_metric(experiment, "a", "reward", [run_id])
    assert loaded is not None


def test_a_run_both_here_and_mirrored_loads_once(experiment):
    run_everything(experiment)
    run_everything(mirror(experiment, "cluster"))
    assert load_runs(experiment, "a").height == 3


def test_merging_leaves_a_mirror_in_place(experiment):
    run_everything(mirror(experiment, "cluster"), num_workers=2)
    assert merge_parts(experiment) == 0
    assert len(completed(experiment)["a"]) == 3


def test_an_unknown_directory_is_not_read_as_a_mirror(experiment):
    run_everything(mirror(experiment, "elsewhere"))
    assert completed(experiment) == {"a": set(), "b": set()}


# --- loading metrics --------------------------------------------------------


def run_ids_by_seed(experiment: Experiment, component: str) -> list[str]:
    return load_runs(experiment, component).sort("seed")["run_id"].to_list()


def test_a_metric_loads_one_series_per_run_in_the_order_asked(experiment):
    run_everything(experiment)
    merge_parts(experiment)
    run_ids = run_ids_by_seed(experiment, "a")[::-1]

    loaded = load_metric(experiment, "a", "reward", run_ids)

    assert [values.tolist() for _, values in loaded] == [
        [2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]
    ]


def test_a_metric_loads_with_its_steps_as_int64(experiment):
    """Steps come back as stored, widened to int64 whatever integer type the
    shard function used.
    """
    shard = plan_experiment(experiment)[0]
    steps = np.array([5, 9, 20], dtype=np.int32)
    with ResultWriter(experiment, "0") as writer:
        writer.save(shard, [{"loss": (steps, np.zeros(3))} for _ in shard.runs])

    [loaded] = load_metric(experiment, shard.component, "loss", [shard.runs[0].id])

    assert loaded is not None
    loaded_steps, _ = loaded
    assert loaded_steps.dtype == np.int64
    assert loaded_steps.tolist() == [5, 9, 20]


def test_a_metric_keeps_its_dtype_and_shape(experiment):
    shard = plan_experiment(experiment)[0]
    padded = np.arange(6, dtype=np.int32).reshape(2, 3)
    lengths = {"lengths": (np.arange(2), padded)}
    with ResultWriter(experiment, "0") as writer:
        writer.save(shard, [lengths for _ in shard.runs])

    [loaded] = load_metric(experiment, shard.component, "lengths", [shard.runs[0].id])

    assert loaded is not None
    _, values = loaded
    assert values.dtype == np.int32
    assert np.array_equal(values, padded)


def test_a_run_without_the_metric_loads_as_none(experiment):
    run_everything(experiment)
    run_ids = run_ids_by_seed(experiment, "a")

    assert load_metric(experiment, "a", "absent", run_ids) == [None, None, None]


def test_loading_a_metric_of_an_unknown_run_is_refused(experiment):
    run_everything(experiment)
    run_ids = [*run_ids_by_seed(experiment, "a"), "nosuchrun"]

    with pytest.raises(KeyError, match="nosuchrun"):
        load_metric(experiment, "a", "reward", run_ids)


def test_a_run_of_another_component_is_unknown(experiment):
    run_everything(experiment)
    run_ids = run_ids_by_seed(experiment, "b")

    with pytest.raises(KeyError):
        load_metric(experiment, "a", "reward", run_ids)


def test_loading_a_metric_of_no_runs_gives_nothing(experiment):
    assert load_metric(experiment, "a", "reward", []) == []


def test_a_metric_loads_across_parts_and_mirrors(experiment):
    plan = plan_experiment(experiment)
    with ResultWriter(experiment, "0") as writer:
        writer.save(plan[0], [result(r.seed) for r in plan[0].runs])
    run_everything(mirror(experiment, "cluster"))
    run_ids = run_ids_by_seed(experiment, "a")

    loaded = load_metric(experiment, "a", "reward", run_ids)

    assert [values[0] for _, values in loaded] == [0.0, 1.0, 2.0]
