"""Tests for run enumeration and shard planning (``experiment.plan``).

Fake configs throughout, so these pin the planner's behaviour rather than any
particular experiment's hyperparameters.
"""

from __future__ import annotations

import dataclasses

import pytest

from experiment.design import Component, Experiment
from experiment.hypers import traced
from experiment.identity import config_id
from experiment.plan import (
    Phase,
    assign_slots,
    batch_key,
    component_runs,
    expand_sweep,
    pack_shards,
    plan_experiment,
    sweep_configs,
)


@dataclasses.dataclass(frozen=True)
class Hypers:
    LR: float = traced(3e-4)
    GAMMA: float = traced(0.99)
    HIDDEN: int = 64


@dataclasses.dataclass(frozen=True)
class Cfg:
    NAME: str = "dqn"
    HYPERS: Hypers = dataclasses.field(default_factory=Hypers)


def component(**kwargs) -> Component:
    kwargs.setdefault("name", "c")
    kwargs.setdefault("config", Cfg())
    kwargs.setdefault("seeds", [0, 1])
    return Component(**kwargs)


# --- sweep expansion --------------------------------------------------------


def test_expand_sweep_takes_the_product_of_its_values():
    assert expand_sweep({"a": [1, 2], "b": ["x"]}) == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
    ]


def test_an_empty_sweep_yields_the_base_config_alone():
    assert expand_sweep({}) == [{}]
    assert sweep_configs(component()) == [Cfg()]


def test_sweep_values_are_applied_to_the_base_config():
    comp = component(sweep={"HYPERS.LR": [1e-3, 5e-4]})
    assert [c.HYPERS.LR for c in sweep_configs(comp)] == [1e-3, 5e-4]


def test_an_override_replaces_the_base_value():
    got = sweep_configs(component(), {"HYPERS.GAMMA": 0.5})
    assert got[0].HYPERS.GAMMA == 0.5


def test_a_swept_path_wins_over_an_override():
    comp = component(sweep={"HYPERS.LR": [1e-3, 5e-4]})
    got = sweep_configs(comp, {"HYPERS.LR": 9.9})
    assert [c.HYPERS.LR for c in got] == [1e-3, 5e-4]


def test_an_override_off_the_sweep_survives_it():
    comp = component(sweep={"HYPERS.LR": [1e-3, 5e-4]})
    got = sweep_configs(comp, {"HYPERS.GAMMA": 0.5})
    assert [c.HYPERS.GAMMA for c in got] == [0.5, 0.5]


def test_an_unknown_override_path_is_rejected():
    with pytest.raises(AttributeError):
        sweep_configs(component(), {"HYPERS.NOPE": 1})


def test_a_text_override_is_read_as_the_fields_type():
    """A run's id hashes its values, so 1 and 1.0 must not be different runs."""
    from_text = sweep_configs(component(), {"HYPERS.LR": "1"})[0]
    from_python = sweep_configs(component(), {"HYPERS.LR": 1.0})[0]
    assert from_text == from_python
    assert config_id(from_text) == config_id(from_python)


def test_a_text_override_reaches_a_static_field_too():
    got = sweep_configs(component(), {"HYPERS.HIDDEN": "32"})[0]
    assert got.HYPERS.HIDDEN == 32


def test_a_text_override_that_does_not_fit_the_field_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        sweep_configs(component(), {"HYPERS.HIDDEN": "1.5"})


# --- runs -------------------------------------------------------------------


def test_runs_cross_every_config_with_every_seed():
    comp = component(sweep={"HYPERS.LR": [1e-3, 5e-4]}, seeds=[0, 1, 2])
    runs = component_runs(comp)
    assert len(runs) == 6
    assert [(r.config.HYPERS.LR, r.seed) for r in runs[:3]] == [
        (1e-3, 0), (1e-3, 1), (1e-3, 2)
    ]


def test_every_run_gets_its_own_id():
    comp = component(sweep={"HYPERS.LR": [1e-3, 5e-4]}, seeds=[0, 1, 2])
    runs = component_runs(comp)
    assert len({r.id for r in runs}) == len(runs)


def test_a_component_with_no_seeds_defines_no_runs():
    assert component_runs(component(seeds=[])) == []


# --- batching ---------------------------------------------------------------


def test_traced_differences_share_a_batch_key():
    a = Cfg(HYPERS=Hypers(LR=1e-3))
    b = Cfg(HYPERS=Hypers(LR=5e-4))
    assert batch_key(a) == batch_key(b)


def test_static_differences_split_the_batch_key():
    a = Cfg(HYPERS=Hypers(HIDDEN=64))
    b = Cfg(HYPERS=Hypers(HIDDEN=32))
    assert batch_key(a) != batch_key(b)


# --- packing ----------------------------------------------------------------


def test_shard_size_bounds_a_shard():
    runs = component_runs(component(sweep={"HYPERS.LR": [1e-3, 5e-4]}))
    shards = pack_shards("c", runs, 3)
    assert [len(s) for s in shards] == [3, 1]


def test_no_shard_size_packs_every_batchable_run_together():
    runs = component_runs(component(sweep={"HYPERS.LR": [1e-3, 5e-4]}))
    assert [len(s) for s in pack_shards("c", runs, None)] == [4]


def test_a_static_sweep_packs_into_one_shard_per_batch_key():
    runs = component_runs(component(sweep={"HYPERS.HIDDEN": [32, 64]}))
    shards = pack_shards("c", runs, None)
    assert [len(s) for s in shards] == [2, 2]
    assert {s.configs[0].HYPERS.HIDDEN for s in shards} == {32, 64}


def test_a_shard_exposes_its_configs_and_seeds_in_order():
    runs = component_runs(component(seeds=[3, 4]))
    shard = pack_shards("c", runs, None)[0]
    assert shard.seeds == (3, 4)
    assert len(shard.configs) == 2
    assert shard.component == "c"


def test_packing_nothing_yields_no_shards():
    assert pack_shards("c", [], None) == []


# --- planning ---------------------------------------------------------------


def experiment(**kwargs) -> Experiment:
    a = component(name="a", sweep={"HYPERS.LR": [1e-3, 5e-4]}, shard_size=2)
    b = component(name="b", seeds=[0, 1, 2], shard_size=2)
    kwargs.setdefault("name", "e")
    kwargs.setdefault("components", [a, b])
    kwargs.setdefault("results_dir", "results")
    return Experiment(**kwargs)


def test_a_plan_pools_components_in_declaration_order():
    shards = plan_experiment(experiment())
    assert [(s.component, len(s)) for s in shards] == [
        ("a", 2), ("a", 2), ("b", 2), ("b", 1)
    ]


def test_stored_runs_are_dropped_before_packing():
    """The remainder is repacked into full shards, not shards full of holes."""
    exp = experiment()
    done = {"a": {r.id for r in component_runs(exp.component("a"))[:2]}}
    shards = [s for s in plan_experiment(exp, done=done) if s.component == "a"]
    assert [len(s) for s in shards] == [2]


def test_a_fully_stored_component_plans_no_shards():
    exp = experiment()
    done = {"a": {r.id for r in component_runs(exp.component("a"))}}
    assert all(s.component != "a" for s in plan_experiment(exp, done=done))


def test_one_components_results_do_not_count_for_another():
    """Run ids cover a config and a seed, not the component that asked for it."""
    same = component(name="twin", seeds=[0, 1], shard_size=2)
    other = dataclasses.replace(same, name="twin2")
    exp = Experiment(name="e", components=[same, other], results_dir="results")
    done = {"twin": {r.id for r in component_runs(same)}}
    assert [s.component for s in plan_experiment(exp, done=done)] == ["twin2"]


def test_a_plan_can_be_restricted_to_some_components():
    shards = plan_experiment(experiment(), components=["b"])
    assert {s.component for s in shards} == {"b"}


def test_shard_size_overrides_every_component():
    shards = plan_experiment(experiment(), shard_size=1)
    assert [len(s) for s in shards] == [1] * 7


def test_overrides_reach_every_component():
    shards = plan_experiment(experiment(), overrides={"HYPERS.GAMMA": 0.5})
    assert all(c.HYPERS.GAMMA == 0.5 for s in shards for c in s.configs)


def test_planning_an_unknown_component_is_rejected():
    with pytest.raises(KeyError, match="nope"):
        plan_experiment(experiment(), components=["nope"])


# --- slot assignment --------------------------------------------------------


def slotted(**parallel: int) -> Experiment:
    a = component(name="a", sweep={"HYPERS.LR": [1e-3, 5e-4]}, shard_size=1)
    b = component(name="b", seeds=[0, 1, 2], shard_size=1)
    return experiment(
        components=[
            dataclasses.replace(c, parallel_shards=parallel.get(c.name, 1))
            for c in (a, b)
        ]
    )


def layout(workers: list[list[Phase]]) -> list[list[tuple[str, list[int]]]]:
    return [
        [(phase.component, [len(slot) for slot in phase.slots]) for phase in phases]
        for phases in workers
    ]


@pytest.mark.parametrize("num_workers", [1, 2, 3, 8])
@pytest.mark.parametrize("parallel", [1, 2, 3])
def test_slots_cover_the_plan_exactly_once(num_workers, parallel):
    exp = slotted(a=parallel, b=parallel)
    shards = plan_experiment(exp)
    workers = assign_slots(exp, shards, num_workers)
    flat = [
        shard
        for phases in workers
        for phase in phases
        for slot in phase.slots
        for shard in slot
    ]
    assert len(workers) == num_workers
    assert sorted(map(id, flat)) == sorted(map(id, shards))


@pytest.mark.parametrize("num_workers", [1, 2, 3, 8])
def test_one_slot_per_component_deals_shards_in_turn(num_workers):
    exp = slotted()
    shards = plan_experiment(exp)
    workers = assign_slots(exp, shards, num_workers)
    for index, phases in enumerate(workers):
        mine = [shard for phase in phases for shard in phase.slots[0]]
        assert mine == shards[index::num_workers]


def test_a_worker_fills_its_slots_before_the_next_worker_gets_any():
    exp = slotted(a=2)
    workers = assign_slots(exp, plan_experiment(exp), 2)
    assert layout(workers) == [
        [("a", [1, 1]), ("b", [2])],
        [("a", [1, 1]), ("b", [1])],
    ]


def test_each_component_gets_its_own_number_of_slots():
    exp = slotted(a=2)
    workers = assign_slots(exp, plan_experiment(exp), 1)
    assert layout(workers) == [[("a", [2, 2]), ("b", [3])]]


def test_parallel_shards_overrides_every_component():
    exp = slotted(a=2)
    workers = assign_slots(exp, plan_experiment(exp), 1, parallel_shards=3)
    assert layout(workers) == [[("a", [2, 1, 1]), ("b", [1, 1, 1])]]


def test_a_worker_gets_no_more_slots_than_shards():
    exp = slotted(b=8)
    workers = assign_slots(exp, plan_experiment(exp), 1)
    assert layout(workers) == [[("a", [4]), ("b", [1, 1, 1])]]


def test_a_worker_with_nothing_to_do_gets_no_phases():
    exp = slotted()
    assert assign_slots(exp, plan_experiment(exp), 8)[7] == []


def test_a_slotted_pool_needs_at_least_one_worker():
    with pytest.raises(ValueError, match="at least 1"):
        assign_slots(slotted(), [], 0)
