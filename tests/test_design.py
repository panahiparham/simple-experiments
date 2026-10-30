from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiment.design import Component, Experiment, SlurmResources
from experiment.plan import component_runs

NOT_IDENTIFIERS = [
    "9lives",
    "my-run",
    "my.run",
    "my run",
    "",
    'a"; DROP TABLE runs; --',
]


@pytest.mark.parametrize("name", NOT_IDENTIFIERS)
def test_a_component_name_that_is_not_an_identifier_is_rejected(name):
    with pytest.raises(ValueError, match="component name"):
        Component(name=name, config=None)


@pytest.mark.parametrize("name", NOT_IDENTIFIERS)
def test_an_experiment_name_that_is_not_an_identifier_is_rejected(name):
    with pytest.raises(ValueError, match="experiment name"):
        Experiment(
            name=name,
            components=[Component(name="a", config=None)],
            results_dir=Path("results"),
        )


def test_a_name_starting_with_an_underscore_is_allowed():
    assert Component(name="_hidden", config=None).name == "_hidden"


@pytest.mark.parametrize("size", [0, -1])
def test_a_shard_size_below_one_is_rejected(size):
    with pytest.raises(ValueError, match="at least 1"):
        Component(name="a", config=None, shard_size=size)


def test_a_shard_size_of_one_is_allowed():
    assert Component(name="a", config=None, shard_size=1).shard_size == 1


def test_a_component_runs_one_shard_at_a_time_by_default():
    assert Component(name="a", config=None).parallel_shards == 1


@pytest.mark.parametrize("parallel", [0, -1])
def test_parallel_shards_below_one_is_rejected(parallel):
    with pytest.raises(ValueError, match=r"parallel_shards=.*at least 1"):
        Component(name="a", config=None, parallel_shards=parallel)


# --- recording spec ---------------------------------------------------------


def test_a_component_records_nothing_by_default():
    """Without a spec a component asks for no particular metric."""
    assert Component(name="a", config=None).record == {}


@pytest.mark.parametrize("name", ["", 3, None])
def test_a_recorded_metric_name_must_be_a_non_empty_string(name):
    """A name the store could not file a metric under is refused up front."""
    with pytest.raises(ValueError, match="non-empty string"):
        Component(name="a", config=None, record={name: "every"})


def test_a_recording_spec_is_kept_as_given():
    """The harness keeps whatever the project describes each metric with, and
    a later change to the caller's mapping does not leak into the component.
    """
    spec = {"loss": ("window", 10), "return": None}
    component = Component(name="a", config=None, record=spec)
    spec["extra"] = None

    assert component.record == {"loss": ("window", 10), "return": None}


def test_a_recording_spec_leaves_run_ids_unchanged():
    """Asking for different metrics never changes which runs a component has."""
    plain = Component(name="a", config={"LR": 1.0}, seeds=[0, 1])
    recorded = Component(
        name="a", config={"LR": 1.0}, seeds=[0, 1], record={"loss": None}
    )

    assert [r.id for r in component_runs(recorded)] == [
        r.id for r in component_runs(plain)
    ]


def test_an_experiment_with_no_components_is_rejected():
    with pytest.raises(ValueError, match="defines no components"):
        Experiment(name="toy", components=[], results_dir=Path("results"))


def test_two_components_sharing_a_name_are_rejected():
    with pytest.raises(ValueError, match="more than once"):
        Experiment(
            name="toy",
            components=[
                Component(name="a", config=None),
                Component(name="a", config=None),
            ],
            results_dir=Path("results"),
        )


def slurm_experiment(slurm: SlurmResources) -> Experiment:
    return Experiment(
        name="toy",
        components=[Component(name="a", config=None)],
        results_dir=Path("results"),
        slurm=slurm,
    )


def test_an_unknown_slurm_key_is_rejected():
    typo: Any = {"tme": "1:00:00"}

    with pytest.raises(ValueError, match=r"unknown slurm key\(s\) \['tme'\]"):
        slurm_experiment(typo)


def test_known_slurm_keys_are_kept():
    slurm = SlurmResources(time="1:00:00", gpus=1, mps=True, account="def-a")

    assert slurm_experiment(slurm).slurm == slurm
