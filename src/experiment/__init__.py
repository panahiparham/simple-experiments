"""Lightweight, domain-agnostic experiment infrastructure.

An :class:`Experiment` is a name, a results directory, and one or more
:class:`Component` s. A component is a base config, a sweep over it, and the
seeds to run each point at. A project supplies those plus one function that
computes a shard of runs; the harness works out which runs are outstanding,
packs them into shards, hands each shard to that function, and stores what comes
back. It never computes a run itself.

    run(EXPERIMENT, process_shard)

:mod:`experiment.commands` is the command line every experiment's ``run.py``
shares; :mod:`experiment.results` is the store to read results back from.
"""

from experiment.commands import run
from experiment.design import Component, Experiment
from experiment.hypers import TRACED, traced
from experiment.identity import config_id, run_id
from experiment.plan import Run, Shard
from experiment.results import (
    completed,
    database_path,
    load_array,
    load_result,
    load_runs,
    merge_parts,
)
from experiment.runner import ShardFn

__all__ = [
    "Component",
    "Experiment",
    "Run",
    "Shard",
    "ShardFn",
    "TRACED",
    "traced",
    "config_id",
    "run_id",
    "run",
    "completed",
    "database_path",
    "load_runs",
    "load_result",
    "load_array",
    "merge_parts",
]
