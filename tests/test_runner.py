from __future__ import annotations

import dataclasses

import numpy as np

from experiment.design import Component, Experiment
from experiment.plan import plan_experiment
from experiment.runner import run_shards


@dataclasses.dataclass(frozen=True)
class Cfg:
    LR: float = 1.0


def test_each_shard_gets_its_components_recording_spec(tmp_path):
    """A shard function is handed the spec of the component its shard belongs
    to, so components of one experiment can ask for different metrics.
    """
    experiment = Experiment(
        name="toy",
        components=[
            Component(name="a", config=Cfg(1.0), seeds=[0], record={"loss": 1}),
            Component(name="b", config=Cfg(2.0), seeds=[0], record={"q": 2}),
        ],
        results_dir=tmp_path,
    )
    seen = {}

    def process(configs, seeds, record):
        seen[configs[0].LR] = record
        return [{"x": (np.arange(1), np.zeros(1))} for _ in seeds]

    run_shards(experiment, plan_experiment(experiment), process)

    assert seen == {1.0: {"loss": 1}, 2.0: {"q": 2}}
