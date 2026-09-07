"""Computing shards and storing what comes back.

The harness never computes a run itself. It decides which runs are outstanding,
packs them into shards, hands each shard to the function the project supplied,
and stores the results. This module is that loop, and the only place the two
sides meet.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from experiment.design import Experiment
from experiment.plan import Shard
from experiment.results import ResultWriter

__all__ = ["ShardFn", "run_shards"]


# process(configs, seeds) -> one result mapping per run, in the order given.
# Everything the harness knows about how a run is computed is on this line.
ShardFn = Callable[[Sequence[Any], Sequence[int]], Sequence[dict]]


def run_shards(
    experiment: Experiment,
    shards: Sequence[Shard],
    process: ShardFn,
    *,
    worker: int = 0,
) -> int:
    """Compute a list of shards and store their results.

    Results are written and committed a shard at a time, so a run that is
    interrupted keeps everything its finished shards produced.

    Args:
        experiment: The experiment the shards belong to.
        shards: The shards this worker is to compute, in order.
        process: The project's shard function.
        worker: This worker's index, which picks the part it writes to.

    Returns:
        The number of runs stored.

    Raises:
        ValueError: If ``process`` does not return one result per run.
    """
    saved = 0
    with ResultWriter(experiment, worker) as writer:
        for shard in shards:
            results = process(shard.configs, shard.seeds)
            saved += writer.save(shard, list(results))
    return saved
