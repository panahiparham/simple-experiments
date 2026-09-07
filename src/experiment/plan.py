"""Working out which runs a component defines.

A component's runs are every point of its sweep applied to its base config,
crossed with every seed. Command-line overrides sit between the two: they
replace values in the base config, and a sweep then overrides them in turn, so
sweeping a path always wins over setting it.

Enumeration is deterministic - sweep values in the order they are declared, then
seeds in the order they are given - so the same component always produces the
same runs in the same order, whatever machine expands it.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

from experiment.design import Component, Experiment
from experiment.hypers import coerce_value, get_path, set_path, split_traced
from experiment.identity import config_id, run_id

__all__ = [
    "Run",
    "Shard",
    "expand_sweep",
    "sweep_configs",
    "component_runs",
    "batch_key",
    "pack_shards",
    "plan_experiment",
    "assign_shards",
]


@dataclasses.dataclass(frozen=True)
class Run:
    """One fully specified unit of work: a config and the seed to run it at.

    Attributes:
        config: The config, with every sweep value and override already applied.
        seed: The run's integer seed.
    """

    config: Any
    seed: int

    @property
    def id(self) -> str:
        """The run's stable id, which a store dedups on."""
        return run_id(self.config, self.seed)


def expand_sweep(sweep: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a sweep into every combination of its values.

    Args:
        sweep: Maps a dotted config path to the values to sweep it over.

    Returns:
        One ``{path: value}`` mapping per combination. An empty sweep yields a
        single empty mapping, i.e. the base config alone.

    >>> expand_sweep({"a": [1, 2], "b": ["x"]})
    [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]
    """
    paths = list(sweep)
    values = [list(sweep[p]) for p in paths]
    return [dict(zip(paths, combo)) for combo in itertools.product(*values)]


def sweep_configs(
    component: Component, overrides: dict[str, Any] | None = None
) -> list[Any]:
    """List the configs a component defines.

    Args:
        component: The component to expand.
        overrides: Values to set on the base config before sweeping. A path the
            sweep also names is overridden by the sweep. A value given as text
            is read as the type of the field it replaces, so command-line
            overrides describe the same run as the equivalent config.

    Returns:
        One config per sweep combination, of the same type as the base config.

    Raises:
        AttributeError: If an override or sweep path names no such field.
        ValueError: If a text override cannot be read as the field's type.
    """
    base = component.config
    for path, value in (overrides or {}).items():
        if isinstance(value, str):
            value = coerce_value(value, get_path(base, path))
        base = set_path(base, path, value)

    configs = []
    for combination in expand_sweep(component.sweep):
        config = base
        for path, value in combination.items():
            config = set_path(config, path, value)
        configs.append(config)
    return configs


def component_runs(
    component: Component, overrides: dict[str, Any] | None = None
) -> list[Run]:
    """List every run a component defines.

    Args:
        component: The component to expand.
        overrides: Values to set on the base config before sweeping.

    Returns:
        One :class:`Run` per config and seed, configs in sweep order and seeds
        in the order the component declares them.

    Raises:
        AttributeError: If an override or sweep path names no such field.
    """
    return [
        Run(config=config, seed=seed)
        for config in sweep_configs(component, overrides)
        for seed in component.seeds
    ]


def batch_key(config: Any) -> str:
    """Identify the set of runs a config can be batched with.

    Args:
        config: The run's config.

    Returns:
        An id for the config's static fields alone. Two configs share a key when
        they differ only in traced values, which is exactly when their runs can
        be computed together.
    """
    static, _ = split_traced(config)
    return config_id(static)


@dataclasses.dataclass(frozen=True)
class Shard:
    """A sequence of runs meant to be executed together.

    Every run in a shard shares a :func:`batch_key`, so the whole shard can be
    computed in one batched call. A shard is the unit of work a single worker
    process or cluster task picks up.

    Attributes:
        component: The name of the component the runs belong to, which is the
            table their results are stored in.
        runs: The runs, in enumeration order.
    """

    component: str
    runs: tuple[Run, ...]

    @property
    def configs(self) -> tuple[Any, ...]:
        """The config of each run, in order."""
        return tuple(run.config for run in self.runs)

    @property
    def seeds(self) -> tuple[int, ...]:
        """The seed of each run, in order."""
        return tuple(run.seed for run in self.runs)

    def __len__(self) -> int:
        return len(self.runs)


def pack_shards(
    component: str, runs: list[Run], shard_size: int | None = None
) -> list[Shard]:
    """Pack runs into shards that can each be computed in one batched call.

    Args:
        component: The name of the component the runs belong to.
        runs: The runs to pack, in enumeration order.
        shard_size: How many runs to put in a shard, or ``None`` to put every
            run that shares a batch key into a single shard.

    Returns:
        The shards, batch keys in the order they first appear and runs in the
        order they were given.
    """
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(batch_key(run.config), []).append(run)

    shards: list[Shard] = []
    for group in grouped.values():
        step = shard_size or len(group)
        for start in range(0, len(group), step):
            shards.append(
                Shard(component=component, runs=tuple(group[start:start + step]))
            )
    return shards


def plan_experiment(
    experiment: Experiment,
    *,
    done: dict[str, set[str]] | None = None,
    overrides: dict[str, Any] | None = None,
    components: list[str] | None = None,
    shard_size: int | None = None,
) -> list[Shard]:
    """Plan the outstanding work of an experiment, pooled across components.

    Runs already in ``done`` are dropped before anything is packed, so shards
    hold only work that still needs doing and a resumed sweep is as small as
    what is left. Because the packing depends on what was already finished, the
    plan is computed once and handed to the workers rather than recomputed by
    each of them, which would let workers that started at different times
    disagree about who runs what.

    Args:
        experiment: The experiment to plan.
        done: The ids of runs already stored, per component name. A run id
            covers a config and a seed but not the component, so two components
            sharing a config would otherwise be taken for one another's work.
        overrides: Values to set on each component's base config before
            sweeping.
        components: Names of the components to plan, or ``None`` for all of
            them.
        shard_size: Replaces every selected component's own shard size.

    Returns:
        The shards still to run, components in the order the experiment
        declares them.

    Raises:
        KeyError: If ``components`` names a component the experiment lacks.
        AttributeError: If an override or sweep path names no such field.
    """
    selected = (
        list(experiment.components)
        if components is None
        else [experiment.component(name) for name in components]
    )

    done = done or {}
    shards: list[Shard] = []
    for component in selected:
        stored = done.get(component.name, frozenset())
        pending = [
            run
            for run in component_runs(component, overrides)
            if run.id not in stored
        ]
        size = shard_size if shard_size is not None else component.shard_size
        shards.extend(pack_shards(component.name, pending, size))
    return shards


def assign_shards(shards: list[Shard], num_workers: int) -> list[list[Shard]]:
    """Divide a plan's shards among a pool of workers.

    Workers are dealt shards in turn, so their loads differ by at most one
    shard however many there are, and consecutive shards - which come from the
    same component and are therefore the most alike in cost - spread across
    different workers rather than piling onto one.

    Args:
        shards: The shards to divide, as planned.
        num_workers: How many workers share the plan.

    Returns:
        One list of shards per worker, in worker order. A worker with nothing
        to do gets an empty list.

    Raises:
        ValueError: If ``num_workers`` is less than 1.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1; got {num_workers}")
    return [shards[worker::num_workers] for worker in range(num_workers)]
