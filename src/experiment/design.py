"""What an experiment is.

A research project has several experiments. An :class:`Experiment` is one
self-contained unit of work: a name, the directory its results live in, and one
or more :class:`Component` s. A component defines a set of runs - a base config,
a sweep over it, and the seeds to run each point at - and owns its own table in
the experiment's database, so components that vary in ways a sweep cannot express
are collected and analysed separately.

An experiment's name is also the section a cluster config looks up for its
resources, so it is stable and identifier-like, as is a component's name: both
become filenames and SQL identifiers.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["Component", "Experiment"]


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _require_identifier(name: str, kind: str) -> None:
    """Raise unless ``name`` is usable as a filename and a SQL identifier.

    Args:
        name: The name to check.
        kind: What is being named, for the error message.

    Raises:
        ValueError: If the name is not identifier-like.
    """
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(
            f"{kind} name {name!r} must be a letter or underscore followed by "
            "letters, digits or underscores"
        )


@dataclasses.dataclass(frozen=True)
class Component:
    """One named sub-experiment: a base config, a sweep over it, and seeds.

    The runs a component defines are every point of its ``sweep`` applied to
    ``config``, crossed with every seed. ``shard_size`` is how many of those runs
    are packed into one shard; ``None`` packs each group of runs that can be
    batched together into a single shard.

    Attributes:
        name: Identifies the component, and names its table in the experiment's
            database.
        config: The base config every run starts from.
        sweep: Maps a dotted config path to the values to sweep it over. Each
            combination overrides ``config`` for one set of runs.
        seeds: The seeds every swept config is run at.
        shard_size: How many runs to pack into a shard, or ``None`` for as many
            as can be batched together.
    """

    name: str
    config: Any
    sweep: dict[str, list] = dataclasses.field(default_factory=dict)
    seeds: Sequence[int] = ()
    shard_size: int | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.name, "component")
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        object.__setattr__(self, "sweep", dict(self.sweep))
        if self.shard_size is not None and self.shard_size < 1:
            raise ValueError(
                f"component {self.name!r} has shard_size={self.shard_size}; "
                "it must be at least 1, or None"
            )


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A named set of components and the directory their results live in.

    Attributes:
        name: Identifies the experiment. Names its database, and the section a
            cluster config looks up for this experiment's resources.
        components: The components making up the experiment, in the order they
            are run.
        results_dir: The directory holding the experiment's database.
    """

    name: str
    components: Sequence[Component]
    results_dir: Path

    def __post_init__(self) -> None:
        _require_identifier(self.name, "experiment")
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        if not self.components:
            raise ValueError(f"experiment {self.name!r} defines no components")
        names = [c.name for c in self.components]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(
                f"experiment {self.name!r} defines component(s) {duplicated} "
                "more than once; each component needs its own name"
            )

    def component(self, name: str) -> Component:
        """Look up one of the experiment's components by name.

        Args:
            name: The component's name.

        Returns:
            The component.

        Raises:
            KeyError: If the experiment defines no such component.
        """
        for comp in self.components:
            if comp.name == name:
                return comp
        defined = sorted(c.name for c in self.components)
        raise KeyError(
            f"experiment {self.name!r} has no component {name!r}; "
            f"defined: {defined}"
        )
