"""Stable identity for a configuration and a run.

The seed is a first-class axis: ``config_id`` hashes the config with the seed
excluded, so one config keeps a single id across all of its seeds, and
``run_id`` pairs the two. A run's PRNG derives from its integer seed alone, so a
result is reproducible regardless of how the work was sharded across workers.

These ids are what a store dedups on, so their spelling is a compatibility
surface: changing how a config hashes orphans every result already computed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np

__all__ = ["config_id", "run_id", "as_point"]


def _canonical(hypers: dict[str, Any]) -> str:
    """Deterministic JSON for a hyper dict: sorted keys, floats rounded to 12
    significant digits (so float noise doesn't change the id). Recurses into
    nested dicts (e.g. ``{"OPTIM": {...}, "MODEL": {...}}``)."""

    def clean(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(f"{float(v):.12g}")
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {k: clean(v[k]) for k in sorted(v)}
        return v

    return json.dumps(clean(hypers), sort_keys=True)


def as_point(config: Any) -> dict[str, Any]:
    """The identity and storage dict for a config.

    Args:
        config: A config object or its ``dataclasses.asdict`` mapping.

    Returns:
        ``dataclasses.asdict(config)`` for a dataclass, else the mapping itself.
    """
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return dict(config)


def config_id(config: Any) -> str:
    """Compute the stable id for a configuration.

    Args:
        config: A config object or its ``dataclasses.asdict`` mapping.

    Returns:
        An 8-character id, excluding the seed, so a config keeps one id across
        all of its seeds.
    """
    digest = hashlib.blake2b(_canonical(as_point(config)).encode(), digest_size=4)
    return digest.hexdigest()


def run_id(config: Any, seed: int) -> str:
    """Compute the stable id for one configuration and seed.

    Args:
        config: A config object or its ``dataclasses.asdict`` mapping.
        seed: The run's integer seed.

    Returns:
        The id ``"<config_id>_s<seed>"``.
    """
    return f"{config_id(config)}_s{int(seed)}"
