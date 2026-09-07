"""Declaring which config fields may vary between the runs of one shard.

A config's fields are either *static* - they select the objects and shapes a run
is built from (an env name, a network preset, a buffer capacity) - or *traced*:
plain numbers a run reads as it goes, which may differ between the runs batched
together in a single shard. :func:`traced` is how a config declares the latter,
next to the hyperparameter it describes.

Marking a field traced does not require a sweep to vary it, and sweeping a static
field is not an error: runs that disagree on any static field simply land in
different shards.

A field is addressed by its dotted path from the top-level config, e.g.
``"AGENT_HYPERS.LR"`` - the same spelling a sweep and a ``--set`` override use.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import Any

__all__ = [
    "TRACED",
    "traced",
    "get_path",
    "set_path",
    "traced_paths",
    "split_traced",
    "merge_traced",
    "coerce_value",
]


TRACED = "experiment.traced"
"""The key marking a field traced, in its :func:`dataclasses.field` metadata."""


def traced(default: Any, **kwargs: Any) -> Any:
    """Declare a config field whose value may vary within one shard.

    Args:
        default: The field's default value.
        **kwargs: Passed through to :func:`dataclasses.field`.

    Returns:
        A dataclass field marked traced.
    """
    metadata = {**kwargs.pop("metadata", {}), TRACED: True}
    return dataclasses.field(default=default, metadata=metadata, **kwargs)


def _require_field(obj: Any, name: str, path: str) -> None:
    """Raise if ``obj`` is not a dataclass carrying a ``name`` field.

    Args:
        obj: The object the field is looked up on.
        name: The field name to require.
        path: The full dotted path, for the error message.

    Raises:
        AttributeError: If ``obj`` has no such field.
    """
    names = (
        {f.name for f in dataclasses.fields(obj)}
        if dataclasses.is_dataclass(obj)
        else set()
    )
    if name not in names:
        raise AttributeError(
            f"no config field {path!r}: {type(obj).__name__} has no {name!r}"
        )


def get_path(config: Any, path: str) -> Any:
    """Read a config field by its dotted path.

    Args:
        config: The config to read from.
        path: A dotted field path, e.g. ``"AGENT_HYPERS.LR"``.

    Returns:
        The field's value.

    Raises:
        AttributeError: If the path names no such field.
    """
    obj = config
    for part in path.split("."):
        _require_field(obj, part, path)
        obj = getattr(obj, part)
    return obj


def set_path(config: Any, path: str, value: Any) -> Any:
    """Copy a config with one dotted path set to a new value.

    Args:
        config: The config to copy.
        path: A dotted field path, e.g. ``"AGENT_HYPERS.LR"``.
        value: The value to set.

    Returns:
        A new config of the same type, with every other field unchanged.

    Raises:
        AttributeError: If the path names no such field.
    """
    head, _, rest = path.partition(".")
    _require_field(config, head, path)
    if rest:
        return dataclasses.replace(
            config, **{head: set_path(getattr(config, head), rest, value)}
        )
    return dataclasses.replace(config, **{head: value})


def _traced_fields(config: Any, prefix: str = ""):
    """Yield ``(dotted path, field)`` for every traced field, depth-first.

    Recursion stops at a traced field: a traced value is a number the run reads,
    never a nested config.
    """
    for field in dataclasses.fields(config):
        path = f"{prefix}{field.name}"
        if field.metadata.get(TRACED, False):
            yield path, field
            continue
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            yield from _traced_fields(value, f"{path}.")


def _default(field: dataclasses.Field, path: str) -> Any:
    """The declared default of a traced field.

    Raises:
        TypeError: If the field declares no default.
    """
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    raise TypeError(f"traced field {path!r} declares no default")


def traced_paths(config: Any) -> tuple[str, ...]:
    """List the dotted paths of a config's traced fields.

    Args:
        config: The config to inspect.

    Returns:
        The paths, in declaration order, so two configs of the same type always
        agree on the order.
    """
    return tuple(path for path, _ in _traced_fields(config))


def split_traced(config: Any) -> tuple[Any, dict[str, Any]]:
    """Separate a config into its static shape and its traced values.

    Args:
        config: The config to split.

    Returns:
        A ``(static, dynamic)`` pair. ``static`` is the config with every traced
        field reset to its declared default, so two configs that differ only in
        traced values give equal statics and can share a shard. ``dynamic`` maps
        each traced path to that config's value.

    Raises:
        TypeError: If a traced field declares no default.
    """
    dynamic: dict[str, Any] = {}
    static = config
    for path, field in _traced_fields(config):
        dynamic[path] = get_path(config, path)
        static = set_path(static, path, _default(field, path))
    return static, dynamic


def merge_traced(static: Any, dynamic: dict[str, Any]) -> Any:
    """Rebuild a config from its static shape and a set of traced values.

    Args:
        static: The static config, as returned by :func:`split_traced`.
        dynamic: A ``{dotted path: value}`` mapping. Values may be tracers, so
            the result is safe to build inside a ``jax`` transformation.

    Returns:
        A config of the same type as ``static``.

    Raises:
        AttributeError: If a path names no such field.
    """
    config = static
    for path, value in dynamic.items():
        config = set_path(config, path, value)
    return config


_TRUE = ("true", "yes", "1")
_FALSE = ("false", "no", "0")


def coerce_value(text: str, current: Any) -> Any:
    """Read a text value as the type of the field it replaces.

    A config's identity is a hash of its values, so ``1`` and ``1.0`` are
    different runs. Reading a command-line value as whatever type the field
    already holds is what stops an override from quietly describing a different
    run than the same config written out in Python.

    Args:
        text: The value as written on the command line.
        current: The value being replaced, whose type is the target.

    Returns:
        ``text`` read as the type of ``current``. A type this does not know is
        parsed as a Python literal.

    Raises:
        ValueError: If ``text`` cannot be read as that type.
    """
    if isinstance(current, bool):
        lowered = text.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"expected a true/false value, got {text!r}")
    if isinstance(current, int):
        number = float(text)
        if not number.is_integer():
            raise ValueError(f"expected a whole number, got {text!r}")
        return int(number)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, str):
        return text
    return ast.literal_eval(text)
