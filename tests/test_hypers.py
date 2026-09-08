"""Tests for the traced/static config split (``experiment.hypers``).

These use tiny fake configs rather than the project's real ones, so they pin the
split's behaviour without depending on which hyperparameters happen to be traced
today.
"""

from __future__ import annotations

import dataclasses

import pytest

from experiment.hypers import (
    TRACED,
    coerce_value,
    get_path,
    merge_traced,
    set_path,
    split_traced,
    traced,
    traced_paths,
)


@dataclasses.dataclass(frozen=True)
class Agent:
    LR: float = traced(3e-4)
    GAMMA: float = traced(0.99)
    HIDDEN: int = 64
    PRESET: str = "mlp"


@dataclasses.dataclass(frozen=True)
class Env:
    SETTING: str = "easy"
    CUTOFF: int = traced(1_000)


@dataclasses.dataclass(frozen=True)
class Cfg:
    NAME: str = "dqn"
    AGENT: Agent = dataclasses.field(default_factory=Agent)
    ENV: Env = dataclasses.field(default_factory=Env)


# --- the marker -------------------------------------------------------------


def test_traced_marks_only_the_fields_it_wraps():
    marked = {f.name: f.metadata.get(TRACED, False) for f in dataclasses.fields(Agent)}
    assert marked == {"LR": True, "GAMMA": True, "HIDDEN": False, "PRESET": False}


def test_traced_keeps_the_declared_default():
    assert Agent().LR == 3e-4
    assert Agent(LR=1.0).LR == 1.0


def test_traced_does_not_change_asdict():
    """Run ids hash asdict, so marking a field must not alter stored identity."""
    assert dataclasses.asdict(Cfg())["AGENT"]["LR"] == 3e-4
    assert set(dataclasses.asdict(Cfg())["AGENT"]) == {
        "LR", "GAMMA", "HIDDEN", "PRESET"
    }


# --- dotted paths -----------------------------------------------------------


def test_get_path_reads_nested_and_top_level_fields():
    cfg = Cfg(AGENT=Agent(LR=1e-3))
    assert get_path(cfg, "AGENT.LR") == 1e-3
    assert get_path(cfg, "NAME") == "dqn"


def test_set_path_copies_rather_than_mutates():
    cfg = Cfg()
    updated = set_path(cfg, "AGENT.LR", 1e-2)
    assert updated.AGENT.LR == 1e-2
    assert cfg.AGENT.LR == 3e-4


def test_set_path_leaves_sibling_fields_alone():
    updated = set_path(Cfg(), "AGENT.LR", 1e-2)
    assert updated.AGENT.HIDDEN == 64
    assert updated.ENV.SETTING == "easy"
    assert updated.NAME == "dqn"


@pytest.mark.parametrize("path", ["AGENT.NOPE", "NOPE", "NAME.LR"])
def test_unknown_path_raises_naming_the_path(path):
    with pytest.raises(AttributeError, match=path.split(".")[0]):
        get_path(Cfg(), path)
    with pytest.raises(AttributeError):
        set_path(Cfg(), path, 1)


# --- the split --------------------------------------------------------------


def test_traced_paths_are_listed_in_declaration_order():
    assert traced_paths(Cfg()) == ("AGENT.LR", "AGENT.GAMMA", "ENV.CUTOFF")


def test_split_collects_every_traced_value():
    cfg = Cfg(AGENT=Agent(LR=1e-3, GAMMA=0.9), ENV=Env(CUTOFF=500))
    _, dynamic = split_traced(cfg)
    assert dynamic == {"AGENT.LR": 1e-3, "AGENT.GAMMA": 0.9, "ENV.CUTOFF": 500}


def test_configs_differing_only_in_traced_values_share_a_static():
    """This is what lets two runs be batched into one shard."""
    a, _ = split_traced(Cfg(AGENT=Agent(LR=1e-3)))
    b, _ = split_traced(Cfg(AGENT=Agent(LR=5e-4)))
    assert a == b


def test_a_static_difference_keeps_configs_apart():
    a, _ = split_traced(Cfg(AGENT=Agent(HIDDEN=64)))
    b, _ = split_traced(Cfg(AGENT=Agent(HIDDEN=32)))
    assert a != b


def test_static_resets_traced_fields_to_their_defaults():
    static, _ = split_traced(Cfg(AGENT=Agent(LR=1e-3, GAMMA=0.9)))
    assert static.AGENT.LR == 3e-4
    assert static.AGENT.GAMMA == 0.99


def test_static_keeps_untraced_fields_untouched():
    static, _ = split_traced(Cfg(NAME="ddqn", AGENT=Agent(LR=1e-3, HIDDEN=32)))
    assert static.NAME == "ddqn"
    assert static.AGENT.HIDDEN == 32


def test_merge_restores_the_original_config():
    cfg = Cfg(AGENT=Agent(LR=1e-3, GAMMA=0.9), ENV=Env(CUTOFF=500))
    assert merge_traced(*split_traced(cfg)) == cfg


def test_merge_accepts_values_the_config_was_not_annotated_for():
    """Under vmap the traced values arrive as tracers, not floats."""
    static, _ = split_traced(Cfg())
    merged = merge_traced(static, {"AGENT.LR": object()})
    assert isinstance(merged.AGENT.LR, object)
    assert merged.AGENT.GAMMA == 0.99


def test_a_traced_field_without_a_default_is_rejected():
    @dataclasses.dataclass(frozen=True)
    class NoDefault:
        LR: float = dataclasses.field(metadata={TRACED: True})

    with pytest.raises(TypeError, match="declares no default"):
        split_traced(NoDefault(LR=1.0))


def test_a_config_with_no_traced_fields_splits_to_itself():
    @dataclasses.dataclass(frozen=True)
    class Plain:
        A: int = 1

    static, dynamic = split_traced(Plain())
    assert static == Plain()
    assert dynamic == {}


@pytest.mark.parametrize("text", ["true", "yes", "1", "TRUE", " true "])
def test_a_true_spelling_reads_as_true(text):
    assert coerce_value(text, False) is True


@pytest.mark.parametrize("text", ["false", "no", "0", "FALSE", " false "])
def test_a_false_spelling_reads_as_false(text):
    assert coerce_value(text, True) is False


def test_a_value_that_is_neither_true_nor_false_is_rejected_naming_it():
    with pytest.raises(ValueError, match="maybe"):
        coerce_value("maybe", True)
