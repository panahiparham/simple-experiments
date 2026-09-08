from __future__ import annotations

import dataclasses

from experiment.identity import as_point


@dataclasses.dataclass(frozen=True)
class Cfg:
    LR: float = 1e-3
    NAME: str = "dqn"


def test_a_dataclass_config_becomes_its_field_mapping():
    assert as_point(Cfg()) == {"LR": 1e-3, "NAME": "dqn"}


def test_a_mapping_config_is_used_as_it_stands():
    assert as_point({"LR": 0.1}) == {"LR": 0.1}


def test_a_mapping_config_is_copied_rather_than_shared():
    original = {"LR": 0.1}

    as_point(original)["LR"] = 0.2

    assert original == {"LR": 0.1}
