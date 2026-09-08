from __future__ import annotations

from pathlib import Path

import pytest

from experiment.design import Component, Experiment

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
