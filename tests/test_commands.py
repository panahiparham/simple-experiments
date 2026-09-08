"""Tests for the shared command line (``experiment.commands``).

A fake config and a fake shard function, so these exercise the CLI itself.
The multi-worker sweep is the exception: it writes a real ``run.py`` and
invokes it, because spawning workers and handing them the plan is the part that
only happens across processes.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import experiment as experiment_package
from experiment.commands import parse_overrides, run
from experiment.design import Component, Experiment
from experiment.results import completed, load_runs

PACKAGE_ROOT = Path(experiment_package.__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Cfg:
    LR: float = 1e-3
    NAME: str = "dqn"


def process(configs, seeds):
    return [{"reward": np.arange(2.0) + seed} for seed in seeds]


def refuse(configs, seeds):
    raise AssertionError("nothing should have been computed")


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    return Experiment(
        name="toy",
        results_dir=tmp_path,
        components=[
            Component(
                name="a", config=Cfg(), sweep={"LR": [1.0, 2.0]},
                seeds=[0, 1, 2], shard_size=2,
            ),
            Component(name="b", config=Cfg(NAME="ddqn"), seeds=[0, 1], shard_size=1),
        ],
    )


@pytest.fixture
def single_component(tmp_path) -> Experiment:
    return Experiment(
        name="solo",
        results_dir=tmp_path,
        components=[Component(name="only", config=Cfg(), seeds=[0, 1])],
    )


# --- assignments ------------------------------------------------------------


def test_assignments_split_on_the_first_equals():
    assert parse_overrides(["A.B=x=y"]) == {"A.B": "x=y"}


def test_a_repeated_path_keeps_the_last_value():
    assert parse_overrides(["A=1", "A=2"]) == {"A": "2"}


@pytest.mark.parametrize("bad", ["nope", "=5"])
def test_a_malformed_assignment_is_rejected(bad):
    with pytest.raises(SystemExit, match="PATH=VALUE"):
        parse_overrides([bad])


# --- dispatch ---------------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["bogus"]])
def test_a_mode_is_required(experiment, argv):
    with pytest.raises(SystemExit, match="mode is required"):
        run(experiment, refuse, argv)


# --- single -----------------------------------------------------------------


def test_single_stores_one_run(experiment):
    run(experiment, process, ["single", "--component", "b", "--seed", "1"])
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 0, "b": 1}


def test_single_uses_the_base_config_not_the_sweep(experiment):
    run(experiment, process, ["single", "--component", "a", "--seed", "0"])
    assert load_runs(experiment, "a")["LR"].to_list() == [1e-3]


def test_single_skips_a_run_it_already_has(experiment, capsys):
    run(experiment, process, ["single", "--component", "b", "--seed", "0"])
    capsys.readouterr()
    run(experiment, refuse, ["single", "--component", "b", "--seed", "0"])
    assert "already stored" in capsys.readouterr().out


def test_single_applies_an_override(experiment):
    run(experiment, process, ["single", "--component", "b", "--set", "LR=0.5"])
    assert load_runs(experiment, "b")["LR"].to_list() == [0.5]


def test_single_may_omit_the_component_when_there_is_one(single_component):
    run(single_component, process, ["single", "--seed", "1"])
    assert len(completed(single_component)["only"]) == 1


def test_single_needs_a_component_when_there_is_a_choice(experiment):
    with pytest.raises(SystemExit, match="needs --component"):
        run(experiment, refuse, ["single"])


def test_single_rejects_an_unknown_component(experiment):
    with pytest.raises(SystemExit, match="nope"):
        run(experiment, refuse, ["single", "--component", "nope"])


# --- sweep ------------------------------------------------------------------


def test_sweep_runs_everything(experiment):
    run(experiment, process, ["sweep"])
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 6, "b": 2}


def test_sweep_expands_the_sweep(experiment):
    run(experiment, process, ["sweep"])
    assert sorted(set(load_runs(experiment, "a")["LR"].to_list())) == [1.0, 2.0]


def test_a_finished_sweep_recomputes_nothing(experiment, capsys):
    run(experiment, process, ["sweep"])
    capsys.readouterr()
    run(experiment, refuse, ["sweep"])
    assert "nothing to run" in capsys.readouterr().out


def test_sweep_resumes_only_what_is_missing(experiment):
    run(experiment, process, ["sweep", "--component", "b"])
    run(experiment, process, ["sweep"])
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 6, "b": 2}


def test_sweep_can_be_restricted_to_a_component(experiment):
    run(experiment, process, ["sweep", "--component", "b"])
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 0, "b": 2}


def test_sweep_overrides_lose_to_the_sweep_but_beat_the_base(experiment):
    run(experiment, process, ["sweep", "--set", "LR=9", "--set", "NAME=x"])
    frame = load_runs(experiment, "a")
    assert sorted(set(frame["LR"].to_list())) == [1.0, 2.0]
    assert set(frame["NAME"].to_list()) == {"x"}


def test_sweep_leaves_no_parts_behind(experiment, tmp_path):
    run(experiment, process, ["sweep"])
    assert not (tmp_path / "toy.parts").exists()


# --- a sweep split into steps -----------------------------------------------


def plan_file(experiment, tmp_path, workers, *extra) -> list:
    """Do the plan step and read back what the workers would be given."""
    path = tmp_path / "plan.pickle"
    run(
        experiment, refuse,
        ["sweep", "--write-plan", str(path), "--num-workers", str(workers), *extra],
    )
    return pickle.loads(path.read_bytes())


def test_the_plan_step_computes_no_runs(experiment, tmp_path):
    plan_file(experiment, tmp_path, 3)
    assert completed(experiment) == {"a": set(), "b": set()}


def test_the_plan_step_gives_every_worker_a_share(experiment, tmp_path):
    """A cluster array is sized before the plan exists, so none may be missing."""
    assert len(plan_file(experiment, tmp_path, 16)) == 16


def test_the_plan_step_covers_every_run(experiment, tmp_path):
    shares = plan_file(experiment, tmp_path, 3)
    assert sum(len(shard) for share in shares for shard in share) == 8


def test_a_worker_runs_only_its_own_share(experiment, tmp_path):
    shares = plan_file(experiment, tmp_path, 3)
    path = tmp_path / "plan.pickle"
    run(experiment, process, ["sweep", "--plan", str(path), "--worker-index", "0"])
    expected = sum(len(shard) for shard in shares[0])
    assert sum(len(v) for v in completed(experiment).values()) == expected


def test_a_worker_without_an_index_is_refused(experiment, tmp_path):
    plan_file(experiment, tmp_path, 3)
    path = tmp_path / "plan.pickle"
    with pytest.raises(SystemExit, match="needs --worker-index"):
        run(experiment, refuse, ["sweep", "--plan", str(path)])


def test_the_steps_together_do_what_one_sweep_does(experiment, tmp_path):
    path = tmp_path / "plan.pickle"
    for index in range(len(plan_file(experiment, tmp_path, 3))):
        run(
            experiment, process,
            ["sweep", "--plan", str(path), "--worker-index", str(index)],
        )
    run(experiment, process, ["sweep", "--merge-only"])
    assert {k: len(v) for k, v in completed(experiment).items()} == {"a": 6, "b": 2}
    assert not (tmp_path / "toy.parts").exists()


def test_an_empty_share_writes_no_part(experiment, tmp_path):
    path = tmp_path / "plan.pickle"
    shares = plan_file(experiment, tmp_path, 16)
    empty = next(i for i, share in enumerate(shares) if not share)
    run(
        experiment, process,
        ["sweep", "--plan", str(path), "--worker-index", str(empty)],
    )
    assert not (tmp_path / "toy.parts" / f"part-{empty}.db").exists()


# --- status -----------------------------------------------------------------


def test_status_counts_an_untouched_experiment(experiment, capsys):
    run(experiment, refuse, ["status"])
    out = capsys.readouterr().out
    assert "8 run(s), 0 done, 8 pending" in out


def test_status_counts_what_has_been_run(experiment, capsys):
    run(experiment, process, ["sweep", "--component", "b"])
    capsys.readouterr()
    run(experiment, refuse, ["status"])
    assert "8 run(s), 2 done, 6 pending" in capsys.readouterr().out


def test_status_reports_the_useful_worker_count(experiment, capsys):
    run(experiment, refuse, ["status", "--shard-size", "1"])
    assert "--num-workers 8" in capsys.readouterr().out


def test_status_says_nothing_about_workers_when_finished(experiment, capsys):
    run(experiment, process, ["sweep"])
    capsys.readouterr()
    run(experiment, refuse, ["status"])
    assert "num-workers" not in capsys.readouterr().out


# --- workers across processes -----------------------------------------------


RUN_PY = '''\
import dataclasses
from pathlib import Path

import numpy as np

from experiment.commands import run
from experiment.design import Component, Experiment


@dataclasses.dataclass(frozen=True)
class Cfg:
    LR: float = 1e-3


def process(configs, seeds):
    return [{"reward": np.arange(2.0) + s} for s in seeds]


EXPERIMENT = Experiment(
    name="toy",
    results_dir=Path(__file__).parent / "results",
    components=[
        Component(name="a", config=Cfg(), sweep={"LR": [1.0, 2.0]},
                  seeds=list(range(6)), shard_size=2),
        Component(name="b", config=Cfg(), seeds=[0, 1], shard_size=1),
    ],
)

if __name__ == "__main__":
    run(EXPERIMENT, process)
'''


@pytest.fixture
def run_py(tmp_path) -> Path:
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(RUN_PY))
    return script


def invoke(script: Path, *argv: str) -> str:
    """Run a real experiment script the way a user would."""
    env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    done = subprocess.run(
        [sys.executable, str(script), *argv],
        capture_output=True, text=True, env=env, check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_a_sweep_across_worker_processes_runs_everything(run_py):
    invoke(run_py, "sweep", "--num-workers", "4")
    assert "14 run(s), 14 done, 0 pending" in invoke(run_py, "status")


def test_every_worker_takes_a_share(run_py):
    out = invoke(run_py, "sweep", "--num-workers", "4")
    assert sum(f"worker {index} stored" in out for index in range(4)) == 4


def test_workers_do_not_duplicate_each_others_runs(run_py):
    invoke(run_py, "sweep", "--num-workers", "4")
    stored = [
        line for line in invoke(run_py, "status").splitlines() if line.startswith("[")
    ]
    assert stored == [
        "[a] 12 run(s): 12 done, 0 pending in 0 shard(s)",
        "[b] 2 run(s): 2 done, 0 pending in 0 shard(s)",
    ]


def test_the_plan_file_is_cleaned_up(run_py, tmp_path):
    invoke(run_py, "sweep", "--num-workers", "4")
    assert not (tmp_path / "results" / "toy.parts").exists()


def test_a_pool_is_capped_at_the_number_of_shards(run_py):
    out = invoke(run_py, "sweep", "--component", "b", "--num-workers", "16")
    assert "across 2 worker(s)" in out
