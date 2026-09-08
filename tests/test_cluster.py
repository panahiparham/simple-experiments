from __future__ import annotations

from pathlib import Path

import pytest

from experiment import slurm


@pytest.fixture(autouse=True)
def _clear_root_cache():
    slurm._ROOT_CACHE.clear()
    yield
    slurm._ROOT_CACHE.clear()


def write_config(directory: Path, body: str) -> Path:
    path = directory / slurm.DEFAULT_CONFIG_PATH
    path.write_text(body)
    return path


def test_a_missing_config_is_refused_naming_the_path(tmp_path):
    missing = tmp_path / slurm.DEFAULT_CONFIG_PATH

    with pytest.raises(SystemExit, match=str(missing)):
        slurm.load_config(missing)


def test_the_project_name_falls_back_to_the_roots_basename(tmp_path):
    path = write_config(tmp_path, '[cluster]\nroot = "/scratch/alice/myproj"\n')

    assert slurm.load_config(path).project == "myproj"


def test_src_dirs_default_to_src(tmp_path):
    path = write_config(tmp_path, '[cluster]\nroot = "/scratch/alice/myproj"\n')

    assert slurm.load_config(path).src_dirs == ["src"]


def test_an_experiment_override_lands_on_top_of_the_defaults(tmp_path):
    path = write_config(
        tmp_path,
        '[slurm]\ntime = "0:10:00"\ncpus_per_task = 2\n'
        '\n[experiments.toy]\ntime = "8:00:00"\n',
    )

    resources = slurm.resources_for(slurm.load_config(path), "toy")

    assert resources == {"time": "8:00:00", "cpus_per_task": 2}


def test_an_experiment_with_no_overrides_gets_the_defaults(tmp_path):
    path = write_config(
        tmp_path,
        '[slurm]\ntime = "0:10:00"\n\n[experiments.toy]\ntime = "8:00:00"\n',
    )

    resources = slurm.resources_for(slurm.load_config(path), "other")

    assert resources == {"time": "0:10:00"}


def test_a_job_asking_for_a_gpu_runs_in_the_gpu_venv():
    assert slurm.venv_name({"gpus": 1}) == "gpu"


def test_a_job_asking_for_no_gpu_runs_in_the_cpu_venv():
    assert slurm.venv_name({"gpus": 0}) == "cpu"


def test_a_job_silent_about_gpus_runs_in_the_cpu_venv():
    assert slurm.venv_name({}) == "cpu"
