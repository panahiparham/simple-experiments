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


def test_an_explicit_mem_wins_over_mem_per_cpu():
    assert slurm.resource_flags({"mem": "16G", "mem_per_cpu": "4G"}) == ["--mem=16G"]


def test_mem_per_cpu_applies_when_no_total_mem_is_set():
    assert slurm.resource_flags({"mem_per_cpu": "4G"}) == ["--mem-per-cpu=4G"]


def test_a_gpu_request_becomes_a_gpus_per_node_flag():
    assert slurm.resource_flags({"gpus": 2}) == ["--gpus-per-node=2"]


def test_a_zero_gpu_request_asks_for_no_gpu():
    assert slurm.resource_flags({"gpus": 0}) == []


def test_an_empty_resource_table_produces_no_flags():
    assert slurm.resource_flags({}) == []


def test_a_full_resource_table_produces_flags_in_command_line_order():
    flags = slurm.resource_flags(
        {"time": "1:00:00", "cpus_per_task": 4, "mem": "16G", "gpus": 1}
    )

    assert flags == [
        "--time=1:00:00",
        "--cpus-per-task=4",
        "--mem=16G",
        "--gpus-per-node=1",
    ]


def test_a_job_with_no_account_anywhere_is_refused(tmp_path):
    cfg = slurm.load_config(write_config(tmp_path, '[cluster]\nroot = "/scratch"\n'))

    with pytest.raises(SystemExit, match="no Slurm account set in"):
        slurm._sbatch_argv(cfg, {}, wrap="true")


def test_a_resource_account_overrides_the_cluster_account(tmp_path):
    cfg = slurm.load_config(
        write_config(tmp_path, '[cluster]\naccount = "def-default"\n')
    )

    argv = slurm._sbatch_argv(cfg, {"account": "def-other"}, wrap="true")

    assert "--account=def-other" in argv


def test_the_wrapped_command_comes_after_every_flag(tmp_path):
    cfg = slurm.load_config(
        write_config(tmp_path, '[cluster]\naccount = "def-default"\n')
    )

    argv = slurm._sbatch_argv(cfg, {"time": "1:00:00"}, "--job-name=x", wrap="true")

    assert argv == [
        "sbatch",
        "--parsable",
        "--account=def-default",
        "--time=1:00:00",
        "--job-name=x",
        "--wrap",
        "true",
    ]
