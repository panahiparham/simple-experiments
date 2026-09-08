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
