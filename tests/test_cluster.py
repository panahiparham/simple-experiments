from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiment import slurm
from experiment.design import Component, Experiment


@pytest.fixture(autouse=True)
def _clear_root_cache():
    slurm._ROOT_CACHE.clear()
    yield
    slurm._ROOT_CACHE.clear()


def write_config(directory: Path, body: str) -> Path:
    path = directory / slurm.DEFAULT_CONFIG_PATH
    path.write_text(body)
    return path


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    write_config(root, '[cluster]\nhost = "cedar"\naccount = "def-a"\n')
    (root / "uv.lock").write_text("version = 1\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    monkeypatch.chdir(root)
    return root


def commit_file(repo: Path, name: str, body: str) -> str:
    (repo / name).write_text(body)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", f"add {name}")
    return git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def cluster(repo, monkeypatch):
    def build(*, gpus: int = 0) -> Path:
        root = repo.parent / "cluster"
        write_config(
            repo,
            f'[cluster]\nhost = "cedar"\nroot = "{root}"\naccount = "def-a"\n'
            '\n[project]\nname = "toy"\nsrc_dirs = ["src"]\n'
            f'\n[slurm]\ntime = "0:10:00"\ncpus_per_task = 2\ngpus = {gpus}\n'
            '\n[venvs]\ncpu = []\ngpu = []\n',
        )
        (repo / "src").mkdir()
        (repo / "src" / "toy.py").write_text("")
        (repo / "run.py").write_text("")
        sha = commit_file(repo, "pyproject.toml", '[project]\nname = "toy"\n')
        git(repo, "init", "--bare", "-q", str(root / "toy.git"))
        git(repo, "remote", "add", "cluster-cedar", str(root / "toy.git"))
        for name in ("cpu", "gpu"):
            binary = root / "envs" / name / ".venv" / "bin" / "python"
            binary.parent.mkdir(parents=True)
            binary.write_text("")
            binary.chmod(0o755)
            (root / "envs" / name / "lock.sha256").write_text(
                slurm._lock_hash(sha, [], "")
            )
        monkeypatch.setenv("EXPERIMENT_LOCAL_MODE", "1")
        return root

    return build


def dispatch_single(repo: Path) -> None:
    slurm.dispatch(
        label="toy",
        run_py=repo / "run.py",
        mode="single",
        argv=["--seed", "0"],
        dry_run=True,
    )


def dispatch_sweep(repo: Path, workers: int) -> None:
    slurm.dispatch(
        label="toy",
        run_py=repo / "run.py",
        mode="sweep",
        argv=["--num-workers", str(workers)],
        dry_run=True,
    )


def job_names(out: str) -> list[str]:
    return [
        line.split("--job-name=", 1)[1].split(" ", 1)[0]
        for line in out.strip().splitlines()
    ]


def jobs_asking_for_a_gpu(out: str) -> list[str]:
    return [
        line.split("--job-name=", 1)[1].split(" ", 1)[0]
        for line in out.strip().splitlines()
        if "--gpus-per-node=" in line
    ]


def experiment_at(results_dir: Path) -> Experiment:
    return Experiment(
        name="toy",
        components=[Component(name="a", config=None, seeds=[0])],
        results_dir=results_dir,
    )


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


def test_a_spaced_worker_count_is_taken_out_of_the_argv():
    workers, rest = slurm._num_workers(["--num-workers", "4", "--component", "a"])

    assert (workers, rest) == (4, ["--component", "a"])


def test_an_equals_worker_count_is_taken_out_of_the_argv():
    workers, rest = slurm._num_workers(["--component", "a", "--num-workers=4"])

    assert (workers, rest) == (4, ["--component", "a"])


def test_a_sweep_without_a_worker_count_is_refused():
    with pytest.raises(SystemExit, match="needs --num-workers"):
        slurm._num_workers(["--component", "a"])


def test_a_worker_count_below_one_is_refused():
    with pytest.raises(SystemExit, match="at least 1"):
        slurm._num_workers(["--num-workers", "0"])


def test_a_single_worker_is_allowed():
    assert slurm._num_workers(["--num-workers", "1"]) == (1, [])


def test_the_snapshots_source_dirs_go_on_pythonpath():
    command = slurm._job_command(
        "/runs/r1", "run.py", "/envs/cpu/.venv", "single", [], ["src", "lib"]
    )

    assert "PYTHONPATH=/runs/r1/src:/runs/r1/lib " in command


def test_the_array_task_id_is_left_for_sbatch_to_expand():
    command = slurm._job_command(
        "/runs/r1",
        "run.py",
        "/envs/cpu/.venv",
        "sweep",
        ["--worker-index", "$SLURM_ARRAY_TASK_ID"],
        ["src"],
    )

    assert command.endswith("--worker-index $SLURM_ARRAY_TASK_ID")


def test_an_ordinary_argument_is_quoted():
    command = slurm._job_command(
        "/runs/r1",
        "run.py",
        "/envs/cpu/.venv",
        "single",
        ["--overrides", "A=1 B=2"],
        ["src"],
    )

    assert command.endswith("--overrides 'A=1 B=2'")


def test_an_mfa_refusal_says_to_open_the_connection_by_hand(tmp_path):
    cfg = slurm.load_config(write_config(tmp_path, '[cluster]\nhost = "cedar"\n'))

    with pytest.raises(SystemExit, match="ssh cedar true"):
        slurm._check_auth(cfg, "cedar: Permission denied (keyboard-interactive)")


def test_a_rejected_key_says_to_open_the_connection_by_hand(tmp_path):
    cfg = slurm.load_config(write_config(tmp_path, '[cluster]\nhost = "cedar"\n'))

    with pytest.raises(SystemExit, match="ssh cedar true"):
        slurm._check_auth(cfg, "cedar: Permission denied (publickey)")


def test_an_error_that_is_not_an_ssh_refusal_is_left_alone(tmp_path):
    cfg = slurm.load_config(write_config(tmp_path, '[cluster]\nhost = "cedar"\n'))

    assert slurm._check_auth(cfg, "rsync: link_stat failed: No such file") is None


def test_a_reported_field_is_read_from_its_key():
    assert slurm._field("VENV=/envs/cpu/.venv\n", "VENV") == "/envs/cpu/.venv"


def test_the_last_report_of_a_field_wins():
    assert slurm._field("RUNDIR=/runs/a\nRUNDIR=/runs/b\n", "RUNDIR") == "/runs/b"


def test_a_field_the_script_did_not_report_reads_as_empty():
    assert slurm._field("VENV=/envs/cpu/.venv\n", "RUNDIR") == ""


def test_the_repo_root_is_the_directory_holding_the_cluster_config(repo, monkeypatch):
    nested = repo / "experiments" / "toy"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert slurm.repo_root() == repo


def test_a_tree_with_no_cluster_config_has_no_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="could not locate the repo root"):
        slurm.repo_root()


def test_a_clean_tree_dispatches_at_head(repo):
    head = git(repo, "rev-parse", "HEAD").stdout.strip()

    assert slurm._require_clean_tree() == head


def test_an_uncommitted_file_blocks_a_dispatch_naming_it(repo):
    (repo / "scratch.py").write_text("x = 1\n")

    with pytest.raises(SystemExit, match=r"scratch\.py"):
        slurm._require_clean_tree()


def test_different_extras_hash_differently(repo):
    head = git(repo, "rev-parse", "HEAD").stdout.strip()

    assert slurm._lock_hash(head, ["cuda"]) != slurm._lock_hash(head, [])


def test_a_code_only_commit_keeps_the_venv_identity(repo):
    before = slurm._lock_hash(commit_file(repo, "a.py", "x = 1\n"), [])

    after = slurm._lock_hash(commit_file(repo, "b.py", "y = 2\n"), [])

    assert after == before


def test_a_changed_post_sync_script_rebuilds_the_venv(repo):
    first = commit_file(repo, "hook.sh", "echo one\n")
    before = slurm._lock_hash(first, [], "hook.sh")

    after = slurm._lock_hash(commit_file(repo, "hook.sh", "echo two\n"), [], "hook.sh")

    assert after != before


def test_a_post_sync_missing_from_the_commit_is_refused(repo):
    head = git(repo, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(SystemExit, match="post_sync"):
        slurm._lock_hash(head, [], "missing.sh")


def test_a_dry_run_reports_the_sbatch_command_it_would_run(cluster, repo, capsys):
    cluster()

    dispatch_single(repo)

    assert "sbatch --parsable --account=def-a --time=0:10:00 --cpus-per-task=2" in (
        capsys.readouterr().out
    )


def test_a_dry_run_records_no_dispatch(cluster, repo):
    cluster()

    dispatch_single(repo)

    assert not (repo / ".cluster").exists()


def test_a_dry_run_leaves_no_run_directory_behind(cluster, repo):
    root = cluster()

    dispatch_single(repo)

    assert list((root / "runs").iterdir()) == []


def test_a_sweep_chains_a_plan_an_array_and_a_merge(cluster, repo, capsys):
    cluster()

    dispatch_sweep(repo, 4)

    assert job_names(capsys.readouterr().out) == ["plan", "sweep", "merge"]


def test_a_sweep_sizes_its_array_to_the_worker_count(cluster, repo, capsys):
    cluster()

    dispatch_sweep(repo, 4)

    assert "--array=0-3" in capsys.readouterr().out


def test_each_sweep_job_waits_for_the_one_before_it(cluster, repo, capsys):
    cluster()

    dispatch_sweep(repo, 4)

    out = capsys.readouterr().out
    assert "afterok:<plan-id>" in out, "the array does not wait for the plan"
    assert "afterok:<array-id>" in out, "the merge does not wait for the array"


def test_only_the_array_job_asks_for_a_gpu(cluster, repo, capsys):
    cluster(gpus=1)

    dispatch_sweep(repo, 2)

    assert jobs_asking_for_a_gpu(capsys.readouterr().out) == ["sweep"]


def test_a_gpu_job_runs_the_gpu_venvs_python(cluster, repo, capsys):
    root = cluster(gpus=1)

    dispatch_single(repo)

    assert f"{root}/envs/gpu/.venv/bin/python" in capsys.readouterr().out


def test_fetch_brings_the_clusters_database_home(cluster, tmp_path):
    remote = cluster() / "results" / "toy"
    remote.mkdir(parents=True)
    (remote / "toy.db").write_text("merged")
    local = tmp_path / "local"

    slurm.fetch(experiment_at(local))

    assert (local / "toy.parts" / "part-cluster.db").read_text() == "merged"


def test_fetch_brings_a_sweeps_leftover_parts_home(cluster, tmp_path):
    remote = cluster() / "results" / "toy" / "toy.parts"
    remote.mkdir(parents=True)
    (remote / "part-0.db").write_text("worker zero")
    local = tmp_path / "local"

    slurm.fetch(experiment_at(local))

    assert (local / "toy.parts" / "part-cluster-0.db").read_text() == "worker zero"


@pytest.fixture
def rsync_calls(monkeypatch) -> list[list[str]]:
    recorded: list[list[str]] = []
    real = slurm._run

    def record(argv, **kwargs):
        if argv[0] == "rsync":
            recorded.append(argv)
        return real(argv, **kwargs)

    monkeypatch.setattr(slurm, "_run", record)
    return recorded


def test_fetch_compresses_the_transfer(cluster, tmp_path, rsync_calls):
    remote = cluster() / "results" / "toy"
    remote.mkdir(parents=True)
    (remote / "toy.db").write_text("merged")

    slurm.fetch(experiment_at(tmp_path / "local"))

    [argv] = rsync_calls
    assert "z" in argv[1], f"rsync ran as {argv}"


def test_fetching_logs_compresses_the_transfer(cluster, repo, rsync_calls):
    rundir = cluster() / "runs" / "toy_x"
    (rundir / "logs").mkdir(parents=True)
    (rundir / "logs" / "toy_0.out").write_text("step 0")
    state = repo / ".cluster" / "toy.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"runid": "toy_x", "rundir": str(rundir)}))

    slurm.logs(label="toy")

    [argv] = rsync_calls
    assert "z" in argv[1], f"rsync ran as {argv}"


def test_fetch_before_anything_was_dispatched_is_refused(cluster, tmp_path):
    cluster()

    with pytest.raises(SystemExit, match="nothing on the cluster"):
        slurm.fetch(experiment_at(tmp_path / "local"))


def test_fetch_from_an_empty_results_dir_is_refused(cluster, tmp_path):
    (cluster() / "results" / "toy").mkdir(parents=True)

    with pytest.raises(SystemExit, match="no results at"):
        slurm.fetch(experiment_at(tmp_path / "local"))


@pytest.mark.parametrize(
    "call",
    [slurm.status, slurm.logs, slurm.is_queued],
    ids=["status", "logs", "is_queued"],
)
def test_a_command_needing_a_dispatch_is_refused_before_one(cluster, call):
    cluster()

    with pytest.raises(SystemExit, match="nothing dispatched yet"):
        call(label="toy")


def test_a_dispatch_that_recorded_no_job_ids_is_refused(cluster, repo):
    cluster()
    state = repo / ".cluster" / "toy.json"
    state.parent.mkdir()
    state.write_text('{"runid": "toy_x", "jobs": {}}')

    with pytest.raises(SystemExit, match="no job ids recorded"):
        slurm.is_queued(label="toy")
