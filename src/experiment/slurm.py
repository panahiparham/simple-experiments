"""Run this repo's experiments on a Compute Canada (Alliance) SLURM cluster, driven
entirely from a laptop.

Nothing is maintained on the cluster by hand. A *bare* repo there is the push target;
each dispatch snapshots exactly one commit into its own run dir (``git archive``, so no
index and no working checkout), and a queued job's code can never be swapped underneath
it. Only committed work is ever submitted, so a run maps to one sha.

Two shared venvs, ``cpu`` and ``gpu``, serve every commit. They are *deps-only*
(``uv sync --no-install-workspace``): installing the project - or this harness,
a workspace member of it - would pin a venv to whichever snapshot built it first,
so jobs instead run ``<venv>/bin/python`` with the snapshot's own source dirs
(``[project] src_dirs``) on ``PYTHONPATH`` and always execute their own source. Each
venv records the ``uv.lock`` hash it was built from, and a dispatch re-syncs it first
if the lock moved. That sync has to happen on the login node: compute nodes have no
internet, and a lock may name dependencies that come from a git URL.

Whatever ``uv sync`` cannot install is the project's own business: ``[project]
post_sync`` names a script in the snapshot, run after the sync with ``EXPERIMENT_VENV``
and ``EXPERIMENT_EXTRAS`` in its environment. Its contents are part of a venv's
identity, so changing it rebuilds the venvs it applies to.

An experiment's results accumulate in one shared directory on the cluster
(``results/<label>/``) that every snapshot of that experiment symlinks to. That is what
lets :func:`sync` pull an experiment down with a single rsync, and it lets the harness's
``run_id`` dedup resume across commits.

Layout under the configured root::

    <project>.git/                  bare repo (the push target)
    envs/{cpu,gpu}/.venv            + lock.sha256
    results/<label>/                shared across every commit
    runs/<runid>/                   snapshot, MANIFEST, logs/

``EXPERIMENT_LOCAL_MODE=1`` runs every "remote" command in a local shell and drops the
``host:``
prefix from rsync and git paths, so the whole flow can be exercised against a directory
on this machine. That is how ``tests/test_cluster.py`` works without a cluster.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ClusterConfig",
    "DEFAULT_CONFIG_PATH",
    "repo_root",
    "load_config",
    "resources_for",
    "resource_flags",
    "venv_name",
    "dispatch",
    "fetch",
    "wipe",
    "is_queued",
    "status",
    "logs",
    "setup",
]

DEFAULT_CONFIG_PATH = "cluster.toml"

# A configured root may contain a literal $HOME for the cluster's shell to expand.
# Expanding it costs an ssh round trip, so cache per (host, root) for this process.
_ROOT_CACHE: dict[tuple[str, str], str] = {}


@dataclasses.dataclass(frozen=True)
class ClusterConfig:
    """``cluster.toml``, parsed."""

    host: str
    root: str
    account: str
    project: str
    src_dirs: list[str]
    post_sync: str
    venvs: dict[str, list[str]]
    slurm: dict[str, Any]
    experiments: dict[str, dict]
    path: Path


def repo_root() -> Path:
    """The repo root, found by walking up from cwd to the directory holding
    cluster.toml.

    Walks up from the current working directory rather than this module's own
    file, since the harness is typically installed as a dependency and does
    not live inside the project whose root it needs to find.
    """
    here = Path.cwd().resolve()
    for path in [here, *here.parents]:
        if (path / DEFAULT_CONFIG_PATH).is_file():
            return path
    raise SystemExit(
        f"could not locate the repo root (no {DEFAULT_CONFIG_PATH} above {here})"
    )


def load_config(path: str | Path | None = None) -> ClusterConfig:
    """Read the cluster/slurm configuration.

    Args:
        path: The ``cluster.toml`` to read, defaulting to the repo root's.

    Returns:
        The parsed configuration.

    Raises:
        SystemExit: If no config exists at ``path``.
    """
    path = repo_root() / DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not path.is_file():
        raise SystemExit(
            f"no cluster config at {path} (see cluster.toml in the repo root)"
        )
    data = tomllib.loads(path.read_text())
    cluster = data.get("cluster", {})
    root = cluster.get("root", "")
    return ClusterConfig(
        host=cluster.get("host", ""),
        root=root,
        account=cluster.get("account", ""),
        project=data.get("project", {}).get("name") or Path(root).name,
        src_dirs=data.get("project", {}).get("src_dirs", ["src"]),
        post_sync=data.get("project", {}).get("post_sync", ""),
        venvs=data.get("venvs", {}),
        slurm=data.get("slurm", {}),
        experiments=data.get("experiments", {}),
        path=path,
    )


def resources_for(cfg: ClusterConfig, label: str) -> dict:
    """Merge an experiment's resource overrides onto the defaults.

    Args:
        cfg: The parsed cluster configuration.
        label: The experiment name, keying ``[experiments.<label>]``.

    Returns:
        The ``[slurm]`` defaults with that experiment's overrides on top.
    """
    return {**cfg.slurm, **cfg.experiments.get(label, {})}


def venv_name(resources: dict) -> str:
    """Pick the shared venv a job runs in.

    Args:
        resources: The job's merged resource table.

    Returns:
        The venv name, which is what makes GPU and CPU jobs share environments.
    """
    return "gpu" if int(resources.get("gpus", 0)) > 0 else "cpu"


# --- transport ---------------------------------------------------------------------


def _local_mode() -> bool:
    return os.environ.get("EXPERIMENT_LOCAL_MODE") == "1"


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def _check_auth(cfg: ClusterConfig, stderr: str) -> None:
    """Turn an ssh auth refusal into something actionable.

    Alliance clusters require MFA over keyboard-interactive, which a subprocess cannot
    answer. Left alone, an expired session surfaces as a failure of whichever remote
    command happened to run first, which reads as a bug in that command rather than as
    "log in again", so name the real problem. Raised regardless of ``check``: this is
    never a command legitimately returning non-zero.
    """
    if "keyboard-interactive" not in stderr and "Permission denied" not in stderr:
        return
    raise SystemExit(
        f"ssh to {cfg.host} was refused:\n"
        f"  {stderr.strip().splitlines()[-1]}\n"
        f"MFA needs a terminal, so open the connection once yourself:\n"
        f"    ssh {cfg.host} true\n"
        f"ControlMaster keeps it alive for the commands that follow. Then retry."
    )


def _ssh(cfg: ClusterConfig, command: str, *, check: bool = True) -> str:
    """Run a shell command on the login node and return its stdout."""
    argv = ["bash", "-c", command] if _local_mode() else ["ssh", cfg.host, command]
    proc = _run(argv)
    _check_auth(cfg, proc.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(f"remote command failed: {command}\n{proc.stderr.strip()}")
    return proc.stdout


def _ssh_script(cfg: ClusterConfig, script: Path, *args: str) -> str:
    """Pipe one of the package's ``remote/*.sh`` to the login node and run it.

    The scripts are never stored on the cluster, so there is no second copy to keep in
    step with this repo. Their stderr is passed through for progress; only their
    ``KEY=value`` stdout is captured.
    """
    quoted = " ".join(shlex.quote(a) for a in args)
    body = script.read_text()
    argv = ["bash", "-s", "--", *args] if _local_mode() else [
        "ssh", cfg.host, f"bash -s -- {quoted}"
    ]
    proc = subprocess.run(argv, input=body, capture_output=True, text=True)
    _check_auth(cfg, proc.stderr)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"{script.name} failed on {cfg.host}")
    return proc.stdout


def _remote_path(cfg: ClusterConfig, path: str) -> str:
    """An rsync/git style location for something on the cluster."""
    return path if _local_mode() else f"{cfg.host}:{path}"


def _field(output: str, key: str) -> str:
    """The last ``KEY=value`` line of a remote script's stdout."""
    for line in reversed(output.splitlines()):
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return ""


def _remote_root(cfg: ClusterConfig) -> str:
    key = (cfg.host, cfg.root)
    if key not in _ROOT_CACHE:
        _ROOT_CACHE[key] = _ssh(cfg, f'printf "%s" "{cfg.root}"').strip()
    return _ROOT_CACHE[key]


def _remote_dir() -> Path:
    """The package's own ``remote/*.sh``, piped to the login node on demand."""
    return Path(__file__).resolve().parent / "remote"


# --- git ---------------------------------------------------------------------------


def _require_clean_tree() -> str:
    """The HEAD sha, or exit if anything is uncommitted.

    A snapshot comes from a commit, so uncommitted work would silently not be part of
    the run. Failing is the only honest option.
    """
    proc = _run(["git", "status", "--porcelain"], cwd=repo_root())
    if proc.stdout.strip():
        raise SystemExit(
            "working tree is dirty - commit or stash before dispatching:\n"
            + proc.stdout.rstrip()
        )
    proc = _run(["git", "rev-parse", "HEAD"], cwd=repo_root(), check=True)
    return proc.stdout.strip()


def _remote_name(cfg: ClusterConfig) -> str:
    return f"cluster-{cfg.host}"


def _set_remote(cfg: ClusterConfig, bare: str) -> None:
    """Point the local git remote at the cluster's bare repo."""
    name, url = _remote_name(cfg), _remote_path(cfg, bare)
    exists = _run(["git", "remote", "get-url", name], cwd=repo_root()).returncode == 0
    _run(["git", "remote", "set-url" if exists else "add", name, url],
         cwd=repo_root(), check=True)


def _push(cfg: ClusterConfig, sha: str, ref: str) -> None:
    """Push a commit to the bare repo under its own ref.

    A per-run ref pins the commit, so a later force-push of a branch can never make an
    existing snapshot unreachable.
    """
    name = _remote_name(cfg)
    proc = _run(["git", "push", "--quiet", name, f"+{sha}:{ref}"], cwd=repo_root())
    _check_auth(cfg, proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(
            f"push to {name} failed - run setup_cluster.py if the bare repo is gone\n"
            + proc.stderr.strip()
        )


def _lock_hash(sha: str, extras: list[str], post_sync: str = "") -> str:
    """Identity of a venv: the locked dependency set, the extras selected for it,
    and the post-sync script that runs on top of them.

    Nothing else about the source, so a code-only commit reuses the venv
    untouched - but a changed post-sync script rebuilds it, since what that
    script installs is part of what the venv holds.
    """
    lock = _run(["git", "show", f"{sha}:uv.lock"], cwd=repo_root(), check=True)
    payload = lock.stdout.encode() + b"\0" + "\0".join(sorted(extras)).encode()
    if post_sync:
        hook = _run(["git", "show", f"{sha}:{post_sync}"], cwd=repo_root())
        if hook.returncode != 0:
            raise SystemExit(
                f"[project] post_sync is {post_sync!r}, which the commit being "
                "dispatched does not carry"
            )
        payload += b"\0" + hook.stdout.encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# --- local state -------------------------------------------------------------------


def _state_path(label: str) -> Path:
    return repo_root() / ".cluster" / f"{label}.json"


def _read_state(label: str) -> dict:
    path = _state_path(label)
    return json.loads(path.read_text()) if path.is_file() else {}


def _write_state(label: str, state: dict) -> None:
    path = _state_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def _require_state(label: str) -> dict:
    state = _read_state(label)
    if not state:
        raise SystemExit(f"[{label}] nothing dispatched yet (no {_state_path(label)})")
    return state


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- sbatch ------------------------------------------------------------------------

# Written unquoted into the --wrap body so sbatch, not any shell along the way, expands
# it per array task.
_TASK_ID = "$SLURM_ARRAY_TASK_ID"


def _job_command(
    rundir: str, run_py_rel: str, venv: str, mode: str, argv: list[str],
    src_dirs: list[str],
) -> str:
    """The shell command one sbatch task runs.

    PYTHONPATH is what makes the snapshot's own source dirs win over the shared
    venv, which deliberately has no copy of the project installed.
    """
    parts = [f"{shlex.quote(venv)}/bin/python", shlex.quote(run_py_rel), mode]
    parts += [a if a == _TASK_ID else shlex.quote(a) for a in argv]
    pythonpath = ":".join(f"{rundir}/{d}" for d in src_dirs)
    return (f"cd {shlex.quote(rundir)} && "
            f"PYTHONPATH={shlex.quote(pythonpath)} " + " ".join(parts))


def resource_flags(resources: dict) -> list[str]:
    """Build the sbatch flags a job actually gets.

    Public so ``plan --slurm`` reports what would be requested rather than the
    merged table, which can mislead: an experiment overriding ``mem`` leaves the
    default ``mem_per_cpu`` sitting in the table unused.

    Args:
        resources: The job's merged resource table.

    Returns:
        The sbatch flags, in command-line order.
    """
    flags = []
    if "time" in resources:
        flags.append(f"--time={resources['time']}")
    if "cpus_per_task" in resources:
        flags.append(f"--cpus-per-task={resources['cpus_per_task']}")
    # --mem and --mem-per-cpu are mutually exclusive, so an explicit --mem wins.
    if "mem" in resources:
        flags.append(f"--mem={resources['mem']}")
    elif "mem_per_cpu" in resources:
        flags.append(f"--mem-per-cpu={resources['mem_per_cpu']}")
    if int(resources.get("gpus", 0)) > 0:
        flags.append(f"--gpus-per-node={resources['gpus']}")
    return flags


def _sbatch_argv(
    cfg: ClusterConfig, resources: dict, *extra: str, wrap: str
) -> list[str]:
    """One sbatch command line. ``--parsable`` prints the job id and nothing else."""
    account = resources.get("account", cfg.account)
    if not account:
        raise SystemExit(
            f"no Slurm account set in {cfg.path} - setup_cluster.py reports the "
            "candidates it finds on the cluster"
        )
    argv = [
        "sbatch", "--parsable", f"--account={account}",
        *resource_flags(resources),
    ]
    return [*argv, *extra, "--wrap", wrap]


def _submit(cfg: ClusterConfig, argv: list[str], *, dry_run: bool) -> str:
    """Submit (or on a dry run just show) one sbatch command; returns the job id."""
    command = shlex.join(argv)
    if dry_run:
        print(command)
        return ""
    job_id = _ssh(cfg, command).strip()
    if not job_id:
        raise SystemExit(f"sbatch returned no job id for: {command}")
    return job_id


def _num_workers(argv: list[str]) -> tuple[int, list[str]]:
    """Pull ``--num-workers N`` out of a sweep's argv, leaving the rest for the job."""
    rest: list[str] = []
    workers: int | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--num-workers" and i + 1 < len(argv):
            workers = int(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--num-workers="):
            workers = int(argv[i].split("=", 1)[1])
            i += 1
            continue
        rest.append(argv[i])
        i += 1
    if workers is None:
        raise SystemExit(
            "sweep needs --num-workers N (run.py status reports the total)"
        )
    if workers < 1:
        raise SystemExit(f"--num-workers must be at least 1, got {workers}")
    return workers, rest


# --- public API --------------------------------------------------------------------


def dispatch(
    *,
    label: str,
    run_py: Path,
    mode: str,
    argv: list[str],
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> None:
    """Run one of this experiment's modes on the cluster instead of here.

    ``sweep`` becomes three chained jobs - one deciding the plan, an array
    working through it, and one merging what the array wrote - and ``single``
    becomes a single job.

    Args:
        label: The experiment name.
        run_py: The experiment's ``run.py``.
        mode: Either ``sweep`` or ``single``.
        argv: Arguments passed through to the remote ``run.py``.
        config_path: The ``cluster.toml`` to read, defaulting to the repo root's.
        dry_run: Build and report the jobs without submitting them.

    Raises:
        SystemExit: If the working tree is dirty, or ``sweep`` is missing
            ``--num-workers``.
    """
    cfg = load_config(config_path)
    if mode not in ("sweep", "single"):
        raise SystemExit(f"[{label}] mode {mode!r} cannot be dispatched to the cluster")

    resources = resources_for(cfg, label)
    venv = venv_name(resources)
    sha = _require_clean_tree()
    runid = f"{label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{sha[:7]}"
    repo = repo_root()
    run_py_rel = run_py.resolve().relative_to(repo).as_posix()
    exp_reldir = run_py.resolve().parent.relative_to(repo).as_posix()

    print(f"[{label}] {runid} (sha {sha[:7]}, {venv} venv)", file=sys.stderr)
    root = _remote_root(cfg)
    _push(cfg, sha, f"refs/gsp/runs/{runid}")

    rundir = _field(
        _ssh_script(
            cfg, _remote_dir() / "prepare.sh", root, runid, sha, label,
            exp_reldir, cfg.project
        ),
        "RUNDIR",
    )
    if not rundir:
        raise SystemExit("prepare.sh did not report a run dir")

    extras = cfg.venvs.get(venv, [])
    venv_path = _field(
        _ssh_script(
            cfg, _remote_dir() / "build_env.sh", root, venv,
            _lock_hash(sha, extras, cfg.post_sync), rundir, cfg.post_sync, *extras
        ),
        "VENV",
    )
    if not venv_path:
        raise SystemExit("build_env.sh did not report a venv")

    if mode == "sweep":
        workers, rest = _num_workers(argv)
        plan_file = f"{rundir}/plan.pickle"
        # Deciding a plan and merging sqlite parts never touch a GPU, so neither
        # job may inherit the array's. Otherwise each queues for scarce hardware
        # it cannot use, with the array waiting behind the first of them.
        cheap = {**resources, "gpus": 0}
        # One job decides the plan, so every task works from the same one. Tasks
        # deciding it for themselves would each see whatever happened to be
        # stored when they started, and could disagree about who runs what.
        plan = _submit(cfg, _sbatch_argv(
            cfg, cheap,
            "--job-name=plan",
            f"--output={rundir}/logs/plan-%j.out",
            wrap=_job_command(rundir, run_py_rel, venv_path, "sweep",
                              ["--write-plan", plan_file,
                               "--num-workers", str(workers), *rest],
                              cfg.src_dirs),
        ), dry_run=dry_run)
        # The filters and overrides in `rest` are baked into the plan already, so
        # a task needs nothing beyond its own index.
        array = _submit(cfg, _sbatch_argv(
            cfg, resources,
            f"--dependency=afterok:{plan or '<plan-id>'}",
            f"--array=0-{workers - 1}",
            "--job-name=sweep",
            f"--output={rundir}/logs/sweep-%A_%a.out",
            wrap=_job_command(rundir, run_py_rel, venv_path, "sweep",
                              ["--plan", plan_file, "--worker-index", _TASK_ID],
                              cfg.src_dirs),
        ), dry_run=dry_run)
        merge = _submit(cfg, _sbatch_argv(
            cfg, cheap,
            f"--dependency=afterok:{array or '<array-id>'}",
            "--job-name=merge",
            f"--output={rundir}/logs/merge-%j.out",
            wrap=_job_command(rundir, run_py_rel, venv_path, "sweep",
                              ["--merge-only"], cfg.src_dirs),
        ), dry_run=dry_run)
        jobs = {"plan": plan, "array": array, "merge": merge}
    else:
        jobs = {"single": _submit(cfg, _sbatch_argv(
            cfg, resources,
            "--job-name=single",
            f"--output={rundir}/logs/single-%j.out",
            wrap=_job_command(rundir, run_py_rel, venv_path, "single", argv,
                              cfg.src_dirs),
        ), dry_run=dry_run)}

    if dry_run:
        _ssh(cfg, f"rm -rf {shlex.quote(rundir)}")
        print(f"dry run: nothing submitted, {rundir} removed "
              f"(the {venv} venv was kept)",
              file=sys.stderr)
        return

    _write_state(label, {
        "runid": runid, "sha": sha, "label": label, "rundir": rundir,
        "venv": venv, "venv_path": venv_path, "jobs": jobs, "submitted": _utc(),
    })
    listed = ", ".join(f"{k} {v}" for k, v in jobs.items())
    print(f"[{label}] submitted {listed}\n"
          f"  run.py status   watch it\n"
          f"  run.py sync     bring the results home", file=sys.stderr)


def fetch(experiment, *, config_path: str | Path | None = None) -> int:
    """Copy an experiment's cluster results into its local results directory.

    The cluster's database lands as one more part beside the local one, and any
    parts a sweep left behind come with it. Merging them is the caller's to do,
    and it is the ordinary merge: ids decide what is already there, so local and
    cluster results combine, nothing is overwritten, and fetching twice changes
    nothing.

    No run id is involved: every commit of an experiment writes into one shared
    results dir on the cluster, so this always covers the whole experiment.

    Args:
        experiment: The experiment to fetch.
        config_path: The ``cluster.toml`` to read, defaulting to the repo root's.

    Returns:
        The number of databases brought home.

    Raises:
        SystemExit: If nothing has been dispatched yet, or rsync fails.
    """
    cfg = load_config(config_path)
    remote = f"{_remote_root(cfg)}/results/{experiment.name}"
    if not _ssh(cfg, f"test -d {shlex.quote(remote)} && echo yes",
                check=False).strip():
        raise SystemExit(f"[{experiment.name}] nothing on the cluster at {remote}")

    # Parts still on the cluster mean a sweep is mid-flight, so its database is
    # missing runs. Worth saying, but whatever is complete is still safe to take.
    leftover = _ssh(
        cfg,
        f"ls -d {shlex.quote(remote)}/*.parts 2>/dev/null || true",
        check=False,
    )
    if leftover.strip():
        print(f"[{experiment.name}] warning: a sweep is still in flight, these "
              f"are incomplete:\n" + leftover.rstrip(), file=sys.stderr)

    parts = experiment.results_dir / f"{experiment.name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    found = 0
    with tempfile.TemporaryDirectory() as staging:
        proc = _run(["rsync", "-az", _remote_path(cfg, remote + "/"), staging + "/"])
        _check_auth(cfg, proc.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"rsync from {remote} failed\n{proc.stderr.strip()}")

        staged = Path(staging)
        merged = staged / f"{experiment.name}.db"
        if merged.exists():
            shutil.move(str(merged), parts / "part-cluster.db")
            print(f"  {merged.name} -> {parts.name}/part-cluster.db", file=sys.stderr)
            found += 1
        # A sweep still in flight, or one whose merge job failed, leaves parts on
        # the cluster. They hold real results, so bring them too.
        for index, part in enumerate(sorted(staged.glob("*.parts/part-*.db"))):
            shutil.move(str(part), parts / f"part-cluster-{index}.db")
            print(f"  {part.name} -> {parts.name}/part-cluster-{index}.db",
                  file=sys.stderr)
            found += 1

    if not found:
        raise SystemExit(f"[{experiment.name}] no results at {remote} yet")
    return found


def wipe(*, label: str, config_path: str | Path | None = None) -> None:
    """Remove an experiment's shared results dir on the cluster.

    Every commit's snapshot symlinks its ``results/`` to this one shared
    directory, so deleting it clears every stored run - the next dispatch
    recomputes its seeds from scratch instead of the harness's usual
    dedup-by-``run_id`` skipping everything that already matches.

    Args:
        label: The experiment name.
        config_path: The ``cluster.toml`` to read, defaulting to the repo
            root's.
    """
    cfg = load_config(config_path)
    remote_results = f"{_remote_root(cfg)}/results/{label}"
    if not _ssh(cfg, f"test -d {shlex.quote(remote_results)} && echo yes",
                check=False).strip():
        print(f"[{label}] nothing to wipe at {remote_results}", file=sys.stderr)
        return
    _ssh(cfg, f"rm -rf {shlex.quote(remote_results)}")
    print(f"[{label}] removed {remote_results}", file=sys.stderr)


def is_queued(*, label: str, config_path: str | Path | None = None) -> bool:
    """Whether any of a label's dispatched jobs are still queued or running.

    Meant for a best-effort caller (e.g. a cron job) that wants to know when
    a dispatch has finished without parsing job states: any
    ``SystemExit`` this raises (nothing dispatched yet, or Vulcan unreachable)
    means "can't tell yet", not a real error.

    Args:
        label: The experiment name.
        config_path: The ``cluster.toml`` to read, defaulting to the repo
            root's.

    Returns:
        ``True`` if any recorded job is still in ``squeue``, ``False`` once
        none are.

    Raises:
        SystemExit: If nothing has been dispatched, or the login node can't
            be reached.
    """
    cfg = load_config(config_path)
    state = _require_state(label)
    ids = ",".join(v for v in state.get("jobs", {}).values() if v)
    if not ids:
        raise SystemExit(f"[{label}] no job ids recorded")
    queued = _ssh(cfg, f"squeue -j {shlex.quote(ids)} -h 2>/dev/null || true",
                 check=False)
    return bool(queued.strip())


def status(*, label: str, config_path: str | Path | None = None) -> None:
    """Report where this experiment's jobs are.

    Args:
        label: The experiment name.
        config_path: The ``cluster.toml`` to read, defaulting to the repo root's.
    """
    cfg = load_config(config_path)
    state = _require_state(label)
    ids = ",".join(v for v in state.get("jobs", {}).values() if v)
    if not ids:
        raise SystemExit(f"[{label}] no job ids recorded")

    print(f"[{label}] {state['runid']} -> {ids}")
    queued = _ssh(cfg, f"squeue -j {shlex.quote(ids)} "
                       "-o '%.18i %.12j %.9T %.10M %R' 2>/dev/null || true",
                       check=False)
    if len(queued.strip().splitlines()) > 1:
        print("-- squeue --")
        print(queued.rstrip())
    else:
        print("-- squeue: nothing queued or running --")

    print("-- sacct --")
    done = _ssh(cfg, f"sacct -j {shlex.quote(ids)} -X "
                     "--format=JobID%-18,JobName%-14,State%-14,Elapsed,ExitCode "
                     "2>/dev/null || true", check=False)
    print(done.rstrip() if done.strip() else "(no accounting data yet)")


def logs(
    *,
    label: str,
    config_path: str | Path | None = None,
    task: str | None = None,
) -> None:
    """Pull this run's slurm output into ``.cluster/logs/<label>/``.

    Args:
        label: The experiment name.
        config_path: The ``cluster.toml`` to read, defaulting to the repo root's.
        task: An array task index. Given one, print that task's log rather than
            listing the fetched files.
    """
    cfg = load_config(config_path)
    state = _require_state(label)
    dest = repo_root() / ".cluster" / "logs" / label
    dest.mkdir(parents=True, exist_ok=True)

    proc = _run(["rsync", "-az", _remote_path(cfg, state["rundir"] + "/logs/"),
                 str(dest) + "/"])
    _check_auth(cfg, proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"[{label}] could not fetch logs\n{proc.stderr.strip()}")

    if task is None:
        found = sorted(p.name for p in dest.iterdir() if p.is_file())
        print(f"[{label}] {dest}")
        print("\n".join(f"  {name}" for name in found) if found else "  (no logs yet)")
        return

    matches = sorted(dest.glob(f"*_{task}.out"))
    if not matches:
        raise SystemExit(f"[{label}] no log for task {task} in {dest}")
    print(matches[0].read_text(), end="")


def setup(*, config_path: str | Path | None = None) -> None:
    """Provision the cluster: bare repo, uv, and both shared venvs.

    Safe to re-run, and worth re-running after a scratch purge. Building the
    venvs needs the matching pyproject.toml and uv.lock on the cluster, so HEAD
    is pushed and snapshotted into a scratch dir that is removed afterwards.

    Args:
        config_path: The ``cluster.toml`` to read, defaulting to the repo root's.

    Raises:
        SystemExit: If the working tree is dirty.
    """
    cfg = load_config(config_path)
    sha = _require_clean_tree()
    print(f"setting up {cfg.host} at {cfg.root}", file=sys.stderr)

    # Expand $HOME on the cluster before anything creates directories with it.
    out = _ssh_script(cfg, _remote_dir() / "bootstrap.sh", _remote_root(cfg),
                      cfg.project)
    root, bare = _field(out, "ROOT"), _field(out, "BARE")
    if not root or not bare:
        raise SystemExit("bootstrap.sh did not report a root and bare repo")

    _set_remote(cfg, bare)
    _push(cfg, sha, "refs/gsp/setup")

    # A throwaway snapshot, only so uv has the pyproject.toml and uv.lock to sync from.
    snapshot = _field(
        _ssh_script(
            cfg, _remote_dir() / "prepare.sh", root, ".setup", sha, ".setup", ".",
            cfg.project
        ),
        "RUNDIR",
    )
    if not snapshot:
        raise SystemExit("prepare.sh did not report a snapshot for the venv build")
    try:
        built = {}
        for name in ("cpu", "gpu"):
            extras = cfg.venvs.get(name, [])
            built[name] = _field(
                _ssh_script(
                    cfg, _remote_dir() / "build_env.sh", root, name,
                    _lock_hash(sha, extras, cfg.post_sync), snapshot,
                    cfg.post_sync, *extras
                ),
                "VENV",
            )
    finally:
        _ssh(cfg, f"rm -rf {shlex.quote(snapshot)} "
                  f"{shlex.quote(root + '/results/.setup')}", check=False)

    print(f"\nready on {cfg.host}\n"
          f"  root       {root}\n"
          f"  bare repo  {bare}\n"
          f"  cpu venv   {built['cpu']}\n"
          f"  gpu venv   {built['gpu']}\n"
          f"  remote     {_remote_name(cfg)}", file=sys.stderr)
    if not cfg.account:
        found = _field(out, "ACCOUNTS") or "none found under ~/projects"
        print(f"\nset account in {cfg.path} before dispatching (candidates: {found})",
              file=sys.stderr)
