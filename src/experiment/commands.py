"""The command line every experiment's ``run.py`` shares.

One entry point with a mode as its first argument. The same command means the
same thing whether the work happens here or on a cluster - ``--slurm`` is the
only difference between running a sweep locally and running it as a cluster
array.
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from experiment.design import Component, Experiment
from experiment.legacy import migrate
from experiment.plan import (
    assign_shards,
    component_runs,
    pack_shards,
    plan_experiment,
)
from experiment.results import _parts_dir, completed, merge_parts
from experiment.runner import ShardFn, run_shards

__all__ = ["parse_overrides", "run"]


MODES = ("single", "sweep", "status", "queue", "sync", "logs", "migrate")


def parse_overrides(assignments: Sequence[str]) -> dict[str, str]:
    """Parse ``--set PATH=VALUE`` assignments into a mapping.

    Values stay as text here. They are read as the type of the field they
    replace when they reach a component's config, so one override can span
    components that type the same path differently.

    Args:
        assignments: The ``PATH=VALUE`` strings as given on the command line.

    Returns:
        A ``{dotted path: text value}`` mapping. A path given twice keeps the
        last value.

    Raises:
        SystemExit: If an assignment has no ``=``, or names no path.
    """
    overrides: dict[str, str] = {}
    for assignment in assignments:
        path, separator, value = assignment.partition("=")
        if not separator or not path.strip():
            raise SystemExit(
                f"--set expects PATH=VALUE, e.g. --set AGENT_HYPERS.LR=0.001; "
                f"got {assignment!r}"
            )
        overrides[path.strip()] = value
    return overrides


def _add_override_flag(parser: argparse.ArgumentParser) -> None:
    """Add the ``--set`` flag, which every mode that runs work accepts."""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a config field, e.g. --set AGENT_HYPERS.LR=0.001",
    )


def _one_component(experiment: Experiment, name: str | None) -> Component:
    """Pick the component a single run belongs to.

    Args:
        experiment: The experiment being run.
        name: The requested component, or ``None`` to take the only one.

    Returns:
        The component.

    Raises:
        SystemExit: If the name is unknown, or omitted when there is a choice.
    """
    if name is not None:
        try:
            return experiment.component(name)
        except KeyError as error:
            raise SystemExit(str(error)) from None
    if len(experiment.components) == 1:
        return experiment.components[0]
    defined = sorted(c.name for c in experiment.components)
    raise SystemExit(
        f"[{experiment.name}] single needs --component NAME (one of: {defined})"
    )


def _single(experiment: Experiment, process: ShardFn, argv: list[str]) -> None:
    """Run one seed of one component, from its base config.

    The component's sweep is ignored - a single run is the base config, plus
    whatever ``--set`` overrides were given.
    """
    parser = argparse.ArgumentParser(prog="run.py single")
    parser.add_argument("--component", default=None)
    parser.add_argument("--seed", type=int, default=0)
    _add_override_flag(parser)
    args = parser.parse_args(argv)

    component = _one_component(experiment, args.component)
    only = dataclasses.replace(
        component, sweep={}, seeds=(args.seed,), shard_size=None
    )
    scoped = dataclasses.replace(experiment, components=(only,))

    shards = plan_experiment(
        scoped,
        done=completed(scoped),
        overrides=parse_overrides(args.overrides),
    )
    label = f"[{experiment.name}:{component.name}]"
    if not shards:
        print(f"{label} seed {args.seed} is already stored")
        return

    saved = run_shards(experiment, shards, process)
    merge_parts(experiment)
    print(f"{label} stored {saved} run(s) for seed {args.seed}")


def _sweep_parser() -> argparse.ArgumentParser:
    """The options ``sweep`` takes, as launcher and as worker."""
    parser = argparse.ArgumentParser(prog="run.py sweep")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--component", nargs="+", default=None)
    _add_override_flag(parser)
    # A local sweep does all three steps itself. A cluster sweep schedules them
    # as separate jobs, so each is reachable on its own: decide the plan once,
    # work through it in parallel, then merge what the workers wrote. Set by the
    # launcher, not by hand.
    parser.add_argument("--write-plan", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plan", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--merge-only", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def _run_workers(assignments: list, plan_path: Path) -> None:
    """Run one child process per worker and wait for all of them.

    Each child re-invokes this experiment's ``run.py``, so it inherits whatever
    environment the parent was started with - single-threaded XLA included.

    Raises:
        SystemExit: If any worker exits non-zero.
    """
    children = [
        subprocess.Popen(
            [
                sys.executable,
                sys.argv[0],
                "sweep",
                "--plan",
                str(plan_path),
                "--worker-index",
                str(index),
            ]
        )
        for index in range(len(assignments))
    ]
    codes = [child.wait() for child in children]
    if any(codes):
        raise SystemExit(f"worker(s) failed with exit codes {codes}")


def _sweep(experiment: Experiment, process: ShardFn, argv: list[str]) -> None:
    """Run every outstanding run of an experiment, across a pool of workers.

    The plan is built once, by whichever process the user invoked, and each
    worker is handed its share. Workers recomputing it would each see a
    different set of stored results depending on when they started, and could
    then disagree about which of them owns a run.
    """
    args = _sweep_parser().parse_args(argv)

    if args.merge_only:
        print(f"[{experiment.name}] merged {merge_parts(experiment)} run(s)")
        return

    if args.plan is not None:
        if args.worker_index is None:
            raise SystemExit("--plan needs --worker-index")
        mine = pickle.loads(Path(args.plan).read_bytes())[args.worker_index]
        saved = run_shards(experiment, mine, process, worker=args.worker_index)
        print(
            f"[{experiment.name}] worker {args.worker_index} stored "
            f"{saved} run(s) from {len(mine)} shard(s)"
        )
        return

    plan = plan_experiment(
        experiment,
        done=completed(experiment),
        overrides=parse_overrides(args.overrides),
        components=args.component,
        shard_size=args.shard_size,
    )

    if args.write_plan is not None:
        # One entry per worker, uncapped: a cluster array is sized when it is
        # submitted, which is before this runs, so every task must find a share
        # waiting for it even if that share is empty.
        path = Path(args.write_plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(assign_shards(plan, max(1, args.num_workers))))
        runs = sum(len(shard) for shard in plan)
        print(
            f"[{experiment.name}] planned {runs} run(s) in {len(plan)} shard(s) "
            f"for {args.num_workers} worker(s) -> {path}"
        )
        return

    if not plan:
        print(f"[{experiment.name}] nothing to run; every run is already stored")
        return

    runs = sum(len(shard) for shard in plan)
    workers = max(1, min(args.num_workers, len(plan)))
    print(
        f"[{experiment.name}] {runs} run(s) in {len(plan)} shard(s) "
        f"across {workers} worker(s)"
    )

    if workers == 1:
        run_shards(experiment, plan, process)
    else:
        assignments = assign_shards(plan, workers)
        plan_path = _parts_dir(experiment) / "plan.pickle"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(pickle.dumps(assignments))
        try:
            _run_workers(assignments, plan_path)
        finally:
            plan_path.unlink(missing_ok=True)

    merged = merge_parts(experiment)
    print(f"[{experiment.name}] stored {merged} run(s)")


def _select(experiment: Experiment, names: list[str] | None) -> list[Component]:
    """The components a mode is to act on, in the order they are declared.

    Raises:
        SystemExit: If a name is not one the experiment defines.
    """
    if names is None:
        return list(experiment.components)
    try:
        return [experiment.component(name) for name in names]
    except KeyError as error:
        raise SystemExit(str(error)) from None


def _status(experiment: Experiment, argv: list[str]) -> None:
    """Report how much of an experiment is done and how much is left.

    Reads only the results directory, so it answers the same either side of a
    cluster run and needs no connection to one.
    """
    parser = argparse.ArgumentParser(prog="run.py status")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--component", nargs="+", default=None)
    _add_override_flag(parser)
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.overrides)
    stored = completed(experiment)
    total_runs = total_done = total_shards = 0
    for component in _select(experiment, args.component):
        runs = component_runs(component, overrides)
        done = stored.get(component.name, set())
        pending = [run for run in runs if run.id not in done]
        size = (
            args.shard_size
            if args.shard_size is not None
            else component.shard_size
        )
        shards = pack_shards(component.name, pending, size)
        total_runs += len(runs)
        total_done += len(runs) - len(pending)
        total_shards += len(shards)
        print(
            f"[{component.name}] {len(runs)} run(s): "
            f"{len(runs) - len(pending)} done, {len(pending)} pending "
            f"in {len(shards)} shard(s)"
        )

    summary = (
        f"{experiment.name}: {total_runs} run(s), {total_done} done, "
        f"{total_runs - total_done} pending"
    )
    if total_shards:
        summary += f" in {total_shards} shard(s) -> up to --num-workers {total_shards}"
    print(summary)


def _migrate(experiment: Experiment, argv: list[str]) -> None:
    """Fold results from the old per-component databases into this one."""
    argparse.ArgumentParser(prog="run.py migrate").parse_args(argv)

    moved = migrate(experiment)
    if not moved:
        print(f"[{experiment.name}] no legacy databases to migrate")
        return
    for component, count in moved.items():
        print(f"[{component}] migrated {count} run(s)")
    print(
        f"{experiment.name}: migrated {sum(moved.values())} run(s) from "
        f"{len(moved)} legacy database(s); the old files are left in place"
    )


@dataclasses.dataclass(frozen=True)
class _Cluster:
    """The ``--slurm`` options, taken out of a mode's own arguments."""

    enabled: bool = False
    dry_run: bool = False
    config: str | None = None


def _split_cluster_flags(argv: list[str]) -> tuple[list[str], _Cluster]:
    """Split the cluster flags out of ``argv``, leaving the mode's own.

    Hand-rolled rather than argparse so a mode's options - which differ per mode -
    can never be consumed, reordered or prefix-abbreviated on their way through.

    Raises:
        SystemExit: If ``--slurm-config`` is given without a path.
    """
    rest: list[str] = []
    enabled = dry_run = False
    config: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--slurm":
            enabled = True
        elif argument == "--slurm-dry-run":
            enabled = dry_run = True  # implies --slurm; passing both is redundant
        elif argument == "--slurm-config":
            index += 1
            if index >= len(argv):
                raise SystemExit("--slurm-config needs a path")
            config = argv[index]
        elif argument.startswith("--slurm-config="):
            config = argument.split("=", 1)[1]
        else:
            rest.append(argument)
        index += 1
    return rest, _Cluster(enabled=enabled, dry_run=dry_run, config=config)


def _sync(experiment: Experiment, cluster: _Cluster, argv: list[str]) -> None:
    """Bring the cluster's results home and merge them into the local store."""
    argparse.ArgumentParser(prog="run.py sync").parse_args(argv)

    from experiment import slurm

    slurm.fetch(experiment, config_path=cluster.config)
    print(f"[{experiment.name}] merged {merge_parts(experiment)} run(s)")


def run(
    experiment: Experiment,
    process: ShardFn,
    argv: list[str] | None = None,
) -> None:
    """Run the command line shared by every experiment's ``run.py``.

    Adding ``--slurm`` to ``single`` or ``sweep`` runs that same work on the
    cluster instead of here, so the workflow is one command and one flag either
    way. ``sync``, ``queue`` and ``logs`` only ever concern the cluster.

    Args:
        experiment: The experiment to act on.
        process: The project's shard function.
        argv: The arguments, defaulting to the process's own.

    Raises:
        SystemExit: If no mode is given, or the mode is not one this
            understands.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    argv, cluster = _split_cluster_flags(argv)
    if not argv or argv[0] not in MODES:
        given = argv[0] if argv else "<none>"
        raise SystemExit(
            f"[{experiment.name}] a mode is required as the first argument "
            f"(one of: {', '.join(MODES)}); got {given!r}"
        )

    mode, rest = argv[0], argv[1:]

    if mode in ("sync", "queue", "logs"):
        from experiment import slurm

        if mode == "sync":
            _sync(experiment, cluster, rest)
        elif mode == "queue":
            slurm.status(label=experiment.name, config_path=cluster.config)
        else:
            slurm.logs(
                label=experiment.name,
                config_path=cluster.config,
                task=rest[0] if rest else None,
            )
        return

    if cluster.enabled:
        if mode != "single" and mode != "sweep":
            raise SystemExit(
                f"[{experiment.name}] {mode} runs here, not on the cluster"
                + (
                    "; queue reports what the cluster is doing"
                    if mode == "status"
                    else ""
                )
            )
        from experiment import slurm

        slurm.dispatch(
            label=experiment.name,
            run_py=Path(sys.argv[0]).resolve(),
            mode=mode,
            argv=rest,
            config_path=cluster.config,
            dry_run=cluster.dry_run,
        )
        return

    if mode == "single":
        _single(experiment, process, rest)
    elif mode == "sweep":
        _sweep(experiment, process, rest)
    elif mode == "status":
        _status(experiment, rest)
    elif mode == "migrate":
        _migrate(experiment, rest)
