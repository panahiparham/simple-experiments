# experiment-harness

Declarative sweeps, run-packed shards, and a SQLite results store, with a shared
CLI and optional SLURM dispatch. The harness knows nothing about what a run
computes: you give it an experiment and one function that computes a shard of
runs.

Depends on numpy and polars. No jax, no matplotlib, no plotting.

## What you supply

Two things, both passed to `run`:

| Element | Contract |
|---|---|
| `Experiment` | a name, a results directory, and its `Component`s |
| shard function | `(configs, seeds) -> one result per run`, in the order given |

A `Component` is `Component(name, config, sweep, seeds, shard_size)`: a base
config, a sweep over it, the seeds to run each point at, and how many runs to
pack into a shard.

A config is any (possibly nested) frozen dataclass whose `dataclasses.asdict` is
JSON-serialisable. Its fields are the hyperparameters.

## An experiment

```python
# config.py
from experiment import Component, Experiment

EXPERIMENT = Experiment(
    name="my_experiment",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(name="baseline", config=Cfg(), seeds=list(range(30)),
                  shard_size=5),
        Component(name="tuned", config=Cfg(), sweep={"LR": [1e-3, 3e-4]},
                  seeds=list(range(30)), shard_size=5),
    ],
)
```

```python
# run.py
from experiment import run

run(EXPERIMENT, process_shard)
```

```bash
run.py status                    # runs done, runs pending, shards pending
run.py sweep --num-workers 6     # here, across 6 local worker processes
run.py sweep --num-workers 6 --slurm     # the same work as a SLURM array
run.py single --component tuned --seed 0
run.py sync                      # bring the cluster's results home
run.py queue | logs              # cluster only
```

## Overrides

`--set PATH=VALUE` overrides a config field on `single`, `sweep` or `status`,
using the same dotted paths a sweep uses:

```bash
run.py sweep --set AGENT_HYPERS.GAMMA=0.95
run.py single --component tuned --seed 0 --set AGENT_HYPERS.LR=0.001
```

A value is read as the type of the field it replaces. Precedence is the base
config, then `--set`, then the sweep, so sweeping a path always wins over
setting it.

## Traced and static fields

A config field is either static or traced. Traced fields are numbers a run reads
as it goes, and runs that differ only in traced values are computed together in
one batched call. Everything else is static, and fixes shapes and objects.

Declare traced fields where they are defined:

```python
from experiment import traced

@dataclass(frozen=True)
class Cfg:
    LR: float = traced(3e-4)
    GAMMA: float = traced(0.99)
    HIDDEN_SIZE: int = 64        # static
```

Sweeping a static field is not an error. Those runs land in separate shards.

## Shards

A shard is a sequence of runs sharing every static field, so the whole shard can
be computed in one call. `shard_size` counts runs; leaving it unset puts every
batchable run of a component into a single shard.

`status` reports the shards still pending and the worker count that would
saturate them.

## Identity and resume

`config_id` is a content hash of the config with the seed excluded, and
`run_id = "<config_id>_s<seed>"`. A run's PRNG derives from its integer seed
alone, so results do not depend on how work was sharded. Runs already stored are
dropped before shards are packed, so extending a sweep by hyperparameter or by
seed computes only the delta, and an interrupted sweep resumes where it stopped.

A sweep's plan is built once, by the process you invoked, and each worker is
handed its share.

## Storage

One database per experiment at `<results_dir>/<name>.db`, holding one table per
component and one row per run: its id, its config's id, its seed, the config,
and the result as a binary blob.

A shared SQLite file is not safe under concurrent writes, so each worker writes
its own `<name>.parts/part-<k>.db` and those are merged at the end of a sweep.
Reads union the merged database with any parts still present.

Read results back with `load_runs` (a polars DataFrame, one row per run, config
flattened to dotted columns), `load_result` and `load_array`.

## Migrating an older store

An earlier layout kept one database per component. `run.py migrate` folds those
into the experiment's database. Run ids are unchanged, so migrated runs still
count as done. The old files are left in place.

## Modules

| Module | Holds |
|---|---|
| `experiment.design` | `Experiment` and `Component` |
| `experiment.hypers` | dotted paths, `traced`, the traced/static split |
| `experiment.identity` | `config_id` and `run_id` |
| `experiment.plan` | sweep expansion, shards, planning, worker assignment |
| `experiment.results` | the SQLite store, parts, merging, the read API |
| `experiment.runner` | the loop that hands shards to the shard function |
| `experiment.commands` | `run`, the CLI every `run.py` shares |
| `experiment.slurm` | cluster dispatch, fetch, queue, logs, and `setup` |
| `experiment.legacy` | one-shot migration from the per-component store |

## Cluster configuration

`experiment.slurm` reads `cluster.toml` from the repo root, which is also how the
harness locates that root. The `[project]` table is what keeps it
project-agnostic:

```toml
[project]
name = "my-project"                  # the cluster's bare repo is <name>.git
src_dirs = ["src", "experiment/src"] # prepended to a job's PYTHONPATH
```

Resources come from `[slurm]`, with per-experiment overrides in
`[experiments.<name>]` keyed by the experiment's name.

A cluster sweep runs as three chained jobs: one deciding the plan, an array
working through it, and one merging what the array wrote.

`EXPERIMENT_LOCAL_MODE=1` runs every "remote" command in a local shell, which is
how the cluster flow is exercised without a cluster.

## Tests

```bash
uv run pytest
```

`experiment/tests/` covers the harness with fake configs and a fake shard
function: `test_hypers.py` the traced/static split, `test_plan.py` enumeration
and planning, `test_results.py` the store, `test_legacy.py` migration, and
`test_commands.py` the CLI end to end. `experiment.slurm` is covered from the
consuming project's suite instead, against a sandbox repo built by its fixtures.
