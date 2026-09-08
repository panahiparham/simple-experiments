# simple-experiments

Declarative sweeps, sharded runs, and a SQLite results store, with a shared
CLI and optional SLURM dispatch. The harness knows nothing about what a run
computes: you give it an `Experiment` and one function that computes a shard
of runs.

## Features

- Declarative sweeps over (possibly nested) frozen-dataclass configs
- Runs sharing static config fields are packed into shards and computed
  together in one call
- Content-addressed run and config ids: extending a sweep by hyperparameter
  or by seed computes only the delta, and an interrupted sweep resumes where
  it stopped
- One SQLite database per experiment, with per-worker part files merged
  safely after a concurrent sweep
- One CLI (`status`, `single`, `sweep`, `sync`, `queue`, `logs`) shared
  between local and SLURM execution
- Depends only on numpy and polars - no jax, no matplotlib, no plotting

## Example usage

### Define an experiment

```python
# config.py
from dataclasses import dataclass
from pathlib import Path

from experiment import Component, Experiment, traced


@dataclass(frozen=True)
class OptimizerCfg:
    LR: float = traced(0.1)


@dataclass(frozen=True)
class Cfg:
    OPTIMIZER: OptimizerCfg = OptimizerCfg()
    NUM_STEPS: int = 200


EXPERIMENT = Experiment(
    name="quadratic_descent",
    results_dir=Path(__file__).resolve().parent / "results",
    components=[
        Component(name="baseline", config=Cfg(), seeds=list(range(20)),
                  shard_size=5),
        Component(name="tuned", config=Cfg(),
                  sweep={"OPTIMIZER.LR": [0.01, 0.05, 0.2]},
                  seeds=list(range(20)), shard_size=5),
    ],
)
```

A config is any (possibly nested) frozen dataclass whose `dataclasses.asdict`
is JSON-serialisable. `traced` marks a field as a number a run reads as it
goes; runs that differ only in traced values are handed to the shard
function together, so it can compute them in one batched call. Everything
else is static and fixes shapes and objects - sweeping a static field is not
an error, it just puts those runs in separate shards.

### Run the experiment

```python
# run.py
import numpy as np

from experiment import run


def process_shard(configs, seeds):
    results = []
    for config, seed in zip(configs, seeds):
        rng = np.random.default_rng(seed)
        x = rng.normal()
        for _ in range(config.NUM_STEPS):
            x -= config.OPTIMIZER.LR * 2 * x
        results.append({"final_x": np.array(x)})
    return results


if __name__ == "__main__":
    run(EXPERIMENT, process_shard)
```

The shard function's only contract is `(configs, seeds) -> one result per
run`, in the order given. It can loop, vectorise, or call out elsewhere -
the harness does not care.

```bash
run.py status                        # runs done, runs pending, shards pending
run.py sweep --num-workers 6         # across 6 local worker processes
run.py sweep --num-workers 6 --slurm # the same work as a SLURM array
run.py single --component tuned --seed 0
run.py sync                          # bring the cluster's results home
run.py queue | logs                  # cluster only
```

`--set PATH=VALUE` overrides a config field on `single`, `sweep` or `status`,
using the same dotted paths a sweep uses:

```bash
run.py sweep --set OPTIMIZER.LR=0.15
run.py single --component tuned --seed 0 --set NUM_STEPS=500
```

A value is read as the type of the field it replaces. Precedence is the base
config, then `--set`, then the sweep, so sweeping a path always wins over
setting it.

### Access results

```python
from experiment import load_result, load_runs

df = load_runs(EXPERIMENT, "tuned")  # run_id, config_id, seed, OPTIMIZER.LR, ...
for run_id, lr in zip(df["run_id"], df["OPTIMIZER.LR"]):
    result = load_result(EXPERIMENT, "tuned", run_id)
    print(lr, result["final_x"])
```

`load_runs` returns a polars DataFrame with one row per run, its config
flattened to dotted columns, but not the result itself - that stays a
binary blob until `load_result` or `load_array` reads one run's back.

## Setting up on a cluster

`experiment.slurm` reads a `cluster.toml` from your project's repo root,
which is also how the harness locates that root:

```toml
[project]
name = "my-project"                  # the cluster's bare repo is <name>.git
src_dirs = ["src"]                   # prepended to a job's PYTHONPATH
post_sync = "scripts/setup_venv.sh"  # optional, run after a venv is synced

[cluster]
host = "my-cluster"                  # an ssh alias that works non-interactively
root = "$HOME/scratch/my-project"
account = "my-slurm-account"

[venvs]
cpu = []
gpu = ["cuda"]

[slurm]
time = "01:00:00"
cpus_per_task = 1
mem_per_cpu = "4G"
gpus = 0

[experiments.quadratic_descent]      # per-experiment overrides, keyed by
time = "00:30:00"                    # the Experiment's name
```

```python
# setup_cluster.py
from experiment.slurm import setup

setup(config_path="cluster.toml")
```

`setup` creates a bare repo on the cluster as the push target, installs
`uv`, and builds one shared venv per entry in `[venvs]`. A dispatch snapshots
exactly one commit (`git archive`, no working checkout), so a queued job's
code can never change underneath it, and a shared venv is re-synced only
when `uv.lock` moves. A cluster sweep runs as three chained jobs: one
deciding the plan, an array working through it, and one merging what the
array wrote.

A project that needs something `uv sync` cannot install can name a script of
its own with `[project] post_sync`, a path inside the repo. It runs from the
snapshot once a venv is synced, with `EXPERIMENT_VENV` set to the venv it
must install into and `EXPERIMENT_EXTRAS` to the extras that venv was built
with, so one script can serve several venvs. What the script contains counts
towards the venv's identity, so editing it rebuilds the venvs it applies to,
and a venv that is already up to date runs nothing.

`EXPERIMENT_LOCAL_MODE=1` runs every "remote" command in a local shell,
which is how the cluster flow is exercised without a cluster.

## Migrating an older store

An earlier layout kept one database per component. `run.py migrate` folds
those into the experiment's database. Run ids are unchanged, so migrated
runs still count as done. The old files are left in place.

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

## Tests

```bash
uv run pytest
```

`tests/` covers the harness with fake configs and a fake shard function:
`test_hypers.py` the traced/static split, `test_plan.py` enumeration and
planning, `test_results.py` the store, `test_legacy.py` migration, and
`test_commands.py` the CLI end to end. `experiment.slurm` is best covered
from the consuming project's own test suite, against a sandbox repo built
by its fixtures.
