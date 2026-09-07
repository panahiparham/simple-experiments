# Definition of an Experiment
- A research project has several Experiments.
- Each Experiment is a self contained unit of work comprised of one or more Components.

1. Experiment
    1. name: a string that identifies the experiment.
    2. a list of Components.

- Each Component defines several Runs of the experiment each with a config and seed.

2. Component
    1. name: a string that identifies the component.
    2. config: a named tuple with any structure, potentially nested, that contains the configuration for the experiment.
    3. Sweep: a dict mapping config keys to list of values to sweep over. These would override the values in the config for each run of the experiment when specified.
    4. Seeds: a list of integers.
    5. Shard size: an integer that defined how many runs to pack into a Shard.

- A Shard is a sequence of Runs that is supposed to be executed together.

3. Shard
    1. Sequence of configs
    2. Sequence of seeds

# Running an Experiment
- An experiment is run either locally or on a cluster.
- In both casese, a fully specified Experiment is expanded into a set of Shards, filtering out any run whose result already exists before partitioning the remaining runs into Shards.
- The user specifies num-workers and the Shards are distributed equally among the workers to be proceessed in parallel. Within each worker, the Shards are processed sequentially.
- The user must provide a function that processes a Shard. So it takes a sequence of configs and a sequence of seeds and returns a sequence of results.
- The results of each Run is stored in a databse in the results folder. One database for each experiment. One table for each Component. One row for each Run. The row contained the run_id (a hash of the config and seed), the config, the seed, and the result stored as a binary blob. The database is used to filter out runs that have already been completed and to store the results of new runs.

# Seamless local and cluster execution
- The user should be able to run the same commands whether local or on a slurm cluster to run and inspect an experiment
- The user should be able to run a single run of an experiment by specifying the component and seed. Base configs are used for this run.
- The user should be able to check how many runs have been completed and how many runs / shards remain to be completed.
- The user should be able to run a full sweep of the experiment
- The user should be able to run altered single or sweep by specifying config overrides with command line arguments. The overrides when used with sweeps will override the config but not the sweep vales which will override the command line override.
- The user should be able to run a single command to sync the results folder from the cluster to local machine.

# What the user provides
- Experiment definitions: name, list of components, path to results folder.
- The config definitions depending of the nature of the project.
- The function that processes a Shard
    - In our jax codebase we can vmap over seeds but only some configs like lr. Other configs like model architecture can not be vmapped. So for simplicity, let's assume we can indeed vmap over runs in a component. So the Shard processing function will be a single vmap over the configs and seeds in the Shard.
- A cluster.toml file that defines the cluster configuration (hostname, user, path to project on remote, etc.)
