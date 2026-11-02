# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, a minor release may break the API.

## [Unreleased]

## [0.2.0] - 2026-11-01

### Added

- Metrics carry the step each value was recorded at (#24).
- `Component.record` names the metrics a component's runs must store (#24).
- Results databases record their schema version (#25).
- Results stored by 0.1.0 are upgraded the first time they are read or
  written (#25).

### Changed

- Shard functions take a third `record` argument and return each metric as a
  `(steps, values)` pair (#24).
- `load_metric` returns a `(steps, values)` pair per run (#24).
- Saving a stored run again replaces all of its metrics (#24).

## [0.1.0] - 2026-10-24

First release: declarative sweeps, sharded runs, one SQLite results database
per component, and a CLI shared between local and SLURM execution.

[Unreleased]: https://github.com/panahiparham/simple-experiments/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/panahiparham/simple-experiments/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/panahiparham/simple-experiments/releases/tag/v0.1.0
