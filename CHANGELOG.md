# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-13

### Added

- Per-rollout `prediction` (mapped SUPPORT/NEI/CONTRADICT/INVALID verdict) and
  `pred_score` (raw parsed integer) columns recorded next to the ground-truth
  `answer`, auto-persisted by the environment without a `--state-columns` flag.

### Changed

- Made benchmark reporting imports lightweight by moving shared dataset constants
  out of the Verifiers-backed taskset path.

## [0.1.0] - 2026-07-12

### Added

- Native Verifiers 0.2 Taskset/Harness environment for MedFact-Bench.
- Pinned Hugging Face and local Parquet dataset loading.
- Paper-faithful score parsing and three-class zero-shot reward.
- Parseability and strict-format rollout metrics.
- Streaming benchmark reporting for legacy results and Verifiers 0.2 traces.
- Offline tests, opt-in dataset integration tests, and reproducible evaluation config.

[Unreleased]: https://github.com/KouateMuhamed/medfact-bench/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/KouateMuhamed/medfact-bench/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/KouateMuhamed/medfact-bench/releases/tag/v0.1.0
