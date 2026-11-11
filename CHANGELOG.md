# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-11

### Added

- An exact dominance index for large archives. Once the front reaches 8192
  points, `Archive.add` switches automatically from the linear scan to a forest
  of static bounding-volume trees queried by Numba. Fronts with many objectives
  now grow without a quadratic scan: at 100 000 archived points an m=6 insert
  costs about 20-30 us instead of ~230 us. Results are identical to the scan;
  nothing is approximated or truncated. If evictions become frequent the index
  disables itself and the run finishes on the scan.
- `tests/test_dominance.py` for the index kernels, forest and archive
  differential tests, and slow-marked performance guards in
  `tests/test_performance.py` (run them with `python -m pytest -m slow`).

### Changed

- The scan path appends in place when an insert evicts nothing, instead of
  compacting the whole buffer and rebuilding the solution list.

## [0.1.2] - 2026-11-10

### Changed

- The README testing section now includes the clone and branch steps, since
  `pip install -e` needs a checkout, and uses `python -m pytest`.

## [0.1.1] - 2026-11-10

### Fixed

- Corrected the MDLS citation to Tricoire (2012), *Computers & Operations
  Research* 39(12), 3089-3101. The previous entry named the wrong paper and
  listed co-authors who did not write it.

## [0.1.0] - 2026-11-10

### Added

- `MDLS` engine with an archive of feasible non-dominated solutions, uniform
  selection by default, and iteration, evaluation and wall-clock budgets.
- `@move(improves=k, attempts=n)` decorator for readable, configurable moves.
- `Archive` with relative-tolerance dominance, duplicate rejection, dynamic
  growth, `knee()` and `extreme(k)` helpers.
- `Result` and `MoveStats` with per-move accounting, including dominated
  candidates, and a text `summary()`.
- Extension points: `is_feasible`, `selection`, `copy_solution`, and the
  `select`, `offer`, `copy` and `on_iteration` hooks.
- Logging-based progress output (`verbose=n`) instead of printing.
- Bi-objective knapsack example with `run` and `plot` commands.
- Test suite, user guide, packaging metadata and GitHub Actions workflows for
  CI and PyPI trusted publishing.

The PyPI distribution is named `pymdls` because `mdls` is blocked by PyPI's
name-similarity rule (it collides with the existing project `mdis`). The
import package is still `mdls`.

[Unreleased]: https://github.com/nsushant/ORBITAL/compare/v0.2.0...MDLS
[0.2.0]: https://github.com/nsushant/ORBITAL/releases/tag/v0.2.0
[0.1.2]: https://github.com/nsushant/ORBITAL/releases/tag/v0.1.2
[0.1.1]: https://github.com/nsushant/ORBITAL/releases/tag/v0.1.1
[0.1.0]: https://github.com/nsushant/ORBITAL/releases/tag/v0.1.0
