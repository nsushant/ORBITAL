# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/nsushant/ORBITAL/compare/v0.1.0...MDLS
[0.1.0]: https://github.com/nsushant/ORBITAL/releases/tag/v0.1.0
