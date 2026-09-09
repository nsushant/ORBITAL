# Results used in the manuscript

Files are grouped by their role rather than manuscript section number.

- `demands/`: 15 fixed realisations for each demand scenario.
- `algorithm_fronts/`: paired MDLS and NSGA-II fronts for all 45 instances.
- `metrics/`: client-population and per-trial performance data.
- `tuning/`: final `irace` configurations and summaries.
- `validation/`: numerical transfer-model comparisons.
- `depot_placement/`: orbital sweeps and lexicographic MILP records. The
  `depot_milp_verified_*` files record status, primal objective, dual bound,
  MIP gap, and selected architecture.
- `figures/`: the four PDF figures used in the manuscript.

The raw transfer lookup table is a large generated intermediate and is not
duplicated here. See the repository README for regeneration guidance.
