# Multi-objective mission planning for depot-supported on-orbit servicing

This is the Python research-code and result release accompanying the paper
*Multi-objective mission planning for depot-supported on-orbit servicing*.
It contains the mission model, multi-directional local search (MDLS), the
NSGA-II comparator, paired-trial analysis, and lexicographic depot-location
MILP. The released workflow is Python-only; superseded development code and
intermediate files are excluded from the release.

The links below refer to manuscript topics and figure contents rather than
section or figure numbers, so they remain valid if numbering changes in review.

## Paper-to-repository map

| Manuscript content | Implementation | Released evidence |
|---|---|---|
| Client population and three demand scenarios | `build_instance.py`, `generate_demands.py`, `starlink/` | `paper_results/metrics/instance_population.csv`, `paper_results/demands/` |
| Time-dependent low-thrust transfer costs and schedule evaluation | `oos/edelbaum.py`, `oos/lu.py`, `oos/phasing.py`, `oos/schedule.py` | `paper_results/validation/` |
| Greedy initial solution | `oos/greedy.py` | `tests/test_greedy.py` |
| Multi-directional local search | `oos/mdls.py`, `run_mdls_trial.py` | MDLS fronts in `paper_results/algorithm_fronts/` |
| NSGA-II request-level encoding | `oos/ga_problem.py`, `run_ga_trial.py` | NSGA-II fronts in `paper_results/algorithm_fronts/` |
| Common `irace` tuning protocol | `run_irace.sh`, `tune/` | `paper_results/tuning/` |
| Paired hypervolume, knee-distance and Wilcoxon comparisons | `analyse_fronts.py` | `paper_results/metrics/section4_metrics.csv` |
| Hypervolume and knee-distance figure | `plots/make_sec4_figures.py` | `paper_results/figures/sec4_hypervolume_knee.pdf` |
| Fleet-size versus unrecovered-value figure | `plots/make_sec4_figures.py` | `paper_results/figures/sec4_fleet_coverage.pdf` |
| Transfer expenditure versus unrecovered-value figure | `plots/make_sec4_figures.py` | `paper_results/figures/sec4_pareto.pdf` |
| Candidate-depot orbital sweep | `run_depot_sweep.py`, `oos/depot_legs.py` | `paper_results/depot_placement/depot_sweep_*.h5` |
| Lexicographic minimum-depot MILP | `run_depot_fronts_milp.py`, `oos/depot_fronts_milp.py` | `paper_results/depot_placement/depot_milp_verified_*` |
| Selected-depot map | `plot_depot_coverage_map.py`, `plots/map_style.py` | `paper_results/figures/depot_coverage_map_grouped.pdf` |

## Installation

Python 3.10 was used for the reported calculations.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Tested runtime dependency versions are recorded in `requirements.txt` and
`pyproject.toml`; the `test` extra also installs the pinned test runner.

## Reproduce figures from released results

The paired algorithm figures can be regenerated without rerunning either
optimiser:

```bash
mkdir -p reproduced_figures
python plots/make_sec4_figures.py \
  --result-dir paper_results/algorithm_fronts \
  --metrics paper_results/metrics/section4_metrics.csv \
  --outdir reproduced_figures \
  --trial 1
```

The depot-location figure can be regenerated from the released sweep and
optimal MILP records:

```bash
python plot_depot_coverage_map.py \
  --sweep-dir paper_results/depot_placement \
  --surface-dir paper_results/depot_placement \
  --surface-prefix depot_milp_verified \
  --B 2e9 \
  --starlink-loss 0.4 \
  --planet-loss 0.4 \
  --out reproduced_figures/depot_coverage_map_grouped.pdf
```

## Reproduce the numerical experiments

The paired algorithm benchmark uses 15 trials per scenario and 10,000 objective
evaluations per algorithm:

```bash
bash run_experiments.sh 15 10000
python analyse_fronts.py
```

The depot-placement demonstration uses 1,000 objective evaluations at each
candidate depot location and is run with:

```bash
bash run_all_depots.sh
```

The paper cases use a USD 2 billion budget, separate lost-value fractions of
0.4 for Starlink and Planet Labs, a per-depot cumulative transfer cap of
12,000 m/s, and zero relative MIP-gap tolerance. For example:

```bash
python run_depot_fronts_milp.py S1_repair \
  --budgets 2e9 \
  --starlink-loss-values 0.4 \
  --planet-loss-values 0.4 \
  --dv-cap 12000 \
  --solver scipy \
  --mip-gap 0 \
  --out outputs/depot_milp_verified_S1_repair
```

Repeat with `S2_refuel` and `S3_deorbit`.

## Tests

```bash
python -m pytest tests -q
```

The tests cover transfer mechanics, schedule feasibility, objective
definitions, the NSGA-II decoder, MDLS neighbourhoods, and archive dominance.

## Large generated inputs

`outputs/cost_table.h5`, `outputs/simulation.h5`, and
`outputs/depot_legs.h5` are generated working files excluded from Git. The
compact `paper_results/` directory contains the evidence and figures reported
in the manuscript. If the full cost table is deposited separately, add its DOI
here as a related dataset.

The local `waste/` directory contains superseded analyses and intermediate
data. It is ignored by Git and must not be included in a release.

## Citation and licence

Citation metadata are in `CITATION.cff`; add the Zenodo DOI after it is reserved
or issued. The code is released under the MIT License; see `LICENSE`.
