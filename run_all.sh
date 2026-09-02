#!/usr/bin/env bash
#
# Everything, in order, from the catalogues to the depot-selection MILP.
#
#   bash run_all.sh            run every stage that is ready
#   bash run_all.sh 3          run from stage 3 onwards
#   bash run_all.sh 3 3        run stage 3 only
#
# Stages 1-3 rebuild the instance from scratch and are the long pole. Stage 4
# is not runnable yet: it needs the Python MDLS, which is not ported. See the
# note under that stage.
#
# Run from the project root, the directory holding `oos/`.

set -euo pipefail

FROM=${1:-1}
TO=${2:-9}
run_stage() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }
banner() { printf '\n\033[1m== stage %s: %s ==\033[0m\n' "$1" "$2"; }

mkdir -p outputs

# Numba caches compiled kernels; keep that off any network or FUSE mount or
# every import recompiles from scratch.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/oos-numba}"
mkdir -p "$NUMBA_CACHE_DIR"

# ---------------------------------------------------------------------------
if run_stage 0; then
banner 0 "gates -- run these before trusting anything below"
python3 tests/test_propagate.py
python3 tests/test_edelbaum.py
python3 tests/test_guards.py
fi

# ---------------------------------------------------------------------------
if run_stage 1; then
banner 1 "instance population from the cached CelesTrak catalogues (seconds)"
# 100 Starlink sampled proportionally across shells at <=2 per 5-degree RAAN
# plane, plus the full 124-object Planet Labs catalogue, plus the nominal depot.
# Shell counts should come out 49 / 3 / 29 / 19 / 0, matching Table 2.
python3 build_instance.py --out outputs/instance_population.csv
fi

# ---------------------------------------------------------------------------
if run_stage 2; then
banner 2 "propagate the ephemeris, RK4 + J2, 400 days (about 10 seconds)"
# dt = 10 s, not the 60 s the Julia used: over 400 days a 60 s step invents
# 11 km of semi-major-axis decay and 4.4 degrees of RAAN. Hourly records.
python3 -m oos.propagate \
    --population outputs/instance_population.csv \
    --out        outputs/simulation.h5 \
    --days 400 --dt 10 --write-every 360
fi

# ---------------------------------------------------------------------------
if run_stage 3; then
banner 3 "client-to-client cost table (HOURS -- see note)"
# 225 nodes, 26 departure epochs x 26 times of flight on a 15-day grid.
# About 1.2 core-seconds per ordered pair, 50,400 ordered pairs: roughly
# 4 hours on 4 cores, 1.5-2 hours on 8. It only has to be built once -- the
# depot sweep in stage 4 rebuilds only the legs that touch the depot.
#
# To split it across sessions, build slices and merge:
#   python3 -m oos.costtable --from-node 0   --to-node 60  --out outputs/ct_000_060.h5
#   python3 -m oos.costtable --from-node 60  --to-node 120 --out outputs/ct_060_120.h5
#   ...
#   python3 -m oos.costtable --merge 'outputs/ct_*.h5' --out outputs/cost_table.h5
python3 -m oos.costtable \
    --sim outputs/simulation.h5 \
    --out outputs/cost_table.h5
fi

# ---------------------------------------------------------------------------
if run_stage 4; then
banner 4 "depot (a, i) location sweep"
cat <<'NOTE'
NOT RUNNABLE YET.

This stage needs `oos/mdls.py`, which does not exist. MDLS is 1616 lines of
Julia with nine neighbourhood operators and is the algorithmic core of the
paper, so it is the next chunk of work on its own rather than something to
hurry into this script.

What the stage will do, once that exists. For each (a, i) on the grid:

  1. oos.depot_legs.depot_legs(...) rebuilds the 448 ordered pairs that touch
     the depot, leaving the client block of stage 3 untouched. Minutes, not
     hours -- this is what makes a grid affordable.
  2. MDLS runs on that table and returns an archive.
  3. The maximum-coverage point of that archive becomes the location's column:
     which clients it serves, and the delta-V and fleet size it costs.

and writes outputs/depot_sweep.h5 with datasets `coverage` (n_loc x n_client,
boolean), `client_value`, `a_km`, `incl_deg`, `dv_m_s`, `fleet`, `client_names`.

Stage 5 reads exactly that file, so it can be checked against a hand-made one
before MDLS lands.
NOTE
fi

# ---------------------------------------------------------------------------
if run_stage 5; then
banner 5 "depot-selection MILP (gurobipy, seconds)"
# Fewest sampled depot locations holding unserved client value under a bound,
# each depot operated at the maximum-coverage point of its own Pareto front.
# Sweeping the bound gives the depots-versus-replacement curve.
python3 -m oos.depot_milp \
    --sweep outputs/depot_sweep.h5 \
    --out   outputs/depot_selection.csv \
    --fractions 0.0,0.05,0.10,0.20,0.30,0.40,0.50
fi

printf '\n\033[1mdone\033[0m\n'
