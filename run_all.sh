#!/usr/bin/env bash
#
# Everything, in order, from the catalogues to the depot-selection MILP.
#
#   bash run_all.sh            run every stage that is ready
#   bash run_all.sh 3          run from stage 3 onwards
#   bash run_all.sh 3 3        run stage 3 only
#   XVAL=1 bash run_all.sh 9 9 run only the Section 3.2.6 cross-validation
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
python tests/test_propagate.py
python tests/test_edelbaum.py
python tests/test_guards.py
python tests/test_fastpath.py
python tests/test_schedule.py
fi

# ---------------------------------------------------------------------------
if run_stage 1; then
banner 1 "instance population from the cached CelesTrak catalogues (seconds)"
# 100 Starlink sampled proportionally across shells at <=2 per 5-degree RAAN
# plane, plus the full 124-object Planet Labs catalogue, plus the nominal depot.
# Shell counts should come out 49 / 3 / 29 / 19 / 0, matching Table 2.
python build_instance.py --out outputs/instance_population.csv
fi

# ---------------------------------------------------------------------------
if run_stage 2; then
banner 2 "propagate the ephemeris, RK4 + J2, 400 days (about 10 seconds)"
# dt = 10 s, not the 60 s the Julia used: over 400 days a 60 s step invents
# 11 km of semi-major-axis decay and 4.4 degrees of RAAN. Hourly records.
python -m oos.propagate \
    --population outputs/instance_population.csv \
    --out        outputs/simulation.h5 \
    --days 400 --dt 10 --write-every 360
fi

# ---------------------------------------------------------------------------
if run_stage 3; then
banner 3 "client-to-client cost table (about 18 minutes)"
# Solved once per ordered pair of constellation planes, then copied to every
# client in those planes with that client's own phasing. Exact, with no
# tolerance: plane membership is declared by build_instance.py, not
# rediscovered here.
#
# The Starlink half is essentially free -- 224 satellites on 14 planes is 295x
# fewer transfer solves. The runtime is all Planet Labs, which has no plane
# lattice (rideshare-deployed, spread over 352-604 km, RAAN-close objects 15-66
# km apart in altitude) so each of its 124 satellites is its own plane.
python -m oos.costtable \
    --sim outputs/simulation.h5 \
    --out outputs/cost_table.h5
fi

# ---------------------------------------------------------------------------
if run_stage 3.5 2>/dev/null || { [ "$FROM" -le 3 ] && [ "$TO" -ge 3 ]; }; then :; fi
if [ "${XVAL:-0}" = "1" ]; then
banner 3b "transfer-model cross-validation for Section 3.2.6 (tens of minutes)"
# Independent of the cost table, so it can run any time. The Lu NLP is the slow
# part. Use a large --n: on the current population most sampled geometries are
# not comparable, either because the servicer cannot afford them or because the
# Lu model is outside its domain at large inclination separation, so a small
# sample leaves too few paired points to quote a mean from. Section 3.2.6 should
# state the sample size it ends up with.
python crossvalidate_transfer_models.py \
    --population outputs/instance_population.csv \
    --n 500 \
    --out outputs/xval_spacevan.csv
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
python -m oos.depot_milp \
    --sweep outputs/depot_sweep.h5 \
    --out   outputs/depot_selection.csv \
    --fractions 0.0,0.05,0.10,0.20,0.30,0.40,0.50
fi

printf '\n\033[1mdone\033[0m\n'
