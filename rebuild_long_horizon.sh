#!/bin/bash
# rebuild_long_horizon.sh — rebuild the cost table on the five-year departure
# axis (D25) and regenerate the demand sets against it.
#
# Why this exists: the departure axis is an absolute epoch, so a leg leaving on
# day 1400 needs a table entry at day 1400. The old table stopped at day 390,
# which made a five-year demand arrival process meaningless -- every request
# released past day 390 was unservable by construction, and the three scenarios
# collapsed to within a point of each other on servable fraction.
#
# Run from basic_project/:
#   bash rebuild_long_horizon.sh
#
# Budget about 70-80 minutes, almost all of it stage 3. Nothing here needs
# supervision; it is safe to leave running.

set -e
cd "$(dirname "$0")"

echo "=== 0. gates (fast) ==="
python tests/test_schedule.py
python tests/test_archive.py
python tests/test_ga_problem.py
python tests/test_greedy.py

echo ""
echo "=== 1. cost table, 5-year departures x 2-year times of flight ==="
echo "    73 departure epochs (15-1800 d) x 37 times of flight (15-720 d)"
echo "    ~2.6 GB, ~70 minutes. Old table is kept as a backup first."
if [ -f outputs/cost_table.h5 ]; then
    mv -f outputs/cost_table.h5 outputs/cost_table_390d.h5
    echo "    previous table saved as outputs/cost_table_390d.h5"
fi
time python -m oos.costtable \
    --sim outputs/simulation.h5 \
    --out outputs/cost_table.h5

echo ""
echo "=== 2. reachability against the new table ==="
python - <<'PY'
import h5py, numpy as np
with h5py.File("outputs/cost_table.h5", "r") as f:
    names = [n.decode() if isinstance(n, bytes) else n for n in f["names"][:]]
    dep, tof = f["departure_days"][:], f["tof_days"][:]
    budget = float(f.attrs["dv_budget_m_s"])
    depot = names.index("depot_1")
    row = f["dv"][depot, :, :, :]
    ph  = f["phasing_days"][depot, :, :, :]
arrive = dep[:, None] + tof[None, :] + ph
aff = (~np.isnan(row)) & (row <= budget)
earliest = np.where(aff, arrive, np.inf).min(axis=(1, 2))[:depot]
print(f"clients ever reachable within {budget:.0f} m/s: "
      f"{int(np.isfinite(earliest).sum())} / {depot}")
print("clients whose earliest affordable arrival <= D:")
for D in (120, 180, 240, 365, 730, 1095, 1826):
    print(f"   D = {D:4d} d : {int((earliest <= D).sum()):3d}")
PY

echo ""
echo "=== 3. demand sets over a five-year arrival process ==="
python generate_demands.py --arrival-horizon 1826 --n-demands 400 --trials 5

echo ""
echo "=== 4. warm-start round trip on the new table ==="
# The outstanding prediction from Round 12: a greedy schedule survived encoding
# into a genome and back only for vehicles operating inside the old 390-day
# departure grid. Past it, snap_departure clamped, the encoded request
# overshot, and legs came back NaN. With departures now running five years a
# vehicle's clock should stay inside the grid, so this should come back
# feasible. If it does not, the encoding has a second problem behind the first
# and the warm start is not ready to seed anything.
python - <<'PY'
import numpy as np
from oos.schedule import load_cost_table, evaluate, is_feasible
from oos.demands import load_demands
from oos.greedy import greedy_schedule, encode_for_ga
from oos.ga_problem import ScheduleProblem

ct = load_cost_table("outputs/cost_table.h5")
depot = ct.names.index("depot_1")
for scen in ("S1_repair", "S2_refuel", "S3_deorbit"):
    dem = load_demands(f"outputs/exp_demands/{scen}_01.h5")
    sched, un = greedy_schedule(ct, dem, depot, max_vehicles=25, refuel_time=0.5)
    f0, g0 = evaluate(sched, dem, ct.dv_budget)
    prob = ScheduleProblem(ct, dem, depot_node=depot, max_vehicles=25,
                           refuel_time=0.5)
    x = encode_for_ga(sched, dem, prob)
    out = {}
    prob._evaluate(np.array([x]), out)
    f1, g1 = out["F"][0], out["G"][0]
    print(f"  {scen:11s} placed {len(dem)-len(un):3d}/{len(dem)}   "
          f"greedy {f0[0]:8.0f} m/s feasible={is_feasible(g0)}   "
          f"round-trip {f1[0]:8.0f} m/s feasible={is_feasible(g1)}  "
          f"g={np.round(g1, 2)}")
PY

echo ""
echo "=== done ==="
echo "Compare the servable fractions above against the 390-day table:"
echo "  S1 12.8%   S2 18.2%   S3 27.1%   (old table, 390-day arrivals)"
echo "  S1  3.9%   S2  4.4%   S3  4.8%   (old table, 5-year arrivals — degenerate)"
echo "If the new numbers are not comfortably above the second row, the extra"
echo "departure coverage did not buy what it was supposed to and is worth a look"
echo "before any algorithm runs on top of it."
