"""The constructive heuristic must produce a *feasible* schedule (Sec. 3.6).

That is the claim the paper makes for it and the only reason it is worth
having: MDLS starts from it, and the GA seeds its population from it. If it
can hand back a schedule that violates a window or the budget, both inherit
the violation.

So the checks here are mostly one check applied in different corners: run the
heuristic, evaluate the result through the same `oos.schedule.evaluate()` the
algorithms use, and require `is_feasible()`. Everything else -- that it places
what it can, leaves the rest unassigned rather than forcing them, refuels
instead of overspending -- is checked around that.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos.schedule import (CostTable, DEPOT_UID, Demands, evaluate, is_feasible,
                          load_cost_table)
from oos.greedy import greedy_schedule

fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)

DEPOT, A, B = 0, 1, 2
DEP_GRID = np.array([0.0, 20.0, 40.0])
TOF_GRID = np.array([10.0, 20.0])
BUDGET = 100.0


def make_ct(overrides):
    shape = (3, 3, len(DEP_GRID), len(TOF_GRID))
    dv = np.full(shape, np.nan)
    ph = np.zeros(shape)
    for k, v in overrides.items():
        dv[k] = v
    return CostTable(dv=dv, phasing=ph, names=["depot", "A", "B"],
                     dep_days=DEP_GRID, tof_days=TOF_GRID,
                     plane_of_node=np.array([-1, 0, 1]), dv_budget=BUDGET)


def make_dem(node, release=None, deadline=None, service=None, value=None):
    n = len(node)
    return Demands(
        node=np.array(node, dtype=np.int64),
        release=np.array(release if release is not None else [0.0] * n),
        deadline=np.array(deadline if deadline is not None else [200.0] * n),
        service=np.array(service if service is not None else [1.0] * n),
        value=np.array(value if value is not None else [1000.0] * n))


# ---------------------------------------------------------------------------
print("1. a reachable demand gets placed, and the result is feasible")
# ---------------------------------------------------------------------------

ct = make_ct({(DEPOT, A, 0, 0): 5.0, (A, DEPOT, 0, 0): 5.0,
              (A, DEPOT, 1, 0): 5.0, (A, DEPOT, 2, 0): 5.0})
dem = make_dem([A])
sched, unassigned = greedy_schedule(ct, dem, DEPOT, max_vehicles=3, refuel_time=0.5)
f, g = evaluate(sched, dem, BUDGET)
check("the demand is served", len(unassigned) == 0)
check("the schedule is feasible", is_feasible(g), f"violations {g}")
check("one vehicle is active", f[1] == 1.0)
check("nothing is left unrecovered", f[2] == 0.0)


# ---------------------------------------------------------------------------
print("2. an unreachable demand is left unassigned, not forced")
# ---------------------------------------------------------------------------

ct = make_ct({})            # no leg exists anywhere
dem = make_dem([A], value=[400.0])
sched, unassigned = greedy_schedule(ct, dem, DEPOT, max_vehicles=3, refuel_time=0.5)
f, g = evaluate(sched, dem, BUDGET)
check("the demand is unassigned", list(unassigned) == [0])
check("no visits were invented", len(sched.node) == 0)
check("it costs f3, not a violation", f[2] == 400.0 and is_feasible(g))


# ---------------------------------------------------------------------------
print("3. a demand whose deadline cannot be met is left alone")
# ---------------------------------------------------------------------------

ct = make_ct({(DEPOT, A, 0, 0): 5.0})
dem = make_dem([A], deadline=[5.0])       # arrival is 10, so 10 + 1 > 5
sched, unassigned = greedy_schedule(ct, dem, DEPOT, max_vehicles=2, refuel_time=0.5)
f, g = evaluate(sched, dem, BUDGET)
check("left unassigned rather than served late", list(unassigned) == [0])
check("still feasible", is_feasible(g), f"violations {g}")


# ---------------------------------------------------------------------------
print("4. a demand released later is reached by flying longer, not arriving early")
# ---------------------------------------------------------------------------

# Departing at epoch 0: the short flight (tof 10) arrives at 10, before the
# demand exists; the long one (tof 20) arrives at 20, after it is released.
# Waiting has to be bought with flight time rather than with a later
# departure, because the decoder always leaves the moment the vehicle is free
# -- so that is the only form of waiting the encoding can express, and the
# only one the heuristic may use. Note it takes the dearer leg (8 over 5)
# because the cheap one is out of window.
ct = make_ct({(DEPOT, A, 0, 0): 5.0, (DEPOT, A, 0, 1): 8.0})
dem = make_dem([A], release=[15.0], deadline=[200.0])
sched, unassigned = greedy_schedule(ct, dem, DEPOT, max_vehicles=2, refuel_time=0.5)
f, g = evaluate(sched, dem, BUDGET)
check("the demand is served", len(unassigned) == 0)
check("arrival respects the release time", len(sched.arrival) and sched.arrival[0] >= 15.0,
      f"arrived {sched.arrival[0] if len(sched.arrival) else float('nan'):.1f}")
check("it took the longer, dearer leg to stay in window",
      len(sched.cost) and sched.cost[0] == 8.0)
check("feasible", is_feasible(g), f"violations {g}")


# ---------------------------------------------------------------------------
print("5. the budget is never exceeded -- it refuels instead")
# ---------------------------------------------------------------------------

# depot->A and A->B each cost 60: 120 > 100, so the run must break at the depot.
ct = make_ct({(DEPOT, A, 0, 0): 60.0, (A, B, 0, 0): 60.0,
              (A, DEPOT, 0, 0): 10.0, (A, DEPOT, 1, 0): 10.0, (A, DEPOT, 2, 0): 10.0,
              (DEPOT, B, 0, 0): 60.0, (DEPOT, B, 1, 0): 60.0, (DEPOT, B, 2, 0): 60.0,
              (B, DEPOT, 0, 0): 10.0, (B, DEPOT, 1, 0): 10.0, (B, DEPOT, 2, 0): 10.0})
dem = make_dem([A, B])
sched, unassigned = greedy_schedule(ct, dem, DEPOT, max_vehicles=2, refuel_time=0.5)
f, g = evaluate(sched, dem, BUDGET)
check("no budget overspend", g[2] == 0.0, f"g3 = {g[2]}")
check("feasible overall", is_feasible(g), f"violations {g}")
check("a refuelling visit was inserted or a second vehicle used",
      int((sched.uid == DEPOT_UID).sum()) > 0 or f[1] > 1.0)


# ---------------------------------------------------------------------------
print("6. against the real instance")
# ---------------------------------------------------------------------------

# rebuild_long_horizon.sh renames the old table while it builds the new one,
# so accept either -- this check is about the heuristic, not about which grid.
ct_path = next((p for p in ("outputs/cost_table.h5", "outputs/cost_table_390d.h5")
                if os.path.exists(p)), None)
dem_path = "outputs/exp_demands/S2_refuel_01.h5"
if ct_path and os.path.exists(dem_path):
    from oos.demands import load_demands
    print(f"     table: {ct_path}")
    ct = load_cost_table(ct_path)
    dem = load_demands(dem_path)
    depot = ct.names.index("depot_1")
    sched, unassigned = greedy_schedule(ct, dem, depot, max_vehicles=25,
                                        refuel_time=0.5)
    f, g = evaluate(sched, dem, ct.dv_budget)
    served = len(dem) - len(unassigned)
    print(f"     {served} of {len(dem)} demands placed on {int(f[1])} vehicles, "
          f"{f[0]:.0f} m/s, ${f[2]/1e6:.1f} M unrecovered")
    check("the real-instance schedule is feasible", is_feasible(g),
          f"violations {g}")
    check("it places at least one demand", served > 0)
    check("every placed demand is inside its window",
          bool(np.all(sched.arrival[sched.uid != DEPOT_UID] >=
                      dem.release[sched.uid[sched.uid != DEPOT_UID]] - 1e-9)))
else:
    print("     skipped: build the cost table and demand sets first")


print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
