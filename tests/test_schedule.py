"""Pin the evaluator's contract: what is an objective and what is a violation.

These are the properties the published comparison turned out not to have. Each
check corresponds to a finding: F18 (f3 silently became service time), F19
(NSGA-III had no constraints and repaired instead), F7 (the two stacks snapped
epochs differently), D19 and D20.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos.schedule import (DEPOT_UID, Demands, Schedule, evaluate, is_feasible,
                          snap_departure, snap_tof)

fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)

grid = np.arange(15.0, 400.0, 15.0)

print("1. epoch snapping (D20): departure to the nearest, flight time up")
check("departure 20 d -> 15", snap_departure(grid, 20.0) == 0)
check("departure 24 d -> 30", snap_departure(grid, 24.0) == 1)
check("departure below the grid clamps", snap_departure(grid, 1.0) == 0)
check("departure above the grid clamps", snap_departure(grid, 9e9) == len(grid) - 1)
check("flight 16 d rounds UP to 30", grid[snap_tof(grid, 16.0)] == 30.0)
check("flight 30 d stays at 30", grid[snap_tof(grid, 30.0)] == 30.0)
check("flight beyond the horizon is refused", snap_tof(grid, 1e9) == -1)

print("\n2. an unserved demand costs f3 and violates nothing (D19)")
dem = Demands(node=np.array([1, 2]), release=np.zeros(2),
              deadline=np.array([400.0, 400.0]), service=np.array([2.0, 2.0]),
              value=np.array([1.0e6, 2.5e6]))
empty = Schedule(np.array([0, 0]), np.array([], int), np.array([], int),
                 np.array([]), np.array([]), np.array([]))
f, g = evaluate(empty, dem, 500.0)
check("serving nobody costs the whole portfolio", f[2] == 3.5e6, f"f3 = ${f[2]:,.0f}")
check("serving nobody uses no vehicles", f[1] == 0)
check("serving nobody violates nothing", is_feasible(g))

print("\n3. f3 is money and never falls back to service time (F18, D18)")
try:
    Demands(node=np.array([1]), release=np.zeros(1), deadline=np.array([1.0]),
            service=np.array([2.0]), value=np.array([np.nan]))
    check("a demand with no value is refused", False)
except ValueError:
    check("a demand with no value is refused", True)

print("\n4. the three violations are detected, and only they (D19)")
one = Schedule(np.array([0, 1]), np.array([1]), np.array([0]),
               np.array([100.0]), np.array([102.0]), np.array([300.0]))
f, g = evaluate(one, dem, 500.0)
check("a clean single visit is feasible", is_feasible(g))
check("  and its value is recovered", f[2] == 2.5e6)
check("  and it activates one vehicle", f[1] == 1)

late = Schedule(np.array([0, 1]), np.array([1]), np.array([0]),
                np.array([450.0]), np.array([452.0]), np.array([300.0]))
_, g = evaluate(late, dem, 500.0)
check("arriving 50 d past the deadline is a violation", abs(g[1] - 50.0) < 1e-9)

early = Schedule(np.array([0, 1]), np.array([1]), np.array([0]),
                 np.array([-10.0]), np.array([-8.0]), np.array([300.0]))
_, g = evaluate(early, dem, 500.0)
check("arriving before release is a violation", abs(g[1] - 10.0) < 1e-9)

over = Schedule(np.array([0, 2]), np.array([1, 2]), np.array([0, 1]),
                np.array([100.0, 200.0]), np.array([102.0, 202.0]),
                np.array([400.0, 300.0]))
_, g = evaluate(over, dem, 500.0)
check("spending 700 of a 500 m/s budget overspends by 200",
      abs(g[2] - 200.0) < 1e-9)

# The same two legs either side of a depot visit are affordable, because
# refuelling resets the budget (Eq. fuel-capacity).
refuel = Schedule(np.array([0, 3]), np.array([1, 0, 2]),
                  np.array([0, DEPOT_UID, 1]),
                  np.array([100.0, 150.0, 200.0]),
                  np.array([102.0, 152.0, 202.0]),
                  np.array([400.0, 50.0, 300.0]))
_, g = evaluate(refuel, dem, 500.0)
check("a depot visit resets the budget", abs(g[2]) < 1e-9)

missing = Schedule(np.array([0, 1]), np.array([1]), np.array([0]),
                   np.array([100.0]), np.array([102.0]), np.array([np.nan]))
f, g = evaluate(missing, dem, 500.0)
check("a leg absent from the table is a violation, not a cost", g[0] == 1.0)
check("  and it does not enter f1", f[0] == 0.0,
      "a finite penalty summed into delta-V is how NSGA-III-T reported 2.8e8 m/s")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
