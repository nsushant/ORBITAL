"""Tests for oos/ga_problem.py (ScheduleProblem), the new NSGA-II wrapper.

Checks the specific things D21's redesign is supposed to guarantee:
  * a genome decodes into a Schedule using literal gene values -- no search,
    no repair;
  * a leg with no table entry (NaN cost) is still flown -- the demand counts
    as served (no f3 cost) and the violation shows up in g1, not f3;
  * a window violation shows up in g2;
  * a budget overspend shows up in g3, and a depot-visit gene between two
    legs resets it;
  * a depot gene with nothing scheduled after it is still flown, not
    silently skipped;
  * n_constr == 3 and out["G"] actually comes from oos.schedule.evaluate(),
    not a bespoke decoder-side feasibility check.

Needs pymoo installed -- run on the machine that has it, not the bridge
this file was written from.

The departure grid is a single point throughout. That is deliberate: with
more than one departure epoch, snap_departure's "nearest" rule depends on
the exact chained state (previous arrival + service time), which is easy to
get subtly wrong by hand when the fixture author cannot execute the test to
check it. A single-point grid makes every departure snap to index 0
unconditionally, so every cost-table override below can be checked by hand
without that risk. tof snapping only needs grid[k] >= requested, which is
enough to reason about directly.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos.schedule import CostTable, Demands, evaluate, is_feasible
from oos.ga_problem import ScheduleProblem

fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)

DEPOT, A, B = 0, 1, 2
N_NODES = 3
DEP_GRID = np.array([0.0])          # single point: every departure snaps to index 0
TOF_GRID = np.array([10.0, 20.0, 50.0])  # index 0 covers requested tof in (0, 10];
                                         # the 50 d point only lifts the arrival
                                         # ceiling clear of T_horizon (see group 8)
DV_BUDGET = 100.0


def make_cost_table(dv_overrides=None):
    """All-NaN dv/phasing except the entries the test explicitly sets."""
    shape = (N_NODES, N_NODES, len(DEP_GRID), len(TOF_GRID))
    dv = np.full(shape, np.nan)
    ph = np.zeros(shape)
    for (i, j, p, q), val in (dv_overrides or {}).items():
        dv[i, j, p, q] = val
    return CostTable(
        dv=dv, phasing=ph, names=["depot", "A", "B"],
        dep_days=DEP_GRID, tof_days=TOF_GRID,
        plane_of_node=np.array([-1, 0, 1]), dv_budget=DV_BUDGET,
    )


def make_demands(node, release=None, deadline=None, service=None, value=None):
    n = len(node)
    return Demands(
        node=np.array(node, dtype=np.int64),
        release=np.array(release if release is not None else [0.0] * n),
        deadline=np.array(deadline if deadline is not None else [40.0] * n),
        service=np.array(service if service is not None else [1.0] * n),
        value=np.array(value if value is not None else [1000.0] * n),
    )


def eval1(prob, x):
    out = {}
    prob._evaluate(np.array([x]), out)
    return out["F"][0], out["G"][0]


# ---------------------------------------------------------------------------
print("1. out[\"F\"] / out[\"G\"] must literally be oos.schedule.evaluate()'s output")
# ---------------------------------------------------------------------------

ct = make_cost_table({(DEPOT, A, 0, 0): 5.0})   # tof<=10 -> grid idx 0
dem = make_demands([A], deadline=[40.0])
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
check("n_constr == 3", prob.n_constr == 3)
check("n_obj == 3", prob.n_obj == 3)

x = np.zeros(2)                        # x[0]: demand gene, x[1]: depot gene
x[0] = 1 + 10.0 / prob.T_horizon        # vehicle 1, arrival == 10.0
f_out, g_out = eval1(prob, x)
sched = prob.decode(x)
f_direct, g_direct = evaluate(sched, dem, DV_BUDGET)
check("out['F'] matches a direct evaluate() call", np.allclose(f_out, f_direct))
check("out['G'] matches a direct evaluate() call", np.allclose(g_out, g_direct))


# ---------------------------------------------------------------------------
print("2. an unserved demand costs f3 and nothing else")
# ---------------------------------------------------------------------------

ct = make_cost_table({(DEPOT, A, 0, 0): 5.0})
dem = make_demands([A], value=[777.0])
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
x = np.zeros(2)   # vehicle 0 == unassigned
f, g = eval1(prob, x)
check("f1 (delta-V) is zero", f[0] == 0.0)
check("f2 (active vehicles) is zero", f[1] == 0.0)
check("f3 is the full recovery potential", f[2] == 777.0)
check("nothing is violated", is_feasible(g))


# ---------------------------------------------------------------------------
print("3. g1: a leg with no table entry is still flown, not dropped")
# ---------------------------------------------------------------------------

ct = make_cost_table({})   # every entry NaN: no feasible leg exists at all
dem = make_demands([A], value=[500.0], deadline=[40.0])
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
x = np.zeros(2)
x[0] = 1 + 10.0 / prob.T_horizon
f, g = eval1(prob, x)
sched = prob.decode(x)
check("the visit is still recorded, not skipped", len(sched.node) == 1)
check("its cost is NaN", np.isnan(sched.cost[0]))
check("a demand the vehicle attempted to visit is not 'unrecovered'", f[2] == 0.0)
check("g1 (missing leg) catches the impossible leg", g[0] >= 1.0)


# ---------------------------------------------------------------------------
print("4. g2: window violation")
# ---------------------------------------------------------------------------

# T_horizon is set from the *max* deadline across all demands, so a demand
# with a smaller deadline than that max can genuinely be pushed late --
# demand 0's own deadline (5) is below demand 1's (40), which is what makes
# T_horizon = 39 large enough to construct a late arrival for demand 0.
ct = make_cost_table({(DEPOT, A, 0, 0): 5.0})
dem = make_demands([A, B], deadline=[5.0, 40.0], service=[1.0, 1.0])
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
check("T_horizon == 39.0 (below the table's ceiling, so unclamped)",
      prob.T_horizon == 39.0)

x = np.zeros(4)                          # 2 demand genes + 2 depot genes
x[0] = 1 + 10.0 / prob.T_horizon         # demand 0 (A): vehicle 1, arrival == 10.0
# x[1] left at 0: demand 1 (B) unassigned, isolates the check to demand 0
f, g = eval1(prob, x)
check("g1 is zero (the leg itself is feasible)", g[0] == 0.0)
check("g2 > 0 (arrival 10 + service 1 = 11 > deadline 5)", g[1] > 0.0)


# ---------------------------------------------------------------------------
print("5. g3: budget overspend, and a depot visit resets it")
# ---------------------------------------------------------------------------

# Every leg here departs at grid epoch 0 (the grid has only one), so each one
# must still *land* after the vehicle set out or leg() refuses it as time
# travel -- which is what the chronology guard in oos/schedule.py enforces.
# That is why the times of flight climb 10 -> 20 -> 50 down the route rather
# than repeating: the vehicle's clock advances while its departure epoch
# cannot, so each successive leg has to be longer than the last to stay
# physical. An earlier version of this fixture chained two 10-day legs and
# only passed because leg() was allowing the second one to arrive before it
# left (F23).
ct = make_cost_table({
    (DEPOT, A, 0, 0): 60.0,     # depot -> A,  tof 10, arrives day 10
    (A, B, 0, 2):     60.0,     # A -> B,      tof 50, arrives day 50
    (A, DEPOT, 0, 1):  1.0,     # A -> depot,  tof 20, arrives day 20
    (DEPOT, B, 0, 2):  1.0,     # depot -> B,  tof 50, arrives day 50
})
dem = make_demands([A, B], deadline=[200.0, 200.0])
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=1, refuel_time=0.5)
check("T_horizon clamped to the table's 50 d ceiling", prob.T_horizon == 50.0,
      f"{prob.T_horizon}")

x = np.zeros(4)                     # 2 demand genes + 2 depot genes
x[0] = 1 + 10.0 / prob.T_horizon    # A: vehicle 1, requested arrival 10
x[1] = 1 + 45.0 / prob.T_horizon    # B: vehicle 1, requested arrival 45 (sorts after A)
f, g = eval1(prob, x)
check("both legs are real (no missing-table entry)", g[0] == 0.0, f"g1 = {g[0]}")
check("two 60-cost legs back to back overspend the 100 budget", g[2] > 0.0,
      f"g3 = {g[2]}")

x2 = x.copy()
x2[2] = 1 + 22.0 / prob.T_horizon   # depot gene 0: vehicle 1, between A and B
f2, g2 = eval1(prob, x2)
check("a depot visit between the two legs resets the budget", g2[2] == 0.0,
      f"g3 = {g2[2]}")
check("and the rerouted legs are still real", g2[0] == 0.0, f"g1 = {g2[0]}")


# ---------------------------------------------------------------------------
print("6. a 'wasted' depot gene (nothing scheduled after it) is still flown")
# ---------------------------------------------------------------------------

ct = make_cost_table({(DEPOT, DEPOT, 0, 0): 0.0})
dem = make_demands([A], deadline=[40.0])
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
x = np.zeros(2)
x[1] = 1 + 5.0 / prob.T_horizon   # depot gene 0: vehicle 1, no demand assigned to it
sched = prob.decode(x)
check("the depot-only vehicle's visit is recorded", len(sched.node) == 1)
check("it is tagged as a depot visit", sched.uid[0] == -1)
check("its cost is recorded (not skipped)", sched.cost[0] == 0.0)


# ---------------------------------------------------------------------------
print("7. T_horizon guard")
# ---------------------------------------------------------------------------

ct = make_cost_table({})
dem = make_demands([A], deadline=[5.0], service=[5.0])   # deadline == service
raised = False
try:
    ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
except ValueError:
    raised = True
check("a degenerate horizon (deadline == service) raises ValueError", raised)


# ---------------------------------------------------------------------------
print("8. T_horizon is clamped to the arrival the cost table can express")
# ---------------------------------------------------------------------------

# A gene mapping past dep_days[-1] + tof_days[-1] is dead range: every leg into
# it returns g1 whatever the rest of the genome does. With five-year release
# epochs the Sec. 3.8 formula overshoots badly -- S3 was wasting 27.7% of its
# range this way on the 390-day table -- so the encoding clamps to the ceiling.
ct = make_cost_table({})
ceiling = float(DEP_GRID[-1] + TOF_GRID[-1])
dem = make_demands([A], deadline=[5000.0], service=[1.0])   # far past the ceiling
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
check("clamped to the table's arrival ceiling", prob.T_horizon == ceiling,
      f"{prob.T_horizon:.0f} d, ceiling {ceiling:.0f} d, deadline 5000 d")
check("no gene can ask for an arrival the table cannot answer",
      prob.T_horizon <= ceiling)

dem = make_demands([A], deadline=[40.0], service=[1.0])     # inside the ceiling
prob = ScheduleProblem(ct, dem, depot_node=DEPOT, max_vehicles=2, refuel_time=0.5)
check("left alone when the deadline is already inside it", prob.T_horizon == 39.0)


print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
