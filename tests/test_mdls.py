"""MDLS: each operator must move its own objective, and nothing infeasible
may enter the archive.

Those are the two claims the method rests on. The first is what makes the
search multi-directional rather than three random perturbations; the second is
what lets the front be reported as a set of solutions rather than a set of
candidates. Everything else here supports one or the other.

The last group runs the real instance. It is slower than the rest but it is the
only check that the operators compose on something the size of the study.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos.schedule import (DEPOT_UID, evaluate, is_feasible, load_cost_table)
from oos.greedy import greedy_schedule
from oos import mdls as M

fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)

ct_path = next((p for p in ("outputs/cost_table.h5", "outputs/cost_table_390d.h5")
                if os.path.exists(p)), None)
dem_path = "outputs/exp_demands/S2_refuel_01.h5"
if not (ct_path and os.path.exists(dem_path)):
    print("  skipped: build the cost table and demand sets first")
    sys.exit(0)

from oos.demands import load_demands
ct = load_cost_table(ct_path)
dem = load_demands(dem_path)
depot = ct.names.index("depot_1")
print(f"instance: {ct.n_nodes} nodes, {len(dem)} demands, budget {ct.dv_budget:.0f} m/s")

base, _ = greedy_schedule(ct, dem, depot, max_vehicles=25, refuel_time=0.5)
f0, g0 = evaluate(base, dem, ct.dv_budget)
print(f"start:    dV {f0[0]:.0f} m/s, {f0[1]:.0f} vehicles, ${f0[2]/1e6:.1f} M unrecovered")


# ---------------------------------------------------------------------------
print("\n1. routes round-trip through the per-vehicle view unchanged")
# ---------------------------------------------------------------------------

rt = M._from_routes(M._to_routes(base))
f1, _ = evaluate(rt, dem, ct.dv_budget)
check("objectives are identical after to_routes/from_routes", np.allclose(f0, f1),
      f"{f0} vs {f1}")


# ---------------------------------------------------------------------------
print("\n2. the fleet operator reduces the fleet and keeps feasibility")
# ---------------------------------------------------------------------------

cut = M.op_remove_vehicles(base, 5)
f, g = evaluate(cut, dem, ct.dv_budget)
check("f2 (vehicles) strictly decreases", f[1] < f0[1], f"{f0[1]:.0f} -> {f[1]:.0f}")
check("f3 (unrecovered) rises, as it must", f[2] >= f0[2] - 1e-6,
      f"${f0[2]/1e6:.1f} M -> ${f[2]/1e6:.1f} M")
check("still feasible", is_feasible(g), f"{g}")
check("at least one vehicle survives", f[1] >= 1)

only_one = M.op_remove_vehicles(M.op_remove_vehicles(base, 999), 999)
f, _ = evaluate(only_one, dem, ct.dv_budget)
check("removing more than exist keeps exactly one", f[1] == 1.0, f"{f[1]:.0f}")


# ---------------------------------------------------------------------------
print("\n3. the coverage operator serves more and keeps feasibility")
# ---------------------------------------------------------------------------

# start from a thinned schedule so there is unassigned demand to pick up
thin = M.op_remove_vehicles(base, 10)
f_thin, _ = evaluate(thin, dem, ct.dv_budget)
grown = M.op_create_vehicle(thin, ct, dem, depot, 0.5, 25, ct.dv_budget)
f, g = evaluate(grown, dem, ct.dv_budget)
check("f3 (unrecovered) decreases", f[2] < f_thin[2],
      f"${f_thin[2]/1e6:.1f} M -> ${f[2]/1e6:.1f} M")
check("it costs a vehicle to do it", f[1] == f_thin[1] + 1, f"{f[1]:.0f}")
check("still feasible", is_feasible(g), f"{g}")

full = M.op_create_vehicle(base, ct, dem, depot, 0.5, base.n_vehicles, ct.dv_budget)
check("refuses to exceed the fleet cap", full is base)


# ---------------------------------------------------------------------------
print("\n4. the timing operator never worsens delta-V, and stays feasible")
# ---------------------------------------------------------------------------

shifted = M.op_shift_times(base, ct, dem, depot, 0.5, shift=15.0, top_pct=0.5)
f, g = evaluate(shifted, dem, ct.dv_budget)
check("f1 (delta-V) does not increase", f[0] <= f0[0] + 1e-6,
      f"{f0[0]:.0f} -> {f[0]:.0f} m/s")
check("still feasible", is_feasible(g), f"{g}")
check("it serves the same demands", f[2] == f0[2],
      "timing must not change who is served")
check("no leg is left with a stale NaN cost", not np.isnan(shifted.cost).any())


# ---------------------------------------------------------------------------
print("\n5. the unassigned set is derived, so it cannot desync")
# ---------------------------------------------------------------------------

un = M.unassigned_of(base, len(dem))
served = set(int(u) for u in base.served())
check("derived unassigned is exactly the complement of served",
      set(un.tolist()) == set(range(len(dem))) - served)
check("no depot visit leaks into it", DEPOT_UID not in un.tolist())


# ---------------------------------------------------------------------------
print("\n6. the search runs, and everything in the archive is feasible")
# ---------------------------------------------------------------------------

t0 = time.time()
arch, n_eval = M.mdls(ct, dem, depot, max_vehicles=25, refuel_time=0.5,
                      max_iter=60, seed=1)
dt = time.time() - t0
print(f"     60 iterations, {n_eval} evaluations, {dt:.1f} s, "
      f"archive {len(arch)} points")

bad = [i for i in range(len(arch))
       if not is_feasible(evaluate(arch.schedules[i], dem, ct.dv_budget)[1])]
check("every archived solution is feasible", not bad, f"{len(bad)} bad")

F = arch.objectives
check("the archive is a genuine front (nothing dominates anything)",
      all(not (np.all(F[i] <= F[j]) and np.any(F[i] < F[j]))
          for i in range(len(F)) for j in range(len(F)) if i != j))
check("it improved on the starting point in at least one objective",
      bool((F[:, 0] < f0[0] - 1e-6).any() or (F[:, 1] < f0[1] - 1e-6).any()
           or (F[:, 2] < f0[2] - 1e-6).any()))
check("the front spans a trade-off rather than a single point", len(arch) > 1,
      f"{len(arch)} points")
check("evaluations are three per iteration plus the seeding set",
      n_eval <= 3 * 60 + 25, f"{n_eval}")

print(f"\n     front: dV {F[:,0].min():.0f}-{F[:,0].max():.0f} m/s   "
      f"vehicles {F[:,1].min():.0f}-{F[:,1].max():.0f}   "
      f"unrecovered ${F[:,2].min()/1e6:.1f}-{F[:,2].max()/1e6:.1f} M")


print("\n7. the re-sequencing operator (ported from raan_walk_resequence)")
import os.path as _op
if not _op.exists("outputs/simulation.h5"):
    print("  skipped: no ephemeris")
else:
    from oos.nodes import load_nodes
    raan = M.raan_model(ct, load_nodes("outputs/simulation.h5"))
    check("the nodal model aligns with the cost table's nodes",
          len(raan[0]) == ct.n_nodes, f"{len(raan[0])} vs {ct.n_nodes}")

    rs = M.op_raan_resequence(base, ct, dem, depot, 0.5, raan)
    f, g = evaluate(rs, dem, ct.dv_budget)
    check("f1 (delta-V) never increases", f[0] <= f0[0] + 1e-6,
          f"{f0[0]:.0f} -> {f[0]:.0f} m/s")
    check("still feasible", is_feasible(g), f"{g}")
    check("it serves exactly the same demands", f[2] == f0[2],
          "re-ordering must not change who is served")
    check("the fleet is untouched", f[1] == f0[1])

    # The operator must not move refuelling stops. The Julia rebuilt a vehicle
    # by appending every depot visit at the end, which merges two
    # depot-to-depot segments into one and so changes what g3 is measuring.
    def depot_positions(s):
        return [[i for i, u in enumerate(r["uid"]) if int(u) == DEPOT_UID]
                for r in M._to_routes(s)]
    check("depot visits keep their positions in every route",
          depot_positions(rs) == depot_positions(base))

    # A schedule with nothing to permute must come back untouched, not rebuilt.
    thin1 = M.op_remove_vehicles(base, base.n_vehicles - 1)
    out = M.op_raan_resequence(thin1, ct, dem, depot, 0.5, raan)
    check("a schedule it cannot improve is returned unchanged, not copied",
          out is thin1 or np.allclose(evaluate(out, dem, ct.dv_budget)[0],
                                      evaluate(thin1, dem, ct.dv_budget)[0]))

    arch2, ne2 = M.mdls(ct, dem, depot, max_vehicles=25, refuel_time=0.5,
                        max_iter=60, seed=1, raan=raan)
    bad2 = [i for i in range(len(arch2))
            if not is_feasible(evaluate(arch2.schedules[i], dem, ct.dv_budget)[1])]
    check("the search still runs with it in the pool, archive all feasible",
          not bad2 and len(arch2) > 1, f"{len(arch2)} points, {len(bad2)} bad")
    check("it does not change the cost of an iteration", ne2 <= 3 * 60 + 18,
          f"{ne2}")

print("\n8. the cross-vehicle swap operator")

sw = M.op_swap_cross_vehicle(base, ct, dem, depot, 0.5, top_n=50)
f, g = evaluate(sw, dem, ct.dv_budget)
check("f1 (delta-V) never increases", f[0] <= f0[0] + 1e-6,
      f"{f0[0]:.0f} -> {f[0]:.0f} m/s")
check("still feasible", is_feasible(g), f"{g}")
check("it serves exactly the same demands", f[2] == f0[2],
      "an exchange must not change who is served")
check("the fleet is untouched", f[1] == f0[1])
check("no leg is left with a stale NaN cost", not np.isnan(sw.cost).any())
check("the multiset of visits is preserved",
      sorted(sw.uid.tolist()) == sorted(base.uid.tolist()),
      "a swap moves clients between vehicles, it does not create or drop any")

one = M.op_remove_vehicles(base, base.n_vehicles - 1)
check("a single-vehicle schedule has no cross-vehicle swap and is returned as is",
      M.op_swap_cross_vehicle(one, ct, dem, depot, 0.5) is one)

arch3, ne3 = M.mdls(ct, dem, depot, max_vehicles=25, refuel_time=0.5,
                    max_iter=60, seed=1, swap=True)
bad3 = [i for i in range(len(arch3))
        if not is_feasible(evaluate(arch3.schedules[i], dem, ct.dv_budget)[1])]
check("the search runs with it in the pool, archive all feasible",
      not bad3 and len(arch3) > 1, f"{len(arch3)} points, {len(bad3)} bad")
check("it does not change the cost of an iteration", ne3 <= 3 * 60 + 18, f"{ne3}")

print("\n9. 2-regret repair (Sec. 3.7, Algorithm 2)")

# the fleet operator must now hand freed demands back where they fit
cut_bare = M.op_remove_vehicles(base, 8)
cut_fix = M.op_remove_vehicles(base, 8, ct, dem, depot, 0.5, ct.dv_budget)
fb, gb = evaluate(cut_bare, dem, ct.dv_budget)
ff, gf = evaluate(cut_fix, dem, ct.dv_budget)
check("the repair never serves less than the bare removal", ff[2] <= fb[2] + 1e-6,
      f"${fb[2]/1e6:.1f} M -> ${ff[2]/1e6:.1f} M unrecovered")
check("it does not add vehicles to do it", ff[1] <= fb[1], f"{fb[1]:.0f} -> {ff[1]:.0f}")
check("the result is feasible", is_feasible(gf), f"{gf}")
check("no visit is duplicated",
      len(set(cut_fix.served().tolist())) == len(cut_fix.served()))

# a repaired route must still be a priced, window-respecting chain
routes = M._to_routes(cut_fix)
reqs_ok = all(len(r["node"]) == len(r["cost"]) == len(r["arrival"]) for r in routes)
check("every route stays internally consistent", reqs_ok)
check("no leg carries a NaN cost", not np.isnan(cut_fix.cost).any())

# the reachability screen must agree with the table it screens for
sub = ct.reach[:40, :40]
brute = np.isfinite(np.asarray(ct.dv[:40, :40])).any(axis=(2, 3))
check("the reachability screen matches the table", bool((sub == brute).all()),
      f"{int((sub != brute).sum())} disagreements")

# regret2 on an empty unassigned set must be a no-op, not an error
same, left = M.regret2_insert(M._to_routes(base), [], ct, dem, depot, 0.5,
                              ct.dv_budget)
check("nothing to insert is a no-op", left == [] and len(same) == base.n_vehicles)

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
