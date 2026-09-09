"""run_depot_sweep.py — sweep the depot over an (altitude, inclination) grid.

For every grid location, rebuild the depot legs of the cost table, run the
tuned MDLS on that table, and keep the location's column (D13a) *and* its full
Pareto front. The maximum-coverage column feeds the Stage-5 geometric bound
(`oos.depot_milp`), while the flattened per-point records let the economic
facility-location MILP (`oos.depot_fronts_milp`) select *which operating point*
each chosen depot runs, not merely that the depot opens. Writes one H5 per
scenario:

    coverage     (n_loc, n_client) bool  — client fully served at max coverage
    client_value (n_client,) $           — total request value of each client
    a_km         (n_loc,)  semi-major axis of the depot orbit
    incl_deg     (n_loc,)  inclination of the depot orbit
    dv_m_s       (n_loc,)  delta-V spent at the max-coverage point
    fleet        (n_loc,)  active vehicles at the max-coverage point

Per-point front records (flattened across locations, so fronts of any length):

    loc_npts          (n_loc,)   int — number of front points per location
    loc_offset        (n_loc,)   int — start index of each location's points
    points_dv_m_s     (n_pts,)   float — total delta-V of each front point
    points_fleet      (n_pts,)   int — active vehicles of each front point
    points_served_value (n_pts,) float — $ of demand value served at the point
    points_covered    (n_pts, n_client) bool — clients fully served (gzip)

A client counts as covered only when *all* of its requests are served, so the
MILP's sum of value_c (1 - z_c) is a conservative bound on the front's true
unrecovered f3. `served_request_value` stores the true served value
(total - f3), which is what the maps plot.

A client counts as covered only when *all* of its requests are served at the
max-coverage point, so the MILP's sum of value_c (1 - z_c) is a conservative
bound on the front's true unrecovered f3. `served_request_value` stores the
true served value (total - f3), which is what the maps plot.

Grid. Reachability is the constraint on where this sweep is worth doing: the
study clients sit in three shells at exactly 53 / 69 / 97 deg, a 500 m/s
budget can afford almost nothing beyond a ~1-2 deg plane change, and the
defaults therefore scan 53-97 deg anchored at 53 so every shell is hit
(53, 55, ..., 97). An off-shell grid is uniformly zero-coverage: an
inclination step that does not land on a shell, or a range that does not
contain one, produces an empty map. Semimajor axis = R_E + altitude.

Usage
-----
    python run_depot_sweep.py S1_repair              # full 17 x 26 grid
    python run_depot_sweep.py S1_repair --smoke      # 3 x 3 subset, minutes
    python run_depot_sweep.py S2_refuel --jobs 4     # fork-parallel workers
    python run_depot_sweep.py S1_repair --legs-cache outputs/depot_legs.h5 \
        --jobs 4                                     # build-or-reuse legs cache

The depot legs do not depend on the scenario, so `--legs-cache` computes the
`(a, i)` grid's legs once and every scenario sweep reuses them; unreachable
locations (no client within the delta-V budget) are skipped entirely and
stored with no rows at all.

Reads:  outputs/cost_table.h5, outputs/simulation.h5,
        outputs/exp_demands/{scenario}_{trial}.h5, outputs/instance_population.csv,
        tune/best-mdls.txt (optional; defaults if absent)
        optional --legs-cache path (built on first use)
Writes: outputs/depot_sweep_{scenario}.h5
"""

import argparse
import os
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("key", help="scenario, e.g. S1_repair")
parser.add_argument("--trial", type=int, default=1)
parser.add_argument("--cost-table", default="outputs/cost_table.h5")
parser.add_argument("--sim", default="outputs/simulation.h5")
parser.add_argument("--demand-dir", default="outputs/exp_demands")
parser.add_argument("--population", default="outputs/instance_population.csv")
parser.add_argument("--mdls-params", default="tune/best-mdls.txt")
parser.add_argument("--out-dir", default="outputs")
parser.add_argument("--dv-budget", type=float, default=None)
parser.add_argument("--max-vehicles", type=int, default=25)
parser.add_argument("--refuel-time", type=float, default=0.5)
parser.add_argument("--n-eval", type=int, default=10_000)
parser.add_argument("--alt-lo", type=float, default=400.0)
parser.add_argument("--alt-hi", type=float, default=1200.0)
parser.add_argument("--alt-step", type=float, default=50.0)
parser.add_argument("--incl-lo", type=float, default=53.0)
parser.add_argument("--incl-hi", type=float, default=97.0)
parser.add_argument("--incl-step", type=float, default=2.0)
parser.add_argument("--jobs", type=int, default=1,
                    help="fork-parallel workers; cap NUMBA threads when > 1")
parser.add_argument("--legs-cache", default=None,
                    help="path to the scenario-independent depot-legs cache H5; "
                         "built on first use, reused across S1/S2/S3")
parser.add_argument("--smoke", action="store_true",
                    help="3 x 3 subset of the grid (endpoint + midpoint each axis)")
args = parser.parse_args()

scenario, trial = args.key, args.trial

from oos.constants import R_E
from oos.demands import load_demands, load_population
from oos.depot_legs import depot_legs
from oos.greedy import greedy_restarts
from oos.mdls import mdls
from oos.nodes import load_nodes
from oos.schedule import load_cost_table

# Shared read-only state. With a fork pool the children inherit these arrays
# and only the handful of depot row/column pages they overwrite are ever
# copied (copy-on-write), so --jobs N does not multiply the 2.6 GB table N
# times in memory.
_G = {}


def mdls_params(path):
    """Tuned --shift/--remove-*/--add-*/--swap line, comments stripped."""
    if not path or not os.path.exists(path):
        return {}
    clean = [line.split("#")[0].strip() for line in open(path) if line.strip()]
    tok = " ".join(clean).split()
    out, i = {}, 0
    while i < len(tok):
        if not tok[i].startswith("--"):
            i += 1
            continue
        name = tok[i][2:].replace("-", "_")
        if i + 1 < len(tok) and not tok[i + 1].startswith("--"):
            out[name], i = tok[i + 1], i + 2
        else:
            out[name], i = True, i + 1
    return out


def grid_from_args():
    alts = np.arange(args.alt_lo, args.alt_hi + 1e-9, args.alt_step)
    incls = np.arange(args.incl_lo, args.incl_hi + 1e-9, args.incl_step)
    if not _G.get("smoke"):
        return np.asarray(alts), np.asarray(incls)
    pick = lambda v: v[[0, len(v) // 2, -1]] if len(v) > 1 else v
    return pick(np.asarray(alts)), pick(np.asarray(incls))


def _covered_mask(served):
    """Clients whose *every* request is served at a given front point.

    A client with some requests served and others not is *not* covered: the
    facility-location MILP carries a client-level binary z_c, so a partially
    served client cannot be counted as recovered without overstating the
    portfolio. Demanding all-or-nothing keeps the MILP's recovered-value sum an
    exact function of the front point (and conservative where f3 allows
    partials)."""
    n_client = len(_G["population"])
    total = np.bincount(_G["client_of"], minlength=n_client)
    served_n = np.bincount(_G["client_of"][served], minlength=n_client) \
        if served.any() else np.zeros(n_client, dtype=np.int64)
    return served_n == total


def one_location(loc_idx):
    """Rebuild, run tuned MDLS, return the max-coverage column *and* the
    location's whole front as flattened per-point records."""
    ct, nodes, demands = _G["ct"], _G["nodes"], _G["demands"]
    n_client = len(_G["population"])
    empty = np.zeros(n_client, dtype=bool)
    dep = _G["depot_node"]

    # A forked worker patches the inherited table in place: only the depot
    # row/column pages are ever copied (OS copy-on-write), which is the same
    # sharing trick the irace runners use -- a read-only mmap there, a COW
    # patch here -- and ~8 MB per worker instead of a 2.6 GB copy. The serial
    # path has no fork to isolate it, so it copies instead.
    if _G["in_place"]:
        my_dv, my_ph = ct.dv, ct.phasing
    else:
        my_dv, my_ph = ct.dv.copy(), ct.phasing.copy()
    cached = _G.get("legs_cache")
    if cached is not None:
        got = cached.get(_G["a_km"][loc_idx], _G["incl_deg"][loc_idx])
        if got is None:
            return (loc_idx, empty, 0.0, 0, 0.0,
                    np.zeros(0), np.zeros(0, dtype=int), np.zeros(0),
                    np.zeros((0, n_client), dtype=bool))
        dv, ph = got
    else:
        dv, ph = depot_legs(nodes, _G["a_km"][loc_idx],
                            np.radians(_G["incl_deg"][loc_idx]),
                            ct.dep_days, dep_days=ct.dep_days, tof_days=ct.tof_days)
    my_dv[dep] = dv[0]
    my_dv[:, dep] = dv[1]
    my_ph[dep] = ph[0]
    my_ph[:, dep] = ph[1]
    ct.dv, ct.phasing, ct.reachable = my_dv, my_ph, None

    init = greedy_restarts(ct, demands, dep, _G["max_vehicles"],
                           _G["refuel_time"], dv_budget=_G["dv_budget"])
    if not init:
        return (loc_idx, empty, 0.0, 0, 0.0,
                np.zeros(0), np.zeros(0, dtype=int), np.zeros(0),
                np.zeros((0, n_client), dtype=bool))
    archive, _ = mdls(ct, demands, dep,
                      max_vehicles=_G["max_vehicles"],
                      refuel_time=_G["refuel_time"], max_iter=_G["max_iter"],
                      seed=_G["seed"], dv_budget=_G["dv_budget"],
                      init=init, **_G["params"])
    if len(archive) == 0:
        return (loc_idx, empty, 0.0, 0, 0.0,
                np.zeros(0), np.zeros(0, dtype=int), np.zeros(0),
                np.zeros((0, n_client), dtype=bool))

    k = archive.max_coverage()
    f1, f2 = archive.objectives[k][0], archive.objectives[k][1]
    served = np.ones(len(demands), dtype=bool)
    served[archive.unassigned[k]] = False
    served_value = float(demands.value[served].sum())
    unserved_c = np.bincount(_G["client_of"][~served],
                             weights=demands.value[~served],
                             minlength=n_client)

    obj = archive.objectives                          # (n_pts, 3) dv/fleet/value
    n_pts = obj.shape[0]
    pts_dv = obj[:, 0].astype(np.float64)
    pts_fleet = np.rint(obj[:, 1]).astype(np.int64)
    pts_served = np.ones((n_pts, len(demands)), dtype=bool)
    for q in range(n_pts):
        pts_served[q, archive.unassigned[q]] = False
    pts_value = pts_served.dot(demands.value)
    pts_covered = np.zeros((n_pts, n_client), dtype=bool)
    for q in range(n_pts):
        pts_covered[q] = _covered_mask(pts_served[q])
    return (loc_idx, unserved_c == 0, float(f1), int(round(f2)), served_value,
            pts_dv, pts_fleet, pts_value, pts_covered)


def main():
    if args.jobs > 1:
        os.environ.setdefault(
            "NUMBA_NUM_THREADS",
            str(max(1, (os.cpu_count() or 2) // args.jobs)))

    ct = load_cost_table(args.cost_table, mmap=False)
    _G.update(
        ct=ct,
        nodes=load_nodes(args.sim),
        demands=load_demands(os.path.join(
            args.demand_dir, f"{scenario}_{trial:02d}.h5")),
        population=load_population(args.population, ct),
        max_vehicles=args.max_vehicles,
        refuel_time=args.refuel_time,
        max_iter=max(1, (args.n_eval - 18) // 3),
        seed=trial,
        smoke=args.smoke,
        in_place=args.jobs > 1,
    )
    assert len(_G["nodes"]) == ct.n_nodes

    _G["client_of"] = np.array([int(np.flatnonzero(
        _G["population"].node == n)[0]) for n in _G["demands"].node])
    _G["depot_node"] = ct.names.index("depot_1")
    _G["dv_budget"] = (args.dv_budget if args.dv_budget is not None
                       else ct.dv_budget)
    params = mdls_params(args.mdls_params)
    for k in ("shift", "top_pct", "remove_lo", "remove_hi", "add_lo", "add_hi"):
        if k in params:
            params[k] = float(params[k])
    params["swap"] = bool(params.pop("swap", False))
    _G["params"] = params

    alts, incls = grid_from_args()
    _G["a_km"] = np.repeat(R_E + alts, len(incls))
    _G["incl_deg"] = np.tile(incls, len(alts))
    n_loc = len(_G["a_km"])
    total_value = float(_G["demands"].value.sum())

    if args.legs_cache:
        from oos.depot_legs_cache import LegsCache, build_cache
        try:
            legs = LegsCache(args.legs_cache)
            if (not legs.check(len(_G["nodes"]), ct.dep_days, ct.tof_days,
                               _G["dv_budget"])
                    or not legs.covers(_G["a_km"], _G["incl_deg"])):
                legs = None
        except (OSError, KeyError):
            legs = None
        if legs is None:
            print(f"building depot-legs cache: {args.legs_cache}")
            build_cache(ct, _G["nodes"], _G["a_km"], _G["incl_deg"],
                        _G["dv_budget"], args.legs_cache, jobs=args.jobs)
            legs = LegsCache(args.legs_cache)
        _G["legs_cache"] = legs
        print(f"reusing cached depot legs ({len(legs.a_km)} live locations)")

    print(f"{scenario} trial {trial}: {len(alts)} x {len(incls)} = "
          f"{n_loc} grid locations, budget {_G['dv_budget']:.0f} m/s")
    print(f"tuned MDLS: {params}")
    print(f"max_iter {_G['max_iter']} (n_eval {args.n_eval}), "
          f"{args.jobs} worker(s)")

    t0 = time.time()
    cover_rows = np.empty((n_loc, len(_G["population"])), dtype=bool)
    a_out = np.empty(n_loc)
    incl_out = np.empty(n_loc)
    dv_out = np.zeros(n_loc)
    fleet_out = np.zeros(n_loc, dtype=int)
    served_out = np.zeros(n_loc)
    front_dv = [np.zeros(0) for _ in range(n_loc)]
    front_fleet = [np.zeros(0, dtype=np.int64) for _ in range(n_loc)]
    front_value = [np.zeros(0) for _ in range(n_loc)]
    front_covered = [np.zeros((0, len(_G["population"])), dtype=bool)
                     for _ in range(n_loc)]

    done = 0
    pool = None
    if args.jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        pool = ctx.Pool(args.jobs)
        it = pool.imap_unordered(one_location, range(n_loc))
    else:
        it = map(one_location, range(n_loc))
    try:
        for res in it:
            loc_idx, cov, dv, fleet, served, pdv, pfleet, pval, pcov = res
            a_out[loc_idx] = _G["a_km"][loc_idx]
            incl_out[loc_idx] = _G["incl_deg"][loc_idx]
            cover_rows[loc_idx] = cov
            dv_out[loc_idx] = dv
            fleet_out[loc_idx] = fleet
            served_out[loc_idx] = served
            front_dv[loc_idx] = pdv
            front_fleet[loc_idx] = pfleet
            front_value[loc_idx] = pval
            front_covered[loc_idx] = pcov
            done += 1
            el = time.time() - t0
            print(f"  [{done:3d}/{n_loc}] a={a_out[loc_idx]:7.1f} "
                  f"i={incl_out[loc_idx]:4.1f}  "
                  f"served ${served/1e6:8.2f} M  fleet {fleet:3d}  "
                  f"dv {dv:6.0f} m/s  {len(pdv):4d} pts  "
                  f"({el/done:6.1f} s/loc, {el:5.0f}s)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    loc_npts = np.asarray([len(v) for v in front_dv], dtype=np.int64)
    loc_offset = np.concatenate(([0], np.cumsum(loc_npts)[:-1])).astype(np.int64)
    if loc_npts.sum():
        pts_dv = np.concatenate(front_dv)
        pts_fleet = np.concatenate(front_fleet)
        pts_value = np.concatenate(front_value)
        pts_covered = np.concatenate(front_covered, axis=0)
    else:
        pts_dv = np.zeros(0)
        pts_fleet = np.zeros(0, dtype=np.int64)
        pts_value = np.zeros(0)
        pts_covered = np.zeros((0, len(_G["population"])), dtype=bool)

    print(f"done in {time.time() - t0:.0f}s, "
          f"{int((served_out > 0).sum())} live locations, "
          f"{int(loc_npts.sum())} front points total")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"depot_sweep_{scenario}.h5")
    import h5py
    with h5py.File(out, "w") as f:
        f.create_dataset("coverage", data=cover_rows)
        f.create_dataset("client_value", data=np.bincount(
            _G["client_of"], weights=_G["demands"].value,
            minlength=len(_G["population"])))
        f.create_dataset("client_n_requests", data=np.bincount(
            _G["client_of"], minlength=len(_G["population"])))
        f.create_dataset("a_km", data=a_out)
        f.create_dataset("incl_deg", data=incl_out)
        f.create_dataset("dv_m_s", data=dv_out)
        f.create_dataset("fleet", data=fleet_out)
        f.create_dataset("served_request_value", data=served_out)
        f.create_dataset("client_names", data=np.asarray(
            _G["population"].name, dtype="S"))
        f.create_dataset("loc_npts", data=loc_npts)
        f.create_dataset("loc_offset", data=loc_offset)
        f.create_dataset("points_dv_m_s", data=pts_dv)
        f.create_dataset("points_fleet", data=pts_fleet)
        f.create_dataset("points_served_value", data=pts_value)
        f.create_dataset("points_covered", data=pts_covered,
                         compression="gzip")
        f.attrs["scenario"] = scenario
        f.attrs["trial"] = trial
        f.attrs["n_demands"] = len(_G["demands"])
        f.attrs["total_value_usd"] = total_value
        f.attrs["mdls_params"] = " ".join(
            f"--{k} {v}" if not isinstance(v, bool) else f"--{k}"
            for k, v in params.items())
        f.attrs["grid"] = (f"alt {float(args.alt_lo)}-{float(args.alt_hi)} "
                           f"km da{float(args.alt_step)}, "
                           f"inc {float(args.incl_lo)}-{float(args.incl_hi)} "
                           f"di{float(args.incl_step)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()