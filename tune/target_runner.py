#!/usr/bin/env python3
"""irace target runner: one algorithm run, scored as a single number.

irace calls this as

    target-runner <config_id> <instance_id> <seed> <instance> <bound> [--par v ...]

and reads one number from stdout, which it minimises.

Turning three objectives into one. The score is the negated hypervolume of the
front the run produces, normalised per instance against a reference box fixed
in advance by `make_reference.py` and never recomputed. Fixing it in advance is
the point: a box derived from the run being scored would move with the
configuration, and a configuration could then improve its own score by
producing a worse front. Points outside the box on the wrong side contribute
nothing; points better than the reference ideal are *not* clipped, so a
configuration that beats the pilot runs is rewarded for it.

Every configuration gets the same evaluation budget, the same instance and the
same seed within a race, so the only thing varying is the parameter vector.

Failures return a large finite cost rather than crashing the race: an
infeasible or empty front is a legitimate outcome for a bad configuration and
irace should be able to rank it, not stop.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAIL = 1.0          # worse than any attainable -hypervolume, which is <= 0


def hv3(P, ref=(1.0, 1.0, 1.0)):
    """Exact 3-D hypervolume by slab decomposition. Minimisation."""
    P = np.asarray(P, float)
    P = P[np.all(P < np.asarray(ref), axis=1)]
    if not len(P):
        return 0.0
    P = P[P[:, 2].argsort()]
    vol = 0.0
    for k in range(len(P)):
        z_next = P[k + 1, 2] if k + 1 < len(P) else ref[2]
        Q = P[:k + 1, :2]
        Q = Q[Q[:, 0].argsort()]
        area, ybest = 0.0, ref[1]
        for i in range(len(Q)):
            x_next = Q[i + 1, 0] if i + 1 < len(Q) else ref[0]
            ybest = min(ybest, Q[i, 1])
            area += max(0.0, x_next - Q[i, 0]) * max(0.0, ref[1] - ybest)
        vol += area * max(0.0, z_next - P[k, 2])
    return vol


def parse(argv):
    """(instance, seed, params) from irace's argument convention."""
    _cfg_id, _inst_id, seed, instance = argv[1], argv[2], argv[3], argv[4]
    rest = argv[6:]                      # argv[5] is the bound, unused here
    params = {}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            params[key] = rest[i + 1]
            i += 2
        else:
            params[key] = True           # a bare flag, e.g. --swap
            i += 1
    return instance, int(seed), params


def run(algo, instance, seed, params, n_eval, cost_table, demand_dir,
        max_vehicles, refuel_time):
    from oos.demands import load_demands
    from oos.schedule import load_cost_table

    ct = load_cost_table(cost_table)
    depot = ct.names.index("depot_1")
    dem = load_demands(os.path.join(demand_dir, instance))

    def f(name, default):
        return float(params.get(name, default))

    def i_(name, default):
        return int(round(float(params.get(name, default))))

    if algo == "mdls":
        from oos.greedy import greedy_restarts
        from oos.mdls import mdls
        seeds = greedy_restarts(ct, dem, depot, max_vehicles, refuel_time,
                                dv_budget=ct.dv_budget)
        if not seeds:
            return None
        max_iter = max(1, (n_eval - len(seeds)) // 3)
        archive, _ = mdls(ct, dem, depot, max_vehicles=max_vehicles,
                          refuel_time=refuel_time, max_iter=max_iter,
                          seed=seed, dv_budget=ct.dv_budget, init=seeds,
                          shift=f("shift", 15.0), top_pct=f("top_pct", 0.5),
                          remove_lo=f("remove_lo", 0.20),
                          remove_hi=f("remove_hi", 0.90),
                          add_lo=f("add_lo", 0.10),
                          add_hi=f("add_hi", 1.00),
                          swap=bool(params.get("swap", False)),
                          swap_top_n=i_("swap_top_n", 50))
        if not len(archive):
            return None
        return archive.objectives[:, [0, 2, 1]]      # -> [dv, unrecovered, veh]

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize

    from oos.ga_problem import GreedySeeding, ScheduleProblem

    problem = ScheduleProblem(cost_table=ct, demands=dem, depot_node=depot,
                              max_vehicles=max_vehicles,
                              refuel_time=refuel_time)
    pm_prob = params.get("pm_prob")
    algorithm = NSGA2(
        pop_size=i_("pop_size", 91),
        sampling=GreedySeeding(),
        crossover=SBX(eta=f("sbx_eta", 20.0), prob=f("sbx_prob", 0.9)),
        mutation=(PM(eta=f("pm_eta", 20.0)) if pm_prob in (None, "")
                  else PM(eta=f("pm_eta", 20.0), prob=float(pm_prob))),
    )
    res = minimize(problem, algorithm, termination=("n_eval", n_eval),
                   seed=seed, verbose=False)
    F = np.asarray(res.algorithm.pop.get("F"))
    cv = np.asarray(res.algorithm.pop.get("CV")).ravel()
    F = F[cv <= 0]
    if not len(F):
        return None                      # nothing feasible: a real, rankable outcome
    F = np.unique(F, axis=0)
    keep = [i for i in range(len(F))
            if not any(j != i and np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                       for j in range(len(F)))]
    return F[keep][:, [0, 2, 1]]


def main():
    algo = os.environ.get("OOS_ALGO", "mdls")
    n_eval = int(os.environ.get("OOS_N_EVAL", "10000"))
    cost_table = os.environ.get("OOS_COST_TABLE",
                                os.path.join(ROOT, "outputs/cost_table.h5"))
    demand_dir = os.environ.get("OOS_DEMAND_DIR",
                                os.path.join(ROOT, "outputs/tune_demands"))
    ref_path = os.environ.get("OOS_REFERENCE",
                              os.path.join(ROOT, "outputs/tune_reference.json"))
    max_vehicles = int(os.environ.get("OOS_MAX_VEHICLES", "25"))
    refuel_time = float(os.environ.get("OOS_REFUEL_TIME", "0.5"))

    # irace runs the target runner from execDir, not from the project root, so
    # every path here is resolved against the package root rather than the
    # working directory. Without this the runner cannot find the cost table and
    # returns the failure value for every configuration -- and irace, with no
    # way to tell a tie from a failure, still reports a "best" one (F43).
    cost_table = os.path.join(ROOT, cost_table) if not os.path.isabs(cost_table) else cost_table
    demand_dir = os.path.join(ROOT, demand_dir) if not os.path.isabs(demand_dir) else demand_dir
    ref_path = os.path.join(ROOT, ref_path) if not os.path.isabs(ref_path) else ref_path

    instance, seed, params = parse(sys.argv)
    try:
        with open(ref_path) as fh:
            ref = json.load(fh)[os.path.basename(instance)]
    except (OSError, KeyError):
        print(FAIL)
        return

    try:
        F = run(algo, instance, seed, params, n_eval, cost_table, demand_dir,
                max_vehicles, refuel_time)
    except Exception as exc:                              # noqa: BLE001
        print(f"target runner failed on {instance}: {exc}", file=sys.stderr)
        print(FAIL)
        return

    if F is None or not len(F):
        print(FAIL)
        return

    ideal = np.asarray(ref["ideal"], float)
    span = np.maximum(np.asarray(ref["nadir"], float) - ideal, 1e-12)
    print(-hv3((F - ideal) / span))


if __name__ == "__main__":
    main()
