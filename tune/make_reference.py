#!/usr/bin/env python3
"""Fix the hypervolume reference box for each tuning instance, once.

irace needs one number per run, and for a three-objective problem that number
is a hypervolume. A hypervolume needs a scale, and the scale must not depend on
the configuration being scored -- otherwise a configuration can improve its own
score by producing a worse front, which moves the box. So the box is computed
here, before any tuning, from pilot runs of both algorithms at their default
parameters, and the target runner only ever reads it.

Both algorithms contribute to every box, so neither is scored on a scale drawn
from its own behaviour.

Writes outputs/tune_reference.json: {instance: {ideal: [...], nadir: [...]}}.

Run:  python tune/make_reference.py [--n-eval 10000] [--pilot-seeds 2]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def mdls_front(ct, dem, depot, seed, n_eval, max_vehicles, refuel_time):
    from oos.greedy import greedy_restarts
    from oos.mdls import mdls
    seeds = greedy_restarts(ct, dem, depot, max_vehicles, refuel_time,
                            dv_budget=ct.dv_budget)
    max_iter = max(1, (n_eval - len(seeds)) // 3)
    archive, _ = mdls(ct, dem, depot, max_vehicles=max_vehicles,
                      refuel_time=refuel_time, max_iter=max_iter, seed=seed,
                      dv_budget=ct.dv_budget, init=seeds)
    return archive.objectives[:, [0, 2, 1]] if len(archive) else None


def nsga2_front(ct, dem, depot, seed, n_eval, max_vehicles, refuel_time):
    """Returns (front, reason). `reason` is None on success and names the
    failure otherwise -- "pymoo is not installed" and "the run produced nothing
    feasible" are very different problems and the first version reported both
    as the first one."""
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
    except ImportError as exc:
        return None, f"pymoo will not import: {exc}"
    from oos.ga_problem import GreedySeeding, ScheduleProblem
    problem = ScheduleProblem(cost_table=ct, demands=dem, depot_node=depot,
                              max_vehicles=max_vehicles,
                              refuel_time=refuel_time)
    res = minimize(problem, NSGA2(pop_size=91, sampling=GreedySeeding()),
                   termination=("n_eval", n_eval), seed=seed, verbose=False)
    F = np.asarray(res.algorithm.pop.get("F"))
    cv = np.asarray(res.algorithm.pop.get("CV")).ravel()
    F = F[cv <= 0]
    if not len(F):
        return None, f"no feasible individual in the final population of {len(cv)}"
    return F[:, [0, 2, 1]], None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demand-dir", default=os.path.join(ROOT, "outputs/tune_demands"))
    p.add_argument("--cost-table", default=os.path.join(ROOT, "outputs/cost_table.h5"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs/tune_reference.json"))
    p.add_argument("--n-eval", type=int, default=10_000)
    p.add_argument("--pilot-seeds", type=int, default=2)
    p.add_argument("--max-vehicles", type=int, default=25)
    p.add_argument("--refuel-time", type=float, default=0.5)
    p.add_argument("--allow-one-sided", action="store_true",
                   help="write the boxes even if one algorithm contributed no "
                        "pilot front; never appropriate for the paper")
    args = p.parse_args()

    from oos.demands import load_demands
    from oos.schedule import load_cost_table

    ct = load_cost_table(args.cost_table)
    depot = ct.names.index("depot_1")
    files = sorted(f for f in os.listdir(args.demand_dir) if f.endswith(".h5"))
    if not files:
        sys.exit(f"no instances in {args.demand_dir}; run generate_demands.py "
                 "with --trial-offset first")

    out, missing_ga, reasons = {}, 0, set()
    print(f"  {len(files)} instances x {args.pilot_seeds} seeds x 2 algorithms "
          f"= {len(files)*args.pilot_seeds*2} pilot runs, a few minutes",
          flush=True)
    print(f"{'instance':24s} {'fronts':>7s} {'ideal dV':>9s} {'nadir dV':>9s}",
          flush=True)
    for fn in files:
        dem = load_demands(os.path.join(args.demand_dir, fn))
        fronts = []
        for s in range(1, args.pilot_seeds + 1):
            a = mdls_front(ct, dem, depot, s, args.n_eval, args.max_vehicles,
                           args.refuel_time)
            if a is not None:
                fronts.append(a)
            b, why = nsga2_front(ct, dem, depot, s, args.n_eval,
                                 args.max_vehicles, args.refuel_time)
            if b is None:
                missing_ga += 1
                reasons.add(why)
            else:
                fronts.append(b)
        if not fronts:
            print(f"{fn:24s}   no pilot front; skipped")
            continue
        U = np.vstack(fronts)
        out[fn] = dict(ideal=U.min(axis=0).tolist(),
                       nadir=U.max(axis=0).tolist(),
                       n_pilot_fronts=len(fronts))
        print(f"{fn:24s} {len(fronts):7d} {U[:,0].min():9.0f} "
              f"{U[:,0].max():9.0f}", flush=True)

    if missing_ga and not args.allow_one_sided:
        sys.exit(
            f"\n{missing_ga} of the NSGA-II pilot runs produced no front, so the\n"
            "reference boxes would come from MDLS alone and the two algorithms\n"
            "would be scored on a scale drawn from only one of them. Nothing was\n"
            "written. Reasons given:\n  "
            + "\n  ".join(sorted(r for r in reasons if r))
            + "\n\nFix that, or pass --allow-one-sided if you have decided the\n"
              "asymmetry is acceptable -- it is not, for the paper.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for v in out.values():
        v["both_algorithms"] = not missing_ga
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}: {len(out)} instances")


if __name__ == "__main__":
    main()
