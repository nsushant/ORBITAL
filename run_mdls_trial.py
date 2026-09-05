"""
run_mdls_trial.py — run MDLS on one (scenario, trial) pair.

The counterpart of run_ga_trial.py, deliberately symmetric with it: same
inputs, same output format, same evaluation budget, and the same starting
schedule from the Sec. 3.6 constructive heuristic. Every number either
produces comes from oos.schedule.evaluate(), so a front from this script and a
front from run_ga_trial.py are comparable by construction rather than by
convention -- which is the whole point of the shared evaluator (D19, D22).

On the budget. MDLS spends three evaluations per iteration, one per objective
(Sec. 3.7, Algorithm 1), so --n-eval is converted to iterations by dividing by
three rather than being passed through. That is what makes "equal evaluation
budget" mean the same thing for a local search and a population method. The
script reports what it actually spent, so the claim can be checked rather than
asserted.

CLI: python run_mdls_trial.py <scenario> <trial>

Reads:  outputs/cost_table.h5
        outputs/exp_demands/{scenario}_{trial:02d}.h5
Writes: outputs/exp_results/mdls_{scenario}_{trial:02d}.csv

Objective column order: f1_dv, f2_unrecovered_value, f3_vehicles -- matching
run_ga_trial.py and the existing analysis scripts.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("key", help="demand file key, e.g. S2_refuel")
parser.add_argument("trial", type=int)
parser.add_argument("--demand-dir", default="outputs/exp_demands")
parser.add_argument("--result-dir", default="outputs/exp_results")
parser.add_argument("--cost-table", default="outputs/cost_table.h5")
parser.add_argument("--dv-budget", type=float, default=None,
                    help="defaults to the cost table's own dv_budget_m_s")
parser.add_argument("--max-vehicles", type=int, default=25)
parser.add_argument("--refuel-time", type=float, default=0.5)
parser.add_argument("--n-eval", type=int, default=10_000,
                    help="evaluation budget; MDLS spends 3 per iteration")
parser.add_argument("--shift", type=float, default=15.0,
                    help="timing operator shift [days]")
parser.add_argument("--top-pct", type=float, default=0.5,
                    help="fraction of costliest legs the timing operator tries")
parser.add_argument("--time-limit", type=float, default=None, help="seconds")
parser.add_argument("--h5-file", default=None)
parser.add_argument("--remove-lo", type=float, default=0.20)
parser.add_argument("--remove-hi", type=float, default=0.90)
parser.add_argument("--add-lo", type=float, default=0.10)
parser.add_argument("--add-hi", type=float, default=1.00)
parser.add_argument("--swap", action="store_true",
                    help="add op_swap_cross_vehicle to the delta-V pool. Off "
                         "by default: real per-route savings, but no measured "
                         "effect on the front and 2-3x the runtime (F41)")
parser.add_argument("--swap-top-n", type=int, default=50)
parser.add_argument("--raan-reseq", action="store_true",
                    help="add op_raan_resequence to the delta-V pool. Off by "
                         "default: measured on this instance it never improves "
                         "a route, and enabling it spends half the f1 budget "
                         "re-evaluating unchanged schedules (F40)")
parser.add_argument("--sim", default="outputs/simulation.h5",
                    help="ephemeris, for --raan-reseq's nodal rates")
args = parser.parse_args()

scenario, trial = args.key, args.trial
trial_str = f"{trial:02d}"

from oos.schedule import evaluate, is_feasible, load_cost_table
from oos.demands import load_demands
from oos.greedy import greedy_restarts
from oos.mdls import mdls, raan_model

os.makedirs(args.result_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"  MDLS   scenario={scenario}  trial={trial}")
print(f"{'='*60}")

ct = load_cost_table(args.cost_table)
dv_budget = args.dv_budget if args.dv_budget is not None else ct.dv_budget
try:
    depot_node = ct.names.index("depot_1")
except ValueError:
    sys.exit(f"'depot_1' not in {args.cost_table}'s node names")

dem_path = os.path.join(args.demand_dir, f"{scenario}_{trial_str}.h5")
dem = load_demands(dem_path)
print(f"Cost table: {ct.n_nodes} nodes, budget {dv_budget:.0f} m/s")
print(f"Demands:    {len(dem)} requests, "
      f"${dem.value.sum()/1e6:.1f} M total recovery potential")

# The same set of starting schedules the GA is seeded with.
init = greedy_restarts(ct, dem, depot_node, args.max_vehicles,
                       args.refuel_time, dv_budget=dv_budget)
F0 = np.array([evaluate(s, dem, dv_budget)[0] for s in init])
ok = all(is_feasible(evaluate(s, dem, dv_budget)[1]) for s in init)
print(f"Seeds:      {len(init)} greedy restarts, all feasible={ok}")
print(f"            dV {F0[:,0].min():.0f}-{F0[:,0].max():.0f} m/s, "
      f"fleet {F0[:,1].min():.0f}-{F0[:,1].max():.0f}, "
      f"unrecovered ${F0[:,2].min()/1e6:.1f}-{F0[:,2].max()/1e6:.1f} M")

# Seeding costs one evaluation per restart, and each iteration costs three
# (one per objective). Both come out of the same budget the GA is held to.
raan = None
if args.raan_reseq:
    from oos.nodes import load_nodes
    raan = raan_model(ct, load_nodes(args.sim))
    print(f"Re-sequencing: on, nodal rates for {len(raan[0])} nodes")

max_iter = max(1, (args.n_eval - len(init)) // 3)
print(f"\n--- MDLS: {max_iter} iterations + {len(init)} seeds "
      f"= {3*max_iter + len(init)} of {args.n_eval} evaluations ---")
t0 = time.time()
archive, n_eval = mdls(ct, dem, depot_node,
                       max_vehicles=args.max_vehicles,
                       refuel_time=args.refuel_time,
                       max_iter=max_iter, seed=trial,
                       shift=args.shift, top_pct=args.top_pct,
                       dv_budget=dv_budget, init=init, raan=raan,
                       swap=args.swap, swap_top_n=args.swap_top_n,
                       remove_lo=args.remove_lo, remove_hi=args.remove_hi,
                       add_lo=args.add_lo, add_hi=args.add_hi,
                       time_limit=args.time_limit, verbose=True)
print(f"  done in {time.time()-t0:.1f}s, {n_eval} evaluations spent, "
      f"{len(archive)} front points")

if len(archive) == 0:
    sys.exit("MDLS returned an empty archive, which should be impossible: the "
             "seed is feasible and always enters. Check oos/archive.py.")

F = archive.objectives                       # [dv, vehicles, unrecovered]
F_out = F[:, [0, 2, 1]]                      # -> [dv, unrecovered, vehicles]
outpath = os.path.join(args.result_dir, f"mdls_{scenario}_{trial_str}.csv")
pd.DataFrame(F_out, columns=["f1_dv", "f2_unrecovered_value", "f3_vehicles"]
             ).to_csv(outpath, index=False)
print(f"  {len(F_out)} front points -> {outpath}")

if args.h5_file is not None:
    import h5py
    group_path = f"mdls/{scenario}/trial_{trial_str}"
    os.makedirs(os.path.dirname(os.path.abspath(args.h5_file)), exist_ok=True)
    with h5py.File(args.h5_file, "a") as fid:
        if group_path in fid:
            del fid[group_path]
        ds = fid.create_dataset(group_path, data=F_out.astype(np.float64))
        ds.attrs["columns"] = "f1_dv,f2_unrecovered_value,f3_vehicles"
        ds.attrs["n_eval"] = n_eval
    print(f"  saved -> {args.h5_file}::{group_path}")

print(f"\n  front: dV {F[:,0].min():.0f}-{F[:,0].max():.0f} m/s   "
      f"vehicles {F[:,1].min():.0f}-{F[:,1].max():.0f}   "
      f"unrecovered ${F[:,2].min()/1e6:.1f}-{F[:,2].max()/1e6:.1f} M")
print(f"\nDone for {scenario} trial {trial}.")
