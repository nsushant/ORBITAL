"""
run_ga_trial.py — run NSGA-II on one (scenario, trial) pair.

D21 (PAPER_COMPLETION_PLAN.md): NSGA-III was replaced by NSGA-II. At this
paper's three objectives NSGA-III's reference-direction machinery (built for
four-plus objectives) buys nothing over NSGA-II's crowding distance, and
NSGA-II is the more standard, more defensible three-objective baseline.

Second change, same decision thread: this now runs a single problem class,
`oos.ga_problem.ScheduleProblem`, wired straight to the shared evaluator
(`oos.schedule.evaluate`) with pymoo's real constraint handling (`n_constr=3`)
instead of a decoder that searches for, or repairs its way to, a feasible
schedule before pymoo ever sees it. See oos/ga_problem.py's docstring for the
full reasoning -- this is "make MDLS and NSGA-II solve the same problem":
both score a `Schedule` through the identical `evaluate()` call, so a result
from this script and a result from the MDLS port are comparable by
construction, not by convention.

The old `nsga2rk` / `nsga2rk_ot` split (OOSProblemRK / OOSProblemRK_OT in
pymoo_stuff.py, both decoder-based, both retired) is gone. There is one
encoding now -- the paper's "time-inclusive" encoding, Sec. 3.8 -- so there
is nothing left for an --algos flag to select between.

CLI: python run_ga_trial.py <scenario> <trial>

Reads:  outputs/cost_table.h5                          (349 nodes)
        outputs/exp_demands/{scenario}_{trial:02d}.h5   (generate_demands.py)
Writes: outputs/exp_results/nsga2_{scenario}_{trial:02d}.csv

Objective column order in the CSV: f1_dv, f2_unrecovered_value, f3_vehicles
(oos.schedule.evaluate returns [dv, vehicles, unrecovered]; reordered to
match the existing CSV convention other scripts in this repo expect).

The population is seeded with the Sec. 3.6 constructive heuristic
(oos.ga_problem.GreedySeeding): individual 0 is that schedule encoded for this
problem, the rest uniform. MDLS starts from the same schedule, so neither
method is handed a better starting point than the other.
"""

import sys, os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("key",   help="demand file key, e.g. tight_normal or size_10")
parser.add_argument("trial", type=int, help="trial number")
parser.add_argument("--demand-dir", default="outputs/exp_demands",
                    help="directory containing demand files")
parser.add_argument("--result-dir", default="outputs/exp_results",
                    help="directory to write result CSVs")
parser.add_argument("--cost-table", default="outputs/cost_table.h5",
                    help="HDF5 cost table (oos.schedule.load_cost_table format)")
parser.add_argument("--dv-budget", type=float, default=None,
                    help="ΔV budget per vehicle per sortie [m/s]; "
                         "defaults to the cost table's own dv_budget_m_s attribute")
parser.add_argument("--max-vehicles", type=int, default=25,
                    help="upper bound on fleet size, n_v in the problem statement")
parser.add_argument("--refuel-time", type=float, default=0.5,
                    help="days spent at the depot per refuel -- must match MDLS")
parser.add_argument("--pop-size", type=int, default=91,
                    help="NSGA-II population size (D21: was len(ref_dirs) under NSGA-III)")
parser.add_argument("--sbx-eta", type=float, default=20.0,
                    help="SBX distribution index; larger keeps children nearer "
                         "their parents")
parser.add_argument("--sbx-prob", type=float, default=0.9,
                    help="crossover probability")
parser.add_argument("--pm-eta", type=float, default=20.0,
                    help="polynomial mutation index; larger perturbs less")
parser.add_argument("--pm-prob", type=float, default=None,
                    help="per-gene mutation probability; pymoo's default is "
                         "1/n_var when unset")
parser.add_argument("--n-eval", type=int, default=10_000,
                    help="Total function evaluations budget")
parser.add_argument("--h5-file", default=None,
                    help="Optional path to HDF5 file; if given, front is also written there")
args = parser.parse_args()

scenario  = args.key
trial     = args.trial
trial_str = f"{trial:02d}"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from oos.schedule import load_cost_table, is_feasible
from oos.demands import load_demands
from oos.ga_problem import GreedySeeding, ScheduleProblem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

EXP_DIR = args.demand_dir
RES_DIR = args.result_dir
os.makedirs(RES_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  scenario={scenario}  trial={trial}")
print(f"{'='*60}")

ct = load_cost_table(args.cost_table)
print(f"Cost table loaded: {ct.n_nodes} nodes, dv_budget={ct.dv_budget:.0f} m/s "
      f"(from {args.cost_table})")

dv_budget = args.dv_budget if args.dv_budget is not None else ct.dv_budget

try:
    depot_node = ct.names.index("depot_1")
except ValueError:
    sys.exit(f"'depot_1' not found in {args.cost_table}'s node names -- "
             f"check --cost-table points at the rebuilt instance.")

dem_path = os.path.join(EXP_DIR, f"{scenario}_{trial_str}.h5")
demands = load_demands(dem_path)
print(f"Demands loaded: {len(demands)} requests, "
      f"${demands.value.sum()/1e6:.1f} M total recovery potential "
      f"(from {dem_path})")

# ---------------------------------------------------------------------------
# Helper — extract front from result, reorder columns, save CSV
# Objective order from oos.schedule.evaluate: F[:,0]=dv F[:,1]=vehicles F[:,2]=unrecovered
# CSV column order:                            f1_dv    f2_unrecovered_value f3_vehicles
# ---------------------------------------------------------------------------

def save_front(res, algo_name):
    outpath = os.path.join(RES_DIR, f"{algo_name}_{scenario}_{trial_str}.csv")

    if res.F is None or len(res.F) == 0:
        print(f"  {algo_name}: no solutions found")
        return

    # res.G is real now (D21): a front point is feasible iff oos.schedule's
    # own violations are all <= 0, exactly the check oos.schedule.is_feasible
    # already applies everywhere else. No decoder-specific re-inspection.
    G = res.G if res.G is not None else np.zeros((len(res.F), 0))
    feasible = [i for i in range(len(res.F)) if is_feasible(G[i])]

    if len(feasible) == 0:
        print(f"  {algo_name}: {len(res.F)} front points, 0 feasible — skipping")
        return

    F = res.F[feasible]

    # Reduce to the unique non-dominated set before reporting. pymoo returns
    # the final population, which after convergence can be many copies of the
    # same solution -- 91 identical objective vectors in one S3 trial. MDLS's
    # archive rejects duplicates by construction (oos/archive.py), so without
    # this the two methods' "front size" would be counting different things.
    # Hypervolume is unaffected either way; the point count is not.
    F = np.unique(F, axis=0)
    keep = [i for i in range(len(F))
            if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                       for j in range(len(F)) if j != i)]
    F = F[keep]
    print(f"  {algo_name}: {len(res.F)} returned, {len(feasible)} feasible, "
          f"{len(F)} unique non-dominated")

    # Reorder: [dv, vehicles, unrecovered] → [dv, unrecovered, vehicles]
    F_out = F[:, [0, 2, 1]]
    pd.DataFrame(F_out, columns=["f1_dv", "f2_unrecovered_value", "f3_vehicles"]).to_csv(
        outpath, index=False
    )
    print(f"  {algo_name}: {len(F_out)} front points → {outpath}")

    if args.h5_file is not None:
        import h5py
        group_path = f"{algo_name}/{scenario}/trial_{trial_str}"
        os.makedirs(os.path.dirname(os.path.abspath(args.h5_file)), exist_ok=True)
        with h5py.File(args.h5_file, "a") as fid:
            if group_path in fid:
                del fid[group_path]
            ds = fid.create_dataset(group_path, data=F_out.astype(np.float64))
            ds.attrs["columns"] = "f1_dv,f2_unrecovered_value,f3_vehicles"
        print(f"  {algo_name}: saved → {args.h5_file}::{group_path}")

# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback(Callback):
    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % 10 == 0 or gen == 1:
            n_eval = algorithm.evaluator.n_eval
            cv = algorithm.pop.get("CV")
            print(f"  [nsga2] gen {gen:4d} | n_eval {n_eval:6d} | "
                  f"n_feasible {int((cv <= 0).sum()):3d}/{len(cv)}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

import time

print("\n--- NSGA-II (time-inclusive encoding, real constraint handling) ---")
problem = ScheduleProblem(
    cost_table=ct, demands=demands, depot_node=depot_node,
    max_vehicles=args.max_vehicles, refuel_time=args.refuel_time,
)
algo = NSGA2(
    pop_size  = args.pop_size,
    sampling  = GreedySeeding(),
    crossover = SBX(eta=args.sbx_eta, prob=args.sbx_prob),
    mutation  = (PM(eta=args.pm_eta) if args.pm_prob is None
                 else PM(eta=args.pm_eta, prob=args.pm_prob)),
)
t0 = time.time()
res = minimize(problem, algo, termination=("n_eval", args.n_eval),
               seed=trial, verbose=False, callback=ProgressCallback())
print(f"  done in {time.time()-t0:.1f}s")
save_front(res, "nsga2")

print(f"\nDone for {scenario} trial {trial}.")
