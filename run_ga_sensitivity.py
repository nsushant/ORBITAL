"""
run_ga_sensitivity.py — run NSGA-III or NSGA-III-T with a single OAT parameter override.

CLI:
  python run_ga_sensitivity.py <key> <trial> <algo> <param_name> <param_level> <h5_file>
                               [--demand-dir DIR] [--dv-budget FLOAT] [--n-eval INT]

algo       : nsga3rk | nsga3rk_ot
param_name : sbx_eta | pm_eta | crossover_prob | shift | top_pct | n_ref_dirs
param_level: float value

Writes dataset /{algo}/{param_name}/{level_str}/trial_{nn} → (n_solutions × 3)
Columns: f1_dv, f2_unrecovered_value, f3_vehicles
"""

import sys, os, argparse
import numpy as np
import h5py

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("key",         help="demand file key, e.g. tight_low_dv")
parser.add_argument("trial",       type=int)
parser.add_argument("algo",        choices=["nsga3rk", "nsga3rk_ot"])
parser.add_argument("param_name",  choices=["sbx_eta", "pm_eta", "crossover_prob", "shift", "top_pct", "n_ref_dirs"])
parser.add_argument("param_level", type=float)
parser.add_argument("h5_file",     help="path to shared HDF5 output file")
parser.add_argument("--demand-dir",  default="outputs/exp_demands")
parser.add_argument("--dv-budget",   type=float, default=5000.0)
parser.add_argument("--n-eval",      type=int,   default=10_000)
args = parser.parse_args()

scenario  = args.key
trial     = args.trial
trial_str = f"{trial:02d}"

# ---------------------------------------------------------------------------
# Nominal parameters (only the swept one is overridden)
# ---------------------------------------------------------------------------

sbx_eta        = 20
pm_eta         = 20
crossover_prob = 0.9
ot_shift       = 15.0
ot_top_pct     = 0.5
n_partitions   = 12

if args.param_name == "sbx_eta":
    sbx_eta = int(args.param_level)
elif args.param_name == "pm_eta":
    pm_eta = int(args.param_level)
elif args.param_name == "crossover_prob":
    crossover_prob = args.param_level
elif args.param_name == "shift":
    ot_shift = args.param_level
elif args.param_name == "top_pct":
    ot_top_pct = args.param_level
elif args.param_name == "n_ref_dirs":
    n_partitions = int(args.param_level)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from loaders import load_sim_name_map, load_demands, load_cost_table_jld2, load_min_tof_table
from pymoo_stuff import OOSProblemRK, OOSProblemRK_OT
from greedy_init import GreedySamplingRK, GreedySamplingRK2, load_greedy_RK_from_json

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

EXP_DIR   = args.demand_dir
dem_path  = os.path.join(EXP_DIR, f"{scenario}_{trial_str}.jld2")
greedy_path = os.path.join(EXP_DIR, f"{scenario}_{trial_str}_greedy.json")

if not os.path.exists(dem_path):
    sys.exit(f"Demand file not found: {dem_path}")

print(f"\n{'='*60}")
print(f"  algo={args.algo}  scenario={scenario}  trial={trial}")
print(f"  {args.param_name}={args.param_level}")
print(f"{'='*60}")

COST_FILE   = "outputs/cost_table.jld2"
REFUEL_TIME = 0.5

d           = load_demands(dem_path)
ct, ct_meta = load_cost_table_jld2(COST_FILE)
nm          = load_sim_name_map("outputs/simulation.h5", depot_idx=ct_meta["depot_idx"])
min_tof     = load_min_tof_table()

sat_ids  = [nm[s] for s in d["sat_identifiers"]]
depot_id = nm["depot_1"]
Ndems    = len(d["demand_deadlines"])
maxV     = 25

greedy_x_rk = load_greedy_RK_from_json(greedy_path, Ndems) if os.path.exists(greedy_path) else None

# ---------------------------------------------------------------------------
# Build problem and algorithm
# ---------------------------------------------------------------------------

ProbClass = OOSProblemRK_OT if args.algo == "nsga3rk_ot" else OOSProblemRK

prob = ProbClass(
    Ndems         = Ndems,
    maxV          = maxV,
    deadlines     = d["demand_deadlines"],
    service_times = d["service_times"],
    asset_values  = d.get("asset_values", None),
    sat_ids       = sat_ids,
    depot_id      = depot_id,
    cost_table    = ct,
    dep_grid      = ct_meta["dep_grid"],
    arr_grid      = ct_meta["arr_grid"],
    min_tof_table = min_tof,
    refuel_time   = REFUEL_TIME,
    dv_budget     = args.dv_budget,
)

ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=n_partitions)
if args.algo == "nsga3rk_ot":
    sampling = GreedySamplingRK2(greedy_x_rk) if greedy_x_rk is not None else None
else:
    sampling = GreedySamplingRK(greedy_x_rk) if greedy_x_rk is not None else None

algo = NSGA3(
    ref_dirs  = ref_dirs,
    pop_size  = len(ref_dirs),
    sampling  = sampling,
    crossover = SBX(eta=sbx_eta, prob=crossover_prob),
    mutation  = PM(eta=pm_eta),
)

class ProgressCallback(Callback):
    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % 20 == 0 or gen == 1:
            print(f"  gen {gen:4d} | n_eval {algorithm.evaluator.n_eval:6d}")

import time
t0  = time.time()
res = minimize(prob, algo, termination=("n_eval", args.n_eval),
               seed=trial, verbose=False, callback=ProgressCallback())
print(f"  done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Extract feasible front
# ---------------------------------------------------------------------------

if res.F is None or len(res.F) == 0:
    print("  no solutions found — skipping")
    sys.exit(0)

feasible = []
for i, x in enumerate(res.X):
    if hasattr(prob, '_decode_rk2'):
        arrivals, _, served_arr, _, _ = prob._decode_rk2(x)
    else:
        arrivals, _, served_arr, _, _ = prob._decode_rk(x)
    served = served_arr > 0
    n_viol = int(np.sum(arrivals[served] + prob.service_times[served] > prob.deadlines[served] + 1e-6)) if served.any() else 0
    if n_viol == 0:
        feasible.append(i)

if len(feasible) == 0:
    print("  0 feasible solutions — skipping")
    sys.exit(0)

F = res.F[feasible]

# Reorder: [dv, vehicles, unrecovered] → [dv, unrecovered, vehicles]
F_out = F[:, [0, 2, 1]]
print(f"  {len(F_out)} feasible solutions")

# ---------------------------------------------------------------------------
# Write to HDF5
# ---------------------------------------------------------------------------

level_str  = str(args.param_level).replace(".", "p")
group_path = f"{args.algo}/{args.param_name}/{level_str}/trial_{trial_str}"

os.makedirs(os.path.dirname(os.path.abspath(args.h5_file)), exist_ok=True)
with h5py.File(args.h5_file, "a") as fid:
    if group_path in fid:
        del fid[group_path]
    ds = fid.create_dataset(group_path, data=F_out.astype(np.float64))
    ds.attrs["columns"] = "f1_dv,f2_unrecovered_value,f3_vehicles"

print(f"  saved → {args.h5_file}::{group_path}")
