from loaders import load_sim_name_map, load_demands, load_cost_table, load_min_tof_table
from pymoo_stuff import OOSProblem, OOSRepair
from greedy_init import load_greedy_from_json, GreedySampling

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

import glob, os

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

_demand_files = sorted(glob.glob("outputs/demands/*.jld2"))
DEMAND_FILE   = _demand_files[-1]
print(f"Using demand file: {os.path.basename(DEMAND_FILE)}")

SIM_FILE  = "outputs/simulation.h5"
COST_FILE = "outputs/cost_table_basic.h5"

d             = load_demands(DEMAND_FILE)
ct, ct_meta   = load_cost_table(COST_FILE)
nm            = load_sim_name_map(SIM_FILE, depot_idx=ct_meta["depot_idx"])
min_tof_table = load_min_tof_table()

sat_ids  = [nm[s] for s in d["sat_identifiers"]]
depot_id = nm["depot_1"]
Ndems    = len(d["demand_deadlines"])
maxV     = 25   # match MDLS nvehicles=20, allow up to 25

# ---------------------------------------------------------------------------
# Build problem
# ---------------------------------------------------------------------------

problem = OOSProblem(
    Ndems         = Ndems,
    maxV          = maxV,
    deadlines     = d["demand_deadlines"],
    service_times = d["service_times"],
    sat_ids       = sat_ids,
    depot_id      = depot_id,
    cost_table    = ct,
    dep_grid      = ct_meta["dep_grid"],
    arr_grid      = ct_meta["arr_grid"],
    min_tof_table = min_tof_table,
)

# ---------------------------------------------------------------------------
# Greedy initial solution (loaded from Julia-generated JSON)
# ---------------------------------------------------------------------------

greedy_x = load_greedy_from_json("outputs/greedy_init.json", Ndems)
n_served = int((greedy_x[Ndems:2*Ndems] > 0).sum())
print(f"Greedy (from JSON): {n_served} served, {Ndems - n_served} unserved")

# ---------------------------------------------------------------------------
# NSGA-III
# ---------------------------------------------------------------------------

class ProgressCallback(Callback):
    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % 10 == 0 or gen == 1:
            cv  = algorithm.pop.get("CV")
            nds = int((cv <= 0).sum())
            print(f"gen {gen:4d} | n_eval {algorithm.evaluator.n_eval:6d} | "
                  f"n_feasible {nds:3d}/{len(cv)} | cv_min {cv.min():.4f}")

ref_dirs  = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)

algorithm = NSGA3(
    ref_dirs = ref_dirs,
    pop_size = len(ref_dirs),
    sampling = GreedySampling(greedy_x),
    repair   = OOSRepair(),
)

res = minimize(
    problem,
    algorithm,
    termination = ("n_eval", 10_000),
    seed        = 42,
    verbose     = False,
    callback    = ProgressCallback(),
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

import numpy as _np, pandas as _pd

if res.F is None:
    print("\nNo feasible solutions found.")
    print(f"Best CV: {res.pop.get('CV').min():.4f}")
else:
    print(f"\nPareto front size: {len(res.F)}")
    print(f"Objectives (dv [m/s], n_vehicles, unserved_time [days]):\n{res.F}")
    # Save Pareto front for comparison plotting
    _pd.DataFrame(res.F, columns=["f1_dv", "f3_vehicles", "f2_unassigned_time"]).to_csv(
        "outputs/nsga3_pareto_comparative.csv", index=False
    )
    print("Saved outputs/nsga3_pareto_comparative.csv")
