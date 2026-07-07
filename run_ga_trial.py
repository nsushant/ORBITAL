"""
run_ga_trial.py — run NSGA-III, MOEA-D, and PSO on one (scenario, trial) pair.
CLI: python run_ga_trial.py <scenario> <trial>

Reads:  outputs/exp_demands/{scenario}_{trial:02d}.jld2
        outputs/exp_demands/{scenario}_{trial:02d}_greedy.json
Writes: outputs/exp_results/nsga3_{scenario}_{trial:02d}.csv
        outputs/exp_results/moead_{scenario}_{trial:02d}.csv
        outputs/exp_results/pso_{scenario}_{trial:02d}.csv

Objective column order in all CSVs: f1_dv, f2_unrecovered_value, f3_vehicles
(pymoo returns [dv, vehicles, unrecovered_value]; we reorder to [dv, unrecovered, vehicles])
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
                    help="directory containing demand JLD2 files")
parser.add_argument("--result-dir", default="outputs/exp_results",
                    help="directory to write result CSVs")
parser.add_argument("--dv-budget", type=float, default=5000.0,
                    help="ΔV budget per vehicle per sortie [m/s]")
parser.add_argument("--algos", nargs="+", default=["nsga3", "pso"],
                    help="Which algorithms to run (nsga3, moead, pso)")
args = parser.parse_args()

scenario  = args.key
trial     = args.trial
trial_str = f"{trial:02d}"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from loaders import load_sim_name_map, load_demands, load_cost_table_jld2, load_min_tof_table
from pymoo_stuff import OOSProblem, OOSRepair, OOSCrossover, OOSMutation, MOPSO_CD_Repair
from greedy_init import load_greedy_from_json, GreedySampling

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

EXP_DIR = args.demand_dir
RES_DIR = args.result_dir
os.makedirs(RES_DIR, exist_ok=True)

dem_path    = os.path.join(EXP_DIR, f"{scenario}_{trial_str}.jld2")
greedy_path = os.path.join(EXP_DIR, f"{scenario}_{trial_str}_greedy.json")

if not os.path.exists(dem_path):
    sys.exit(f"Demand file not found: {dem_path} — run generate_experiment_demands.jl first")

print(f"\n{'='*60}")
print(f"  scenario={scenario}  trial={trial}")
print(f"{'='*60}")

SIM_FILE     = "outputs/simulation.h5"
COST_FILE    = "outputs/cost_table.jld2"
REFUEL_TIME  = 0.5   # days — must match MDLS (run_mdls_trial.jl)

d             = load_demands(dem_path)
ct, ct_meta   = load_cost_table_jld2(COST_FILE)
nm            = load_sim_name_map(SIM_FILE, depot_idx=ct_meta["depot_idx"])
min_tof_table = load_min_tof_table()

sat_ids  = [nm[s] for s in d["sat_identifiers"]]
depot_id = nm["depot_1"]
Ndems    = len(d["demand_deadlines"])
maxV     = 25

print(f"Demands loaded: {Ndems} demands")

problem = OOSProblem(
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
    min_tof_table = min_tof_table,
    refuel_time   = REFUEL_TIME,
    dv_budget     = args.dv_budget,
)

# Greedy warm start
if os.path.exists(greedy_path):
    greedy_x = load_greedy_from_json(greedy_path, Ndems)
    n_served = int((greedy_x[Ndems:2*Ndems] > 0).sum())
    print(f"Greedy loaded: {n_served} served, {Ndems - n_served} unserved")
else:
    print(f"WARNING: greedy file not found ({greedy_path}), using random init")
    greedy_x = None

# ---------------------------------------------------------------------------
# Helper — extract front from result, reorder columns, save CSV
# Pymoo objective order: F[:,0]=dv  F[:,1]=vehicles  F[:,2]=unserved_time
# CSV column order:      f1_dv      f2_unassigned_time  f3_vehicles
# ---------------------------------------------------------------------------

PENALTY = 1e6

def save_front(res, algo_name):
    outpath = os.path.join(RES_DIR, f"{algo_name}_{scenario}_{trial_str}.csv")

    if res.F is None:
        print(f"  {algo_name}: no feasible solutions")
        return
    F = res.F

    # Filter penalty solutions
    mask = F[:, 0] < PENALTY
    F    = F[mask]
    if len(F) == 0:
        print(f"  {algo_name}: all solutions penalised — skipping")
        return

    # Reorder: [dv, vehicles, unrecovered] → [dv, unrecovered, vehicles]
    F_out = F[:, [0, 2, 1]]
    pd.DataFrame(F_out, columns=["f1_dv", "f2_unrecovered_value", "f3_vehicles"]).to_csv(
        outpath, index=False
    )
    print(f"  {algo_name}: {len(F_out)} front points → {outpath}")

# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressCallback(Callback):
    def __init__(self, algo_name):
        super().__init__()
        self.algo_name = algo_name

    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % 10 == 0 or gen == 1:
            n_eval = algorithm.evaluator.n_eval
            if hasattr(algorithm, "pareto_archive"):
                extra = f"archive {len(algorithm.pareto_archive):3d}"
            else:
                cv = algorithm.pop.get("CV")
                extra = f"n_feasible {int((cv <= 0).sum()):3d}/{len(cv)}"
            print(f"  [{self.algo_name}] gen {gen:4d} | n_eval {n_eval:6d} | {extra}")

# ---------------------------------------------------------------------------
# Shared ref dirs (NSGA-III + PSO) and MOEA-D-specific ref dirs
# ---------------------------------------------------------------------------

import time
ref_dirs       = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)  # 91 vectors
ref_dirs_moead = get_reference_directions("das-dennis", n_dim=3, n_partitions=15)  # 136 vectors

# ---------------------------------------------------------------------------
# Run NSGA-III
# ---------------------------------------------------------------------------

if "nsga3" in args.algos:
    print("\n--- NSGA-III ---")
    sampling_nsga3 = GreedySampling(greedy_x) if greedy_x is not None else None
    algo_nsga3 = NSGA3(
        ref_dirs  = ref_dirs,
        pop_size  = len(ref_dirs),
        sampling  = sampling_nsga3,
        crossover = OOSCrossover(eta=20),
        mutation  = OOSMutation(eta=20),
        repair    = OOSRepair(stochastic=True),
    )
    t0 = time.time()
    res_nsga3 = minimize(problem, algo_nsga3, termination=("n_eval", 10_000),
                         seed=trial, verbose=False, callback=ProgressCallback("nsga3"))
    print(f"  done in {time.time()-t0:.1f}s")
    save_front(res_nsga3, "nsga3")

# ---------------------------------------------------------------------------
# Run MOEA-D
# ---------------------------------------------------------------------------

if "moead" in args.algos:
    print("\n--- MOEA-D ---")
    sampling_moead = GreedySampling(greedy_x) if greedy_x is not None else None
    algo_moead = MOEAD(
        ref_dirs      = ref_dirs_moead,
        n_neighbors   = 20,
        sampling      = sampling_moead,
        crossover     = OOSCrossover(eta=20),
        mutation      = OOSMutation(eta=20),
        repair        = OOSRepair(stochastic=True),
        normalize     = True,
        decomposition = PBI(theta=5.0),
    )
    t0 = time.time()
    res_moead = minimize(problem, algo_moead, termination=("n_eval", 10_000),
                         seed=trial, verbose=False, callback=ProgressCallback("moead"))
    print(f"  done in {time.time()-t0:.1f}s")
    save_front(res_moead, "moead")

# ---------------------------------------------------------------------------
# Run PSO (MOPSO-CD)
# ---------------------------------------------------------------------------

if "pso" in args.algos:
    print("\n--- PSO (MOPSO-CD) ---")
    sampling_pso = GreedySampling(greedy_x) if greedy_x is not None else None
    algo_pso = MOPSO_CD_Repair(
        repair       = OOSRepair(stochastic=True),
        pop_size     = len(ref_dirs),
        archive_size = len(ref_dirs) * 2,
        sampling     = sampling_pso,
    )
    t0 = time.time()
    res_pso = minimize(problem, algo_pso, termination=("n_eval", 10_000),
                       seed=trial, verbose=False, callback=ProgressCallback("pso"))
    print(f"  done in {time.time()-t0:.1f}s")
    save_front(res_pso, "pso")

print(f"\nAll GAs done for {scenario} trial {trial}.")
