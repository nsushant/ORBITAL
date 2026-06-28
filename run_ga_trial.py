"""
run_ga_trial.py — run NSGA-III, MOEA-D, and PSO on one (scenario, trial) pair.
CLI: python run_ga_trial.py <scenario> <trial>

Reads:  outputs/exp_demands/{scenario}_{trial:02d}.jld2
        outputs/exp_demands/{scenario}_{trial:02d}_greedy.json
Writes: outputs/exp_results/nsga3_{scenario}_{trial:02d}.csv
        outputs/exp_results/moead_{scenario}_{trial:02d}.csv
        outputs/exp_results/pso_{scenario}_{trial:02d}.csv

Objective column order in all CSVs: f1_dv, f2_unassigned_time, f3_vehicles
(pymoo returns [dv, vehicles, unserved_time]; we reorder to [dv, unserved, vehicles])
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
args = parser.parse_args()

scenario  = args.key
trial     = args.trial
trial_str = f"{trial:02d}"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from loaders import load_sim_name_map, load_demands, load_cost_table, load_min_tof_table
from pymoo_stuff import OOSProblem, OOSRepair
from greedy_init import load_greedy_from_json, GreedySampling

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population

# ---------------------------------------------------------------------------
# MOPSO (copied from setup_problem_pso.py)
# ---------------------------------------------------------------------------

class MOPSO(Algorithm):
    def __init__(self, pop_size=91, w=0.7, c1=1.5, c2=1.5,
                 archive_size=None, sampling=None, repair=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size             = pop_size
        self.w                    = w
        self.c1                   = c1
        self.c2                   = c2
        self.pareto_archive_size  = archive_size or pop_size
        self.sampling             = sampling
        self.repair               = repair
        self.velocities           = None
        self.pbest_X              = None
        self.pbest_F              = None
        self.pareto_archive       = []

    def _initialize_infill(self):
        if self.sampling is not None:
            X = self.sampling._do(self.problem, self.pop_size)
        else:
            X = np.random.uniform(self.problem.xl, self.problem.xu,
                                  (self.pop_size, self.problem.n_var))
        pop = Population.new(X=X)
        if self.repair is not None:
            pop = self.repair.do(self.problem, pop)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        X = self.pop.get("X")
        F = self.pop.get("F")
        v_range = (self.problem.xu - self.problem.xl) * 0.1
        self.velocities = np.random.uniform(-v_range, v_range,
                                            (self.pop_size, self.problem.n_var))
        self.pbest_X = X.copy()
        self.pbest_F = F.copy()
        self._update_archive(X, F)

    def _infill(self):
        X = self.pop.get("X")
        n, d = X.shape
        arc_X = np.array([a[0] for a in self.pareto_archive])
        gidx  = np.random.randint(0, len(arc_X), n)
        gbest = arc_X[gidx]
        r1 = np.random.rand(n, d)
        r2 = np.random.rand(n, d)
        self.velocities = (self.w  * self.velocities
                         + self.c1 * r1 * (self.pbest_X - X)
                         + self.c2 * r2 * (gbest - X))
        X_new = np.clip(X + self.velocities, self.problem.xl, self.problem.xu)
        off = Population.new(X=X_new)
        if self.repair is not None:
            off = self.repair.do(self.problem, off)
        return off

    def _advance(self, infills=None, **kwargs):
        F_new = infills.get("F")
        X_new = infills.get("X")
        for i in range(self.pop_size):
            if self._dominates(F_new[i], self.pbest_F[i]):
                self.pbest_X[i] = X_new[i].copy()
                self.pbest_F[i] = F_new[i].copy()
        self.pop = infills
        self._update_archive(X_new, F_new)

    @staticmethod
    def _dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def _update_archive(self, X, F):
        for i in range(len(X)):
            self.pareto_archive.append((X[i].copy(), F[i].copy()))
        n = len(self.pareto_archive)
        dominated = np.zeros(n, dtype=bool)
        Fa = np.array([a[1] for a in self.pareto_archive])
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i != j and not dominated[j] and self._dominates(Fa[j], Fa[i]):
                    dominated[i] = True
                    break
        self.pareto_archive = [self.pareto_archive[i] for i in range(n) if not dominated[i]]
        if len(self.pareto_archive) > self.pareto_archive_size:
            self.pareto_archive = self.pareto_archive[-self.pareto_archive_size:]

    def _set_optimum(self, **kwargs):
        if self.pareto_archive:
            arc_X = np.array([a[0] for a in self.pareto_archive])
            arc_F = np.array([a[1] for a in self.pareto_archive])
            self.opt = Population.new(X=arc_X, F=arc_F)
        else:
            self.opt = self.pop

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

SIM_FILE  = "outputs/simulation.h5"
COST_FILE = "outputs/cost_table_basic.h5"

d             = load_demands(dem_path)
ct, ct_meta   = load_cost_table(COST_FILE)
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
    sat_ids       = sat_ids,
    depot_id      = depot_id,
    cost_table    = ct,
    dep_grid      = ct_meta["dep_grid"],
    arr_grid      = ct_meta["arr_grid"],
    min_tof_table = min_tof_table,
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

def save_front(res_or_arc, algo_name):
    outpath = os.path.join(RES_DIR, f"{algo_name}_{scenario}_{trial_str}.csv")

    if isinstance(res_or_arc, list):
        # PSO pareto_archive: list of (X, F) tuples
        if not res_or_arc:
            print(f"  {algo_name}: empty archive — skipping")
            return
        F = np.array([a[1] for a in res_or_arc])
        if F.ndim == 1:
            F = F.reshape(1, -1)
    else:
        # Standard pymoo result
        if res_or_arc.F is None:
            print(f"  {algo_name}: no feasible solutions")
            return
        F = res_or_arc.F

    # Filter penalty solutions
    mask = F[:, 0] < PENALTY
    F    = F[mask]
    if len(F) == 0:
        print(f"  {algo_name}: all solutions penalised — skipping")
        return

    # Reorder: [dv, vehicles, unserved] → [dv, unserved, vehicles]
    F_out = F[:, [0, 2, 1]]
    pd.DataFrame(F_out, columns=["f1_dv", "f2_unassigned_time", "f3_vehicles"]).to_csv(
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
        if gen % 20 == 0 or gen == 1:
            n_eval = algorithm.evaluator.n_eval
            if hasattr(algorithm, "pareto_archive"):
                extra = f"archive {len(algorithm.pareto_archive):3d}"
            else:
                cv = algorithm.pop.get("CV")
                extra = f"n_feasible {int((cv <= 0).sum()):3d}/{len(cv)}"
            print(f"  [{self.algo_name}] gen {gen:4d} | n_eval {n_eval:6d} | {extra}")

# ---------------------------------------------------------------------------
# Run NSGA-III
# ---------------------------------------------------------------------------

print("\n--- NSGA-III ---")
ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)
import time

sampling_nsga3 = GreedySampling(greedy_x) if greedy_x is not None else None
algo_nsga3 = NSGA3(
    ref_dirs = ref_dirs,
    pop_size = len(ref_dirs),
    sampling = sampling_nsga3,
    repair   = OOSRepair(),
)
t0 = time.time()
res_nsga3 = minimize(problem, algo_nsga3, termination=("n_eval", 10_000),
                     seed=trial, verbose=False, callback=ProgressCallback("nsga3"))
print(f"  done in {time.time()-t0:.1f}s")
save_front(res_nsga3, "nsga3")

# ---------------------------------------------------------------------------
# Run MOEA-D
# ---------------------------------------------------------------------------

print("\n--- MOEA-D ---")
sampling_moead = GreedySampling(greedy_x) if greedy_x is not None else None
algo_moead = MOEAD(
    ref_dirs    = ref_dirs,
    n_neighbors = 15,
    sampling    = sampling_moead,
    repair      = OOSRepair(),
)
t0 = time.time()
res_moead = minimize(problem, algo_moead, termination=("n_eval", 10_000),
                     seed=trial, verbose=False, callback=ProgressCallback("moead"))
print(f"  done in {time.time()-t0:.1f}s")
save_front(res_moead, "moead")

# ---------------------------------------------------------------------------
# Run PSO
# ---------------------------------------------------------------------------

print("\n--- PSO ---")
pop_size = len(ref_dirs)
sampling_pso = GreedySampling(greedy_x) if greedy_x is not None else None
algo_pso = MOPSO(
    pop_size     = pop_size,
    w            = 0.7,
    c1           = 1.5,
    c2           = 1.5,
    archive_size = pop_size * 2,
    sampling     = sampling_pso,
    repair       = OOSRepair(),
)
t0 = time.time()
res_pso = minimize(problem, algo_pso, termination=("n_eval", 10_000),
                   seed=trial, verbose=False, callback=ProgressCallback("pso"))
print(f"  done in {time.time()-t0:.1f}s")
final_algo = res_pso.algorithm if (hasattr(res_pso, "algorithm") and res_pso.algorithm is not None) else algo_pso
save_front(final_algo.pareto_archive, "pso")

print(f"\nAll GAs done for {scenario} trial {trial}.")
