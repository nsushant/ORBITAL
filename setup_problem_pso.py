from loaders import load_sim_name_map, load_demands, load_cost_table, load_min_tof_table
from pymoo_stuff import OOSProblem, OOSRepair
from greedy_init import load_greedy_from_json, GreedySampling

from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.ref_dirs import get_reference_directions

import numpy as np
import glob, os

# ---------------------------------------------------------------------------
# MOPSO implementation
# ---------------------------------------------------------------------------

class MOPSO(Algorithm):
    """
    Multi-Objective Particle Swarm Optimisation.

    Maintains an external Pareto archive from which global bests are drawn.
    Personal bests updated whenever a particle finds a solution that
    dominates its current personal best.
    """

    def __init__(self, pop_size=91, w=0.7, c1=1.5, c2=1.5,
                 archive_size=None, sampling=None, repair=None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size    = pop_size
        self.w           = w               # inertia weight
        self.c1          = c1              # cognitive (personal best) weight
        self.c2          = c2              # social (archive best) weight
        self.pareto_archive_size = archive_size or pop_size
        self.sampling    = sampling
        self.repair      = repair

        self.velocities       = None
        self.pbest_X          = None
        self.pbest_F          = None
        self.pareto_archive          = []         # list of (X, F) tuples

    # ── initialisation ───────────────────────────────────────────────────────

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

    # ── per-generation step ───────────────────────────────────────────────────

    def _infill(self):
        X = self.pop.get("X")
        n, d = X.shape

        # Sample global bests from archive
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

        # Update personal bests
        for i in range(self.pop_size):
            if self._dominates(F_new[i], self.pbest_F[i]):
                self.pbest_X[i] = X_new[i].copy()
                self.pbest_F[i] = F_new[i].copy()

        self.pop = infills
        self._update_archive(X_new, F_new)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def _update_archive(self, X, F):
        # Add new candidates
        for i in range(len(X)):
            self.pareto_archive.append((X[i].copy(), F[i].copy()))

        # Keep non-dominated set
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

        # Cap archive size — prune by crowding in objective space if over limit
        if len(self.pareto_archive) > self.pareto_archive_size:
            self.pareto_archive = self.pareto_archive[-self.pareto_archive_size:]

    # ── result extraction ─────────────────────────────────────────────────────

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
maxV     = 25

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
# Greedy initial solution
# ---------------------------------------------------------------------------

greedy_x = load_greedy_from_json("outputs/greedy_init.json", Ndems)
n_served = int((greedy_x[Ndems:2*Ndems] > 0).sum())
print(f"Greedy (from JSON): {n_served} served, {Ndems - n_served} unserved")

# ---------------------------------------------------------------------------
# MOPSO
# ---------------------------------------------------------------------------

class ProgressCallback(Callback):
    def notify(self, algorithm):
        gen = algorithm.n_gen
        if gen % 10 == 0 or gen == 1:
            n_arc = len(algorithm.pareto_archive)
            print(f"gen {gen:4d} | n_eval {algorithm.evaluator.n_eval:6d} | "
                  f"archive {n_arc:3d}")

pop_size = get_reference_directions("das-dennis", n_dim=3, n_partitions=12).shape[0]

algorithm = MOPSO(
    pop_size     = pop_size,
    w            = 0.7,
    c1           = 1.5,
    c2           = 1.5,
    archive_size = pop_size * 2,
    sampling     = GreedySampling(greedy_x),
    repair       = OOSRepair(),
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

import pandas as _pd

PENALTY = 1e6

final_algo = res.algorithm if hasattr(res, "algorithm") and res.algorithm is not None else algorithm

if not final_algo.pareto_archive:
    print("\nPareto archive is empty — no results to save.")
else:
    arc_F = np.array([a[1] for a in final_algo.pareto_archive])
    if arc_F.ndim == 1:
        arc_F = arc_F.reshape(1, -1)
    mask  = arc_F[:, 0] < PENALTY
    arc_F = arc_F[mask]

    print(f"\nPareto archive size: {len(arc_F)} valid points")
    if len(arc_F) > 0:
        _pd.DataFrame(arc_F, columns=["f1_dv", "f3_vehicles", "f2_unassigned_time"]).to_csv(
            "outputs/pso_pareto_comparative.csv", index=False
        )
        print("Saved outputs/pso_pareto_comparative.csv")
