"""Why does NSGA-II's front collapse to copies of the seed?

The Sec. 4 run produced fronts of 91 points carrying a single distinct
objective vector. That is not a front; it is one solution reported many times.
This script establishes which of two things is happening:

  (a) NSGA-II never produces a *second* feasible individual, so constrained
      domination correctly drives the whole population onto the one it was
      given -- in which case the comparison is measuring whether the encoding
      can maintain feasibility under SBX and polynomial mutation, not whether
      population search beats directed local search;

  (b) it produces feasible individuals but they are all objective-equivalent
      to the seed, which would be a different and less damaging story.

It also measures what a larger budget and a broader seeding would buy, since
those are the two levers available if (a) turns out to be the answer.

Run:  python diagnose_nsga2.py [scenario] [trial]
"""

import sys

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from oos.demands import load_demands
from oos.ga_problem import GreedySeeding, ScheduleProblem
from oos.greedy import encode_for_ga, greedy_schedule
from oos.schedule import load_cost_table

scen = sys.argv[1] if len(sys.argv) > 1 else "S2_refuel"
trial = int(sys.argv[2]) if len(sys.argv) > 2 else 1

ct = load_cost_table("outputs/cost_table.h5")
depot = ct.names.index("depot_1")
dem = load_demands(f"outputs/exp_demands/{scen}_{trial:02d}.h5")
prob = ScheduleProblem(ct, dem, depot_node=depot, max_vehicles=25, refuel_time=0.5)
print(f"{scen} trial {trial}: {len(dem)} demands, {prob.n_var} genes\n")


class Trace(Callback):
    """Report feasibility, not just objectives -- that is the question here."""

    def notify(self, algo):
        cv = np.asarray(algo.pop.get("CV")).ravel()
        F = np.asarray(algo.pop.get("F"))
        n_feas = int((cv <= 0).sum())
        uniq = len(np.unique(F[cv <= 0], axis=0)) if n_feas else 0
        if algo.n_gen % 20 == 0 or algo.n_gen == 1:
            print(f"   gen {algo.n_gen:4d}  feasible {n_feas:3d}/{len(cv)}  "
                  f"distinct feasible objectives {uniq:3d}  "
                  f"median CV {np.median(cv):12.1f}  min CV {cv.min():10.1f}")


def run(label, sampling, n_eval, pop=91):
    print(f"-- {label}")
    res = minimize(prob, NSGA2(pop_size=pop, sampling=sampling,
                               crossover=SBX(eta=20, prob=0.9),
                               mutation=PM(eta=20)),
                   termination=("n_eval", n_eval), seed=trial,
                   verbose=False, callback=Trace())
    cv = np.asarray(res.algorithm.pop.get("CV")).ravel()
    F = np.asarray(res.algorithm.pop.get("F"))
    feas = F[cv <= 0]
    u = np.unique(feas, axis=0) if len(feas) else np.empty((0, 3))
    print(f"   final: {len(feas)} feasible, {len(u)} distinct objective vectors")
    if len(u):
        print(f"   dV {u[:,0].min():.0f}-{u[:,0].max():.0f}   "
              f"fleet {u[:,1].min():.0f}-{u[:,1].max():.0f}   "
              f"unrec ${u[:,2].min()/1e6:.1f}-{u[:,2].max()/1e6:.1f} M")
    print()
    return u


class MultiSeed(GreedySeeding):
    """Seed a fraction of the population with randomised greedy restarts.

    One seed gives crossover nothing feasible to recombine with: every partner
    is a uniform individual carrying hundreds of impossible legs, so every
    child is infeasible and the seed is the only survivor. Several *different*
    feasible schedules give the operators feasible material on both sides.
    The restarts differ by capping the fleet, which changes which demands the
    heuristic reaches and in what order, so the seeds are genuinely distinct
    rather than perturbations of one another.
    """

    def __init__(self, frac=0.2):
        super().__init__()
        self.frac = frac

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.uniform(problem.xl, problem.xu, (n_samples, problem.n_var))
        n_seed = max(1, int(round(self.frac * n_samples)))
        caps = np.linspace(3, problem.max_vehicles, n_seed).astype(int)
        for i, cap in enumerate(caps):
            s, _ = greedy_schedule(problem.ct, problem.dem, problem.depot_node,
                                   int(cap), problem.refuel_time,
                                   dv_budget=problem.ct.dv_budget)
            X[i] = encode_for_ga(s, problem.dem, problem)
        return X


base = run("as run in Sec. 4: one seed, 10,000 evaluations",
           GreedySeeding(), 10_000)
big = run("same seeding, 100,000 evaluations (10x the budget)",
          GreedySeeding(), 100_000)
multi = run("20 % of the population seeded with greedy restarts, 10,000 evaluations",
            MultiSeed(0.2), 10_000)

print("=" * 68)
print("Reading this:")
print("  If 'feasible' stays at 1 throughout the first run, the encoding cannot")
print("  maintain feasibility under SBX and PM, and the Sec. 4 comparison is")
print("  measuring that rather than search strategy. Whether the 10x budget or")
print("  the broader seeding changes it says which remedy is worth taking.")
