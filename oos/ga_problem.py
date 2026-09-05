"""The pymoo `Problem` that wraps the shared evaluator (D21).

D21 (PAPER_COMPLETION_PLAN.md) settled on a single encoding for the NSGA-II
comparator: the "time-inclusive" encoding of MAIN_DOCUMENT_V0.tex Sec. 3.8,
`2N` genes -- N demand genes (vehicle assignment + arrival time) and N depot
genes (vehicle assignment + arrival time), both decoded as
`frac(x[i]) * T_horizon`. Sec. 2 of the paper lists both arrival time and
depot-visit placement as decision variables the planner determines, not
quantities a decoder should infer, so this is the only one of the two
encodings the paper describes (Sec. 3.7 vs 3.8) where a genuinely infeasible
schedule is even expressible -- which is the precondition for pymoo's
constraint handling to have anything to do.

What changed relative to the retired `OOSProblemRK_OT` (kept in
pymoo_stuff.py for reference, no longer used): that decoder treated the
gene's arrival time as a lower bound and searched forward on the grid for
the first feasible, budget-respecting slot, silently dropping (`served=0`)
whatever it could not place, and ran a slice of MDLS's own arrival-shifting
local search (`_opt_times_pass`) inside the decode step itself, before
evaluation. Both defeat the point of switching to real constraint handling:
the first repairs infeasibility away before pymoo ever sees it (the same
failure mode F19 found in NSGA-III, just via search instead of a named
repair class), and the second hands NSGA-II a piece of MDLS's own algorithm.
Neither happens here. Genes are decoded literally -- no search, no
repair -- and a leg whose cost comes back NaN (no table entry, or a
requested time of flight the grid cannot support) is still recorded as
flown. Its violation feeds pymoo's constraint vector instead of being
filtered out beforehand.

A depot gene may land on a vehicle with nothing scheduled after it, or
immediately beside another depot visit. It is still flown: the wasted
delta-V lands in f1 and discourages it on its own, rather than a decoder
rule quietly deciding on the GA's behalf which depot visits are worth
taking.

The three constraints exposed via `out["G"]` are exactly D19's: g1 a leg
with no table entry, g2 a release/deadline window violation, g3 delta-V
budget overspend between depot visits. Every number in `out["F"]` and
`out["G"]` comes from `oos.schedule.evaluate()` -- the same function MDLS's
operators score against -- so this file's only job is turning one genome
into a `Schedule`, not scoring it. That is deliberate: it is the one
function call that makes "MDLS and NSGA-II solve the same problem" true
rather than merely intended.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

from .schedule import DEPOT_UID, Schedule, evaluate, leg


class ScheduleProblem(Problem):
    """2N time-inclusive encoding, decoded straight into a `Schedule`.

    x[0:N]  demand genes: floor = vehicle (0 = unserved), frac = arrival / T_horizon
    x[N:2N] depot genes:  floor = vehicle (0 = inactive), frac = arrival / T_horizon

    T_horizon = max(deadlines) - max(service_times), per Sec. 3.8 of the
    paper. Arrival times are taken literally: no forward search, no repair.
    Feasibility is pymoo's job now, via `out["G"]`, not the decoder's.
    """

    def __init__(self, cost_table, demands, depot_node, max_vehicles,
                 refuel_time, **kwargs):
        self.ct           = cost_table
        self.dem          = demands
        self.depot_node   = int(depot_node)
        self.max_vehicles = int(max_vehicles)
        self.refuel_time  = float(refuel_time)

        N = len(demands)
        self.n_dem = N
        # Sec. 3.8 defines T_horizon as max(deadlines) - max(service times), but
        # a gene mapping past the last arrival the table can express is dead
        # range: every leg to it comes back as g1 no matter what the rest of the
        # genome does. Clamping to the table's own ceiling spends the whole
        # [0,1) fraction on arrivals the cost table can actually answer for.
        ceiling = float(cost_table.dep_days[-1] + cost_table.tof_days[-1])
        self.T_horizon = min(
            float(np.max(demands.deadline) - np.max(demands.service)), ceiling)
        if not np.isfinite(self.T_horizon) or self.T_horizon <= 0:
            raise ValueError(
                "T_horizon = max(deadlines) - max(service_times) must be "
                f"positive and finite; got {self.T_horizon}. Check the "
                "demand file -- a single demand whose service time equals "
                "its deadline, or an empty demand set, both produce this.")

        xl = np.zeros(2 * N)
        # floor(x) is the vehicle and the fraction is the arrival, so the bound
        # has to sit just below max_vehicles + 1: at exactly max_vehicles the
        # last vehicle is reachable only with a zero fraction, i.e. only ever
        # arriving at t=0, and any gene clipped to the bound loses its arrival
        # entirely. OOSProblemRK had this right (maxV + 1 - 1e-9);
        # OOSProblemRK_OT did not, and this inherited the mistake.
        xu = np.full(2 * N, float(self.max_vehicles) + 1.0 - 1e-9)
        super().__init__(n_var=2 * N, n_obj=3, n_constr=3, xl=xl, xu=xu, **kwargs)

    def decode(self, x):
        """One genome -> one `Schedule`. Pure decode: no search, no repair.

        Visit order within a vehicle is the sort order of its genes'
        arrival times, per Sec. 3.7/3.8 ("visit order ... derived from
        sorted arrival times -- no separate order variable needed"). Each
        leg's cost is looked up once, literally, from the running state
        (previous node, previous departure) to (this node, this gene's
        arrival). Each visit records the arrival the *table* says that leg
        achieves, not the one the gene asked for: the gene sets the visit
        order and the request, and snapping (D20) decides the rest.
        Recording the request instead makes a schedule claim an arrival
        that no leg in it produces, which is how a feasible schedule turned
        infeasible merely by being round-tripped through the genome (F22) --
        the same disagreement about when a vehicle arrives that F7 found
        between the two published stacks. A NaN cost (g1) does not stop the
        chain: there is no table answer to record, so the request stands and
        the next leg still has a departure to chain from, with g1 carrying
        the penalty for the impossible leg.
        """
        N = self.n_dem
        vehicle_d   = np.floor(x[:N]).astype(np.int64)
        frac_d      = x[:N] - vehicle_d
        vehicle_dep = np.floor(x[N:]).astype(np.int64)
        frac_dep    = x[N:] - vehicle_dep

        arrival_d   = frac_d * self.T_horizon
        arrival_dep = frac_dep * self.T_horizon

        dv, ph, dg, tg = self.ct.dv, self.ct.phasing, self.ct.dep_days, self.ct.tof_days

        veh_start = [0]
        node_l, uid_l, arrival_l, departure_l, cost_l = [], [], [], [], []

        for v in range(1, self.max_vehicles + 1):
            dem_idx = np.where(vehicle_d == v)[0]
            dep_idx = np.where(vehicle_dep == v)[0]

            sequence = [(float(arrival_d[i]), "D", int(i)) for i in dem_idx]
            sequence += [(float(arrival_dep[j]), "R", int(j)) for j in dep_idx]
            sequence.sort(key=lambda t: t[0])

            state_node = self.depot_node
            state_dep  = 0.0

            for t_arr, kind, idx in sequence:
                target = int(self.dem.node[idx]) if kind == "D" else self.depot_node
                leg_dv, true_arr, ok = leg(dv, ph, dg, tg, state_node, target,
                                           state_dep, t_arr - state_dep)
                arrival = true_arr if ok else t_arr

                node_l.append(target)
                arrival_l.append(arrival)
                cost_l.append(leg_dv if ok else np.nan)

                if kind == "D":
                    uid_l.append(idx)
                    t_dep = arrival + float(self.dem.service[idx])
                else:
                    uid_l.append(DEPOT_UID)
                    t_dep = arrival + self.refuel_time
                departure_l.append(t_dep)

                state_node, state_dep = target, t_dep

            veh_start.append(len(node_l))

        return Schedule(
            veh_start=np.array(veh_start, dtype=np.int64),
            node=np.array(node_l, dtype=np.int64),
            uid=np.array(uid_l, dtype=np.int64),
            arrival=np.array(arrival_l, dtype=np.float64),
            departure=np.array(departure_l, dtype=np.float64),
            cost=np.array(cost_l, dtype=np.float64),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop = len(X)
        F = np.empty((pop, self.n_obj))
        G = np.empty((pop, self.n_constr))
        for k in range(pop):
            sched = self.decode(X[k])
            f, g = evaluate(sched, self.dem, self.ct.dv_budget)
            F[k] = f
            G[k] = g
        out["F"] = F
        out["G"] = G


class GreedySeeding(Sampling):
    """Seed the population with the constructive heuristic, then fill randomly.

    The first individuals are Sec. 3.6 schedules built at a spread of fleet
    caps (`oos.greedy.greedy_restarts`); the rest are uniform. A single seed
    is not enough, and the reason is measured rather than assumed: with one,
    NSGA-II reached full feasibility by generation 40 but held only 1-7
    distinct objective vectors and never reduced the fleet below the cap,
    because every survivor descended from the one feasible individual it
    started with. MDLS is initialised from the same set, so neither method
    begins from a better position than the other.

    This is what makes the comparison viable at all rather than a nicety. A
    uniform genome on the real instance flies 773 visits of which 760 have no
    cost-table entry, with some 36,000 days of window violation: NSGA-II
    starting cold has nothing feasible to select on, so constrained domination
    spends its early generations ranking degrees of impossibility. The seed is
    feasible, so there is a feasible point in the population from generation
    one, and MDLS starts from the same schedule.

    Replaces `GreedySamplingRK2` in greedy_init.py, which encoded its
    fractional gene as a position in the visit order while this decoder reads
    that same fraction as arrival / T_horizon -- seeding with it would have
    put order indices where arrival times belong.
    """

    def __init__(self, schedules=None, n_seed=18):
        super().__init__()
        self.schedules = schedules
        self.n_seed = n_seed

    def _do(self, problem, n_samples, **kwargs):
        from .greedy import encode_for_ga, greedy_restarts

        X = np.random.uniform(problem.xl, problem.xu, (n_samples, problem.n_var))
        seeds = self.schedules
        if seeds is None:
            seeds = greedy_restarts(problem.ct, problem.dem, problem.depot_node,
                                    problem.max_vehicles, problem.refuel_time,
                                    n=self.n_seed,
                                    dv_budget=problem.ct.dv_budget)
        for i, s in enumerate(seeds[:n_samples]):
            X[i] = encode_for_ga(s, problem.dem, problem)
        return X
