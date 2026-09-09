"""Facility-location MILP over every Pareto point of the depot sweep.

Each (a, i) sweep location contributes its *whole* MDLS front, so an opened
depot also chooses *which operating point* of its front it runs. That removes
the "maximum coverage is the most expensive point" caveat of the legacy column
model (`oos.depot_milp`): under a spend cap the solver simply slides each
depot down its front.

Model
-----
    min       sum_d y_d                         (pass 1: fewest depots)
      s.t.    sum_q w_{d,q} = y_d               each open depot runs one point
              z_c <= sum_{(d,q): covers c} w_{d,q}    for every client c
              sum_c value_c (1 - z_c) <= b            lost-value cap (b = loss_usd)
              sum_{d,q} C_{d,q} w_{d,q} <= B          contract spend cap

    pass 2, with sum_d y_d fixed at N*: min total spend (tie-break among
    equal-depot solutions, so the min-depots-vs-B curve is well posed).

Coverage is at the asset level: the sweep's points_covered flags a client only
when *every one of its requests* is served, so value_c puts the whole asset at
stake (`z_c = 0` loses it entirely) and the loss sum never double counts. The
recovered-value floor of the older theta formulation is the same row here:
`sum value_c z_c >= Gamma - b` with b = (1 - theta) * Gamma.

Candidate costs (portfolio / contract cost of operating point (d, q), in US$)
    C_{d,q} = fleet (C_srv + r m_dry_srv)        servicers: manufacture + launch
            + M_prop (C_Xe + r)                  campaign xenon + its launch mass
            + C_dep + r m_dry_dep                depot: manufacture + launch

with propellant from the rocket equation, per vehicle share of total delta-V,
    M_prop = fleet * m_dry_srv * (exp(dv / (fleet v_e)) - 1),  v_e = isp * g0.

The depot therefore stores the propellant its own operating point's fleet will
draw, and the xenon is costed once (its launch kg travels with the depot), so
nothing is double counted. Values default to the caller's agreed numbers
(servicer $15 M, depot $60 M, launch $3 700/kg), all available as flags.

Recovered / lost value basis
-----------------------------
`value_c` is the client's *total request value* (the sweep's client_value
dataset, the currency of f3): the whole asset is at stake, so a client with any
unserved request is counted as lost with its full value. `--value-basis asset`
divides value_c by the client's request count so each demanded asset is
recovered exactly once -- the currency of the `C_infra < P_served` portfolio
breakeven. The lost-value cap is `b = loss_frac * Gamma` of the chosen basis;
`theta = 1 - loss_frac` is the equivalent recovered-value floor.

Candidate pruning
-----------------
A candidate is a (location, cost, covered-client-set) triple. Per the one
point per open depot rule a candidate dominated by a cheaper one at the SAME
location is never needed; and any candidate whose coverage is a subset of a
cheaper candidate's (any location) is globally redundant. Both rows of
dominance are pruned before the model is built, which collapses fronts of
near-duplicate MDLS points into distinct coverage signatures.

Feasibility of the lost-value cap
---------------------------------
b -- the maximum value of assets we are willing to lose -- is bounded below by
Gamma - max_candidate_recovered: a cap below the loss of the best single
operating point is only satisfiable by opening multiple depots, and a cap below
the loss of the union of all candidates is infeasible outright.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G0 = 9.80665


@dataclass
class Costs:
    """Default cost model -- the agreed constants of the design study."""

    c_srv: float = 15.0e6       # $/servicer, manufacturing
    c_dep: float = 60.0e6       # $/depot, manufacturing
    launch_kg: float = 3700.0   # $/kg, Falcon 9 full-rate (2 x $1850 half-rate)
    m_dry_srv: float = 300.0    # kg
    m_dry_dep: float = 300.0    # kg (depot bus; stored propellant added below)
    c_xe: float = 340.0         # $/kg xenon (Tirila et al. 2023)
    isp: float = 1100.0         # s, spacevan-class Hall thruster

    @property
    def ve(self):
        return self.isp * G0

    def propellant_total(self, dv_total, fleet):
        """Campaign propellant [kg] for a front point (dv_total, fleet)."""
        fleet = max(1, int(round(float(fleet))))
        dv = max(0.0, float(dv_total))
        per_veh = self.m_dry_srv * (np.exp(dv / (fleet * self.ve)) - 1.0) \
            if dv > 0.0 else 0.0
        return float(fleet * per_veh)

    def point_cost(self, dv_total, fleet):
        """Contract cost C_{d,q} [US$] to operate a front point."""
        fleet = max(1, int(round(float(fleet))))
        m_prop = self.propellant_total(dv_total, fleet)
        servicers = fleet * (self.c_srv + self.launch_kg * self.m_dry_srv)
        propellant = m_prop * (self.c_xe + self.launch_kg)
        depot = self.c_dep + self.launch_kg * self.m_dry_dep
        return float(servicers + propellant + depot)


def load_fronts(path, value_basis="request", costs=None):
    """Read an extended depot-sweep H5 into flat candidate arrays.

    Returns dict with loc_id (n_pts,), dv, fleet, cost, recovered (n_pts,),
    covers (n_pts, n_client) bool, client_value, total_value, a_km, incl_deg,
    loc_npts, loc_offset, n_pts, n_client.
    """
    import h5py

    with h5py.File(path, "r") as f:
        loc_id = np.repeat(np.arange(f["loc_npts"].shape[0]),
                           f["loc_npts"][:].astype(np.int64))
        a_km = f["a_km"][:]
        incl_deg = f["incl_deg"][:]
        client_value = f["client_value"][:]
        dv = f["points_dv_m_s"][:]
        fleet = f["points_fleet"][:]
        covers = f["points_covered"][:].astype(bool)
        n_req = f["client_n_requests"][:].astype(np.int64) \
            if "client_n_requests" in f else None
        loc_npts = np.asarray(f["loc_npts"][:], dtype=np.int64)
        loc_offset = np.asarray(f["loc_offset"][:], dtype=np.int64)
    if value_basis == "asset":
        if n_req is None:
            raise SystemExit(
                "asset value basis requires client_n_requests in the sweep H5")
        assert (n_req > 0).all(), "clients with requests must have n_req > 0"
        value_for_milp = client_value / n_req
    elif value_basis == "request":
        value_for_milp = client_value
    else:
        raise SystemExit(f"unknown value basis {value_basis!r}")
    total_value = float(value_for_milp.sum())
    covered_val = value_for_milp.dot(covers.T)
    costs = costs or Costs()
    cost = np.asarray([costs.point_cost(dv[ii], fleet[ii])
                       for ii in range(len(dv))])
    return dict(
        loc_id=loc_id, dv=dv, fleet=fleet,
        cost=cost, recovered=covered_val, covers=covers,
        client_value=value_for_milp, total_value=total_value,
        a_km=a_km, incl_deg=incl_deg,
        loc_npts=loc_npts, loc_offset=loc_offset,
        n_pts=len(dv), n_client=covers.shape[1], value_basis=value_basis)


def prune_candidates(loc_id, cost, coversets):
    """Drop operationally and globally dominated candidates.

    loc_id    (n_pts,) int — location of each candidate
    cost      (n_pts,) float — C_{d,q}
    coversets (n_pts,) of frozenset[int] — covered client indices

    Per location: after sorting by cost, drop a point whose set is a subset of
    a cheaper kept point's (same size set keeps the cheaper). Then globally:
    drop any candidate whose set is a subset of a cheaper candidate's. Return
    (kept global indices, locations that still have a kept candidate).
    """
    n = len(loc_id)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    keep = np.zeros(n, dtype=bool)
    for loc in np.unique(loc_id):
        idx = np.flatnonzero(loc_id == loc)
        kept_sets = []
        for r in np.argsort(cost[idx], kind="stable"):
            s = coversets[idx[r]]
            if not any(s.issubset(ks) for ks in kept_sets):
                keep[idx[r]] = True
                kept_sets.append(s)
    kept = np.flatnonzero(keep)
    order = np.argsort(cost[kept], kind="stable")
    global_keep = np.zeros(len(kept), dtype=bool)
    kept_sets = []
    for r in order:
        s = coversets[kept[r]]
        if any(s.issubset(ks) for ks in kept_sets):
            continue
        global_keep[r] = True
        kept_sets.append(s)
    out = kept[np.flatnonzero(global_keep)]
    return out, np.unique(loc_id[out])


def _solve_gurobi(loc_id, cost, recovered, covers, client_value, total_value,
                  loss, budget, group_loss_caps=None, time_limit=120.0,
                  mip_gap=0.001, verbose=False, diagnostics=None):
    """Fewest depots losing < `loss` US$ of client value within `budget` US$
    of contract spend, choosing one front point per open depot.

    Returns (chosen_local_idx, chosen_loc_id, chosen_cost, chosen_recovered,
    recovered_total, n_depots, spend) or None when infeasible.
    """
    import gurobipy as gp
    from gurobipy import GRB

    loc_ids = np.unique(loc_id)
    n_local = len(loc_ids)
    n_pts = len(loc_id)
    n_client = covers.shape[1]
    local_of = {d: i for i, d in enumerate(loc_ids)}

    m = gp.Model("depot_fronts")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap

    y = m.addVars(n_local, vtype=GRB.BINARY, name="y")
    w = m.addVars(n_pts, vtype=GRB.BINARY, name="w")
    z = m.addVars(n_client, vtype=GRB.BINARY, name="z")

    served_by_loc = {d: [] for d in loc_ids}
    for p in range(n_pts):
        served_by_loc[loc_id[p]].append(p)
    for d, plist in served_by_loc.items():
        m.addConstr(gp.quicksum(w[p] for p in plist) == y[local_of[d]],
                    name=f"one_point_{d}")

    for c in range(n_client):
        serving = np.flatnonzero(covers[:, c])
        if serving.size == 0:
            m.addConstr(z[c] == 0, name=f"unreachable_{c}")
        else:
            m.addConstr(z[c] <= gp.quicksum(w[int(p)] for p in serving),
                        name=f"cover_upper_{c}")
            for p in serving:
                m.addConstr(z[c] >= w[int(p)],
                            name=f"cover_lower_{c}_{int(p)}")

    if group_loss_caps:
        for name, mask, cap in group_loss_caps:
            members = np.flatnonzero(mask)
            group_total = float(client_value[members].sum())
            m.addConstr(gp.quicksum(float(client_value[c]) * z[c]
                                    for c in members) >= group_total - cap,
                        name=f"loss_cap_{name}")
    else:
        m.addConstr(gp.quicksum(float(client_value[c]) * z[c]
                                for c in range(n_client)) >= total_value - loss,
                    name="loss_cap")
    m.addConstr(gp.quicksum(float(cost[p]) * w[p]
                            for p in range(n_pts)) <= budget,
                name="contract_budget")

    m.setObjective(gp.quicksum(y[i] for i in range(n_local)), GRB.MINIMIZE)
    m.optimize()
    if diagnostics is not None:
        diagnostics.update(
            solver_backend="gurobi",
            stage1_status=int(m.Status),
            stage1_status_message=str(m.Status),
            stage1_optimal=int(m.Status == GRB.OPTIMAL),
            stage1_objective=float(m.ObjVal) if m.SolCount else np.nan,
            stage1_bound=float(m.ObjBound) if m.SolCount else np.nan,
            stage1_mip_gap=float(m.MIPGap) if m.SolCount else np.nan,
        )
    if m.SolCount == 0:
        return None

    n_dep = int(round(sum(y[i].X for i in range(n_local))))
    m.addConstr(gp.quicksum(y[i] for i in range(n_local)) == n_dep,
                name="depot_count_fixed")
    m.setObjective(gp.quicksum(float(cost[p]) * w[p]
                               for p in range(n_pts)), GRB.MINIMIZE)
    m.Params.TimeLimit = max(1.0, time_limit / 2.0)
    m.optimize()
    if diagnostics is not None:
        diagnostics.update(
            stage2_status=int(m.Status),
            stage2_status_message=str(m.Status),
            stage2_optimal=int(m.Status == GRB.OPTIMAL),
            stage2_objective=float(m.ObjVal) if m.SolCount else np.nan,
            stage2_bound=float(m.ObjBound) if m.SolCount else np.nan,
            stage2_mip_gap=float(m.MIPGap) if m.SolCount else np.nan,
        )
    if m.SolCount == 0:
        return None

    chosen = [p for p in range(n_pts) if w[p].X > 0.5]
    spend = float(sum(cost[p] for p in chosen))
    _, rec = covered_recovery(chosen, covers, client_value)
    return (chosen, loc_id[chosen], np.asarray([cost[p] for p in chosen]),
            np.asarray([recovered[p] for p in chosen]), rec, n_dep, spend)


def _solve_scipy(loc_id, cost, recovered, covers, client_value, total_value,
                 loss, budget, group_loss_caps=None, time_limit=120.0,
                 mip_gap=0.001, verbose=False, diagnostics=None):
    """Same model and interface as `_solve_gurobi`, on the free HiGHS MILP.

    Used as a fallback when gurobipy is absent or its environment carries only
    a size-limited licence: the full (a, i) sweep produces tens of thousands
    of candidate binaries, beyond the ~2000-var community limit.

    A cell is truly infeasible only when HiGHS returns no incumbent at all
    (``res.x is None``). scipy sets ``success=False`` for a time-limit exit,
    even when a valid incumbent exists, so ``res.success`` alone must not be
    used to drop a cell; a non-optimal incumbent is accepted with a warning.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack

    n_pts = len(loc_id)
    loc_ids = np.unique(loc_id)
    n_loc = len(loc_ids)
    n_client = covers.shape[1]
    local_of = {d: i for i, d in enumerate(loc_ids)}
    n = n_pts + n_loc + n_client
    i_w = lambda p: p
    i_y = lambda d: n_pts + local_of[d]
    i_z = lambda c: n_pts + n_loc + c

    ub_rows, ub_cols, ub_vals = [], [], []
    ub_rhs = []
    for c in range(n_client):
        r = len(ub_rhs)
        ub_rhs.append(0.0)
        serving = np.flatnonzero(covers[:, c])
        ub_rows.extend([r] * (serving.size + 1))
        for p in serving:
            ub_cols.append(i_w(int(p)))
            ub_vals.append(-1.0)
        ub_cols.append(i_z(c))
        ub_vals.append(1.0)
        # Exact recovery indicator: every selected point that covers client c
        # forces z_c = 1, complementing z_c <= sum_p A_pc w_p above.
        for p in serving:
            r = len(ub_rhs)
            ub_rhs.append(0.0)
            ub_rows.extend((r, r))
            ub_cols.extend((i_w(int(p)), i_z(c)))
            ub_vals.extend((1.0, -1.0))
    # Loss caps: -sum value_c z_c <= -(Gamma - b), either for the complete
    # portfolio or separately for each client group.
    if group_loss_caps:
        for _name, mask, cap in group_loss_caps:
            members = np.flatnonzero(mask)
            group_total = float(client_value[members].sum())
            r = len(ub_rhs)
            ub_rhs.append(-(group_total - cap))
            ub_rows.extend([r] * len(members))
            for c in members:
                ub_cols.append(i_z(int(c)))
                ub_vals.append(-client_value[c])
    else:
        r = len(ub_rhs)
        ub_rhs.append(-(total_value - loss))
        ub_rows.extend([r] * n_client)
        for c in range(n_client):
            ub_cols.append(i_z(c))
            ub_vals.append(-client_value[c])
    # contract:  sum C_p w_p <= B
    r = len(ub_rhs)
    ub_rhs.append(budget)
    ub_rows.extend([r] * n_pts)
    for p in range(n_pts):
        ub_cols.append(i_w(p))
        ub_vals.append(cost[p])
    A_ub = csr_matrix((ub_vals, (ub_rows, ub_cols)),
                      shape=(len(ub_rhs), n))

    def pass1_assemble():
        eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
        for d in loc_ids:
            r = len(eq_rhs)
            eq_rhs.append(0.0)
            for p in np.flatnonzero(loc_id == d):
                eq_rows.append(r)
                eq_cols.append(i_w(int(p)))
                eq_vals.append(1.0)
            eq_rows.append(r)
            eq_cols.append(i_y(d))
            eq_vals.append(-1.0)
        A_eq = csr_matrix((eq_vals, (eq_rows, eq_cols)),
                          shape=(len(eq_rhs), n))
        c_obj = np.zeros(n)
        for d in loc_ids:
            c_obj[i_y(d)] = 1.0
        return A_eq, eq_rhs, c_obj

    options = {"time_limit": time_limit, "mip_rel_gap": mip_gap}
    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n, dtype=np.int8)

    A_eq, eq_rhs, c_obj = pass1_assemble()
    res1 = milp(c_obj, integrality=integrality, bounds=bounds,
                constraints=[LinearConstraint(A_ub, -np.inf, ub_rhs),
                             LinearConstraint(A_eq, eq_rhs, eq_rhs)],
                options=options)
    if diagnostics is not None:
        diagnostics.update(
            solver_backend="highs",
            stage1_status=int(res1.status),
            stage1_status_message=str(res1.message),
            stage1_optimal=int(res1.status == 0),
            stage1_objective=float(res1.fun) if res1.fun is not None else np.nan,
            stage1_bound=float(getattr(res1, "mip_dual_bound", np.nan)),
            stage1_mip_gap=float(getattr(res1, "mip_gap", np.nan)),
            stage1_nodes=int(getattr(res1, "mip_node_count", -1)),
        )
    if res1.x is None:
        return None
    if res1.status != 0:
        print(f"[HiGHS] pass1 status {res1.status}: accepting feasible "
              f"incumbent (optimality not proven)")
    x1 = np.asarray(res1.x)
    n_dep = int(round(float(x1[[i_y(d) for d in loc_ids]].sum())))

    def pass2_assemble():
        A_eq2, eq_rhs2, _ = pass1_assemble()
        fixed_rows = [0] * len(loc_ids)
        fixed_cols = [i_y(d) for d in loc_ids]
        fixed = csr_matrix((np.ones(len(loc_ids)),
                            (fixed_rows, fixed_cols)), shape=(1, n))
        A_eq2 = vstack((A_eq2, fixed), format="csr")
        eq_rhs2 = np.concatenate((np.asarray(eq_rhs2), [float(n_dep)]))
        c_obj2 = np.zeros(n)
        for p in range(n_pts):
            c_obj2[i_w(p)] = cost[p]
        return A_eq2, eq_rhs2, c_obj2

    A_eq2, eq_rhs2, c2 = pass2_assemble()
    res2 = milp(c2, integrality=integrality, bounds=bounds,
                constraints=[LinearConstraint(A_ub, -np.inf, ub_rhs),
                             LinearConstraint(A_eq2, eq_rhs2, eq_rhs2)],
                options={"time_limit": max(1.0, time_limit / 2.0),
                         "mip_rel_gap": mip_gap})
    if diagnostics is not None:
        diagnostics.update(
            stage2_status=int(res2.status),
            stage2_status_message=str(res2.message),
            stage2_optimal=int(res2.status == 0),
            stage2_objective=float(res2.fun) if res2.fun is not None else np.nan,
            stage2_bound=float(getattr(res2, "mip_dual_bound", np.nan)),
            stage2_mip_gap=float(getattr(res2, "mip_gap", np.nan)),
            stage2_nodes=int(getattr(res2, "mip_node_count", -1)),
        )
    if res2.x is None:
        return None
    if res2.status != 0:
        print(f"[HiGHS] pass2 status {res2.status}: accepting feasible "
              f"incumbent (optimality not proven)")
    x2 = np.asarray(res2.x)
    chosen = [p for p in range(n_pts) if x2[i_w(p)] > 0.5]
    spend = float(sum(cost[p] for p in chosen))
    _, rec = covered_recovery(chosen, covers, client_value)
    return (chosen, loc_id[chosen], np.asarray([cost[p] for p in chosen]),
            np.asarray([recovered[p] for p in chosen]), rec, n_dep, spend)


def solve(loc_id, cost, recovered, covers, client_value, total_value,
          loss, budget, group_loss_caps=None, time_limit=120.0, mip_gap=0.001,
          verbose=False, solver="auto", diagnostics=None):
    """Fewest depots, choosing one front point per open depot.

    `loss` caps the value of assets left unserved (US$); `budget` caps contract
    spend. solver='gurobi' | 'scipy' | 'auto'. 'auto' prefers gurobipy and
    falls back to scipy.optimize.milp (HiGHS, no licence) on a missing module
    or a size-limited licence -- the pip-installed gurobipy only solves models
    of a few thousand binaries, while the full sweep offers tens of thousands.
    """
    if solver in ("gurobi", "auto"):
        try:
            import gurobipy
        except ModuleNotFoundError:
            gurobipy = None
        if gurobipy is not None:
            try:
                return _solve_gurobi(loc_id, cost, recovered, covers,
                                     client_value, total_value, loss, budget,
                                     group_loss_caps=group_loss_caps,
                                     time_limit=time_limit, mip_gap=mip_gap,
                                     verbose=verbose, diagnostics=diagnostics)
            except gurobipy.GurobiError as e:
                if not ("size-limited" in str(e) or "license" in str(e).lower()):
                    raise
                print("gurobipy licence is size-limited; "
                      "falling back to scipy.optimize.milp (HiGHS)")
    if solver == "gurobi":
        raise SystemExit("solver=gurobi requested but gurobipy unusable")
    return _solve_scipy(loc_id, cost, recovered, covers, client_value,
                        total_value, loss, budget,
                        group_loss_caps=group_loss_caps,
                        time_limit=time_limit, mip_gap=mip_gap, verbose=verbose,
                        diagnostics=diagnostics)


def covered_recovery(chosen, covers, client_value):
    """Deterministic recovered value from the covered union of chosen points.

    `chosen` is a list of candidate indices; returns (mask, value_sum). The
    MILP's recovery flag z is otherwise under-determined by the objective, so
    reporting from the covered union keeps both solvers bit-identical.
    """
    mask = np.zeros(covers.shape[1], dtype=bool)
    if len(chosen):
        mask |= np.asarray(covers[chosen]).any(axis=0)
    return mask, float(client_value[mask].sum())


def best_recoverable(covers, client_value, total_value):
    """Largest recoverable value fraction: the union of all candidates."""
    union = covers.any(axis=0)
    return float(client_value[union].sum()) / total_value \
        if total_value > 0 else 0.0


def min_loss(covers, client_value, total_value):
    """Smallest achievable lost value [US$]: the complement of the union."""
    return float(total_value) * (1.0 - best_recoverable(
        covers, client_value, total_value)) if total_value > 0 else 0.0
