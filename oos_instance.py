"""
oos_instance.py — turn a demand scenario + cost table into an `Instance` and a
`dv_cost` callback for the Hexaly model in ../oos_hexaly.py.

    from oos_instance import build_instance
    from oos_hexaly import OOSModel

    inst, dv_cost, meta = build_instance("outputs/exp_demands/tight_normal_01.jld2",
                                         n_clients=25, n_vehicles=3)
    with OOSModel(inst, dv_cost) as model:
        sol = model.solve(time_limit=60)

Conventions bridged here
------------------------
units       Everything in the model is in **days** (the unit of the cost table
            grids, the demand deadlines and the service times).  `dv_cost`
            returns m/s and Isp is in seconds; Tsiolkovsky only ever sees a
            total delta-v, so the time unit never enters it.
node ids    Tables are 1-based Julia indices (`sat_N` -> N, the single depot ->
            the highest index).  The model uses 0-based global ids: client c ->
            c, depot d -> n_clients + d.  `node_tbl` maps one to the other.
grids       Departure and arrival times are snapped up to the table grid, the
            same rule as `loaders.snap_cost`.  A leg with no table entry (or a
            masked entry >= 1e7) costs `dv_cap`, which is far above any usable
            budget, so eq. (8) rejects it while keeping the objective finite.

Cost tables
-----------
outputs/cost_table.jld2       nodes 1..225 (224 sats + depot), 15-day grids,
                              34M entries, ~1.3 GB       <- default: what the
                                                            exp_demands were
                                                            generated from
outputs/cost_table_basic.h5   nodes 1..201 (200 sats + depot), 30-day grids,
                              6.8M entries, ~260 MB

Either loads in a couple of seconds -- the file is streamed in chunks and only
the legs between this instance's nodes are kept.  The basic table covers sats
1..200 only, so a scenario's demands on sats 201..224 are dropped (counted in
`meta["dropped_out_of_table"]`), and its 30-day arrival grid starting at day 60
is coarse against the 10-100 day deadlines of the `tight_*` scenarios.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos_hexaly import Instance  # noqa: E402

from loaders import load_demands  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASIC_TABLE = os.path.join(HERE, "outputs", "cost_table_basic.h5")
MASTER_TABLE = os.path.join(HERE, "outputs", "cost_table.jld2")

INVALID_COST = 1e7        # the mask generate_demands.jl uses for unusable legs
CHUNK = 1_000_000


# ---------------------------------------------------------------------------
# Cost table
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    start: float
    step: float
    n: int

    def slot(self, t: float) -> int:
        """Smallest grid index whose time is >= t (clipped), as in snap_cost."""
        k = int(math.ceil((t - self.start) / self.step))
        return 0 if k < 0 else (self.n - 1 if k >= self.n else k)


def _iter_table(path):
    """Yield (from, to, dep, arr, cost) chunks from either table format."""
    with h5py.File(path, "r") as f:
        if "CostTable" in f:                     # JLD2 written by Julia
            kv = f[f["CostTable"][()].item()[0]]
            for i in range(0, kv.shape[0], CHUNK):
                a = kv[i:i + CHUNK]
                yield (a["first"]["1"], a["first"]["2"],
                       a["first"]["3"], a["first"]["4"], a["second"])
        else:                                    # flat HDF5 from export_cost_table.jl
            n = f["cost"].shape[0]
            for i in range(0, n, CHUNK):
                sl = slice(i, i + CHUNK)
                yield (f["from_idx"][sl], f["to_idx"][sl], f["dep"][sl],
                       f["arr"][sl], f["cost"][sl])


def _table_geometry(path):
    """One pass for the node count and the two time grids."""
    max_node = 0
    dep_lo = arr_lo = math.inf
    dep_hi = arr_hi = -math.inf
    dep_vals, arr_vals = set(), set()
    for fr, to, dep, arr, _ in _iter_table(path):
        max_node = max(max_node, int(fr.max()), int(to.max()))
        dep_lo, dep_hi = min(dep_lo, dep.min()), max(dep_hi, dep.max())
        arr_lo, arr_hi = min(arr_lo, arr.min()), max(arr_hi, arr.max())
        if len(dep_vals) < 4096:
            dep_vals.update(np.unique(dep).tolist())
            arr_vals.update(np.unique(arr).tolist())
    dep_step = float(np.diff(sorted(dep_vals)).min())
    arr_step = float(np.diff(sorted(arr_vals)).min())
    dep_g = Grid(float(dep_lo), dep_step, int(round((dep_hi - dep_lo) / dep_step)) + 1)
    arr_g = Grid(float(arr_lo), arr_step, int(round((arr_hi - arr_lo) / arr_step)) + 1)
    return max_node, dep_g, arr_g


def load_dense_costs(path, node_tbl, dep_g, arr_g, verbose=True):
    """
    Dense (node, node, dep_slot, arr_slot) delta-v block for the nodes actually
    used by this instance.  Entries never written stay at inf.

    `node_tbl[p]` is the table index of local node p.
    """
    n_node = len(node_tbl)
    flat = np.full(n_node * n_node * dep_g.n * arr_g.n, np.inf)

    tbl2pos = np.full(int(node_tbl.max()) + 1, -1, dtype=np.int64)
    tbl2pos[node_tbl] = np.arange(n_node)

    kept = 0
    for fr, to, dep, arr, cost in _iter_table(path):
        pf = tbl2pos[np.clip(fr, 0, len(tbl2pos) - 1)]
        pt = tbl2pos[np.clip(to, 0, len(tbl2pos) - 1)]
        keep = (pf >= 0) & (pt >= 0) & (fr < len(tbl2pos)) & (to < len(tbl2pos))
        keep &= cost < INVALID_COST
        if not keep.any():
            continue
        pf, pt = pf[keep], pt[keep]
        kd = np.rint((dep[keep] - dep_g.start) / dep_g.step).astype(np.int64)
        ka = np.rint((arr[keep] - arr_g.start) / arr_g.step).astype(np.int64)
        idx = ((pf * n_node + pt) * dep_g.n + kd) * arr_g.n + ka
        flat[idx] = cost[keep]
        kept += int(keep.sum())

    if verbose:
        print(f"  cost table: {kept:,} legs retained for {n_node} nodes "
              f"({np.isfinite(flat).mean():.1%} of the dense block filled)")
    return flat


def make_dv_cost(flat, node_tbl, dep_g, arr_g, dv_cap):
    """Build the `dv_cost(i, j, arrive_i, service_i, arrive_j)` callback."""
    n_node = len(node_tbl)
    nd, na = dep_g.n, arr_g.n
    d0, dstep = dep_g.start, dep_g.step
    a0, astep = arr_g.start, arr_g.step

    def dv_cost(i, j, arrive_i, service_i, arrive_j):
        dep = arrive_i + service_i
        kd = int(math.ceil((dep - d0) / dstep))
        kd = 0 if kd < 0 else (nd - 1 if kd >= nd else kd)
        ka = int(math.ceil((arrive_j - a0) / astep))
        ka = 0 if ka < 0 else (na - 1 if ka >= na else ka)
        c = flat[((i * n_node + j) * nd + kd) * na + ka]
        return c if c < dv_cap else dv_cap

    return dv_cost


# ---------------------------------------------------------------------------
# Instance builder
# ---------------------------------------------------------------------------
def build_instance(
    demand_path: str,
    cost_path: str = MASTER_TABLE,
    n_clients: Optional[int] = 25,
    n_vehicles: int = 3,
    n_depots: int = 1,
    dv_max: float = 8000.0,
    min_tof: Optional[float] = None,
    horizon: Optional[float] = None,
    # servicer physics: Xe Hall thruster, per BCR_plan.md (Tirila et al. 2023)
    dry_mass: float = 300.0,
    isp: float = 2800.0,
    prop_cost: float = 340.0,
    # Fixed costs are EXERCISE VALUES, not the project's figures.  The real
    # ones (launch at $3,700/kg of wet mass + SSCM manufacturing, ~$74M per
    # servicer -- see plot_bcr_architecture.py) exceed the recoverable asset
    # value of all 200 demands, so profit maximisation correctly answers
    # "serve nobody".  That is the BCR study's conclusion, not a model fault;
    # to see routes, the fixed costs have to sit below the revenue on offer.
    depot_cost: float = 5.0e5,
    vehicle_cost: float = 3.0e5,
    fee_fraction: float = 1.0,      # BCR model pays the operator 0.10
    depot_service: float = 1.0,
    depot_idx: Optional[int] = None,
    verbose: bool = True,
):
    """
    Returns (Instance, dv_cost, meta).

    n_clients   how many demands to keep (None = all).  Demands are taken in
                file order, which is already randomised by the generator.
    min_tof     defaults to one departure-grid step, below which the snapped
                delta-v is meaningless.
    horizon     defaults to the last arrival grid point.
    depot_idx   table index of the depot; inferred from the table if omitted.
    """
    dem = load_demands(demand_path)
    if verbose:
        print(f"scenario   : {os.path.basename(demand_path)} ({len(dem['UIDs'])} demands)")
        print(f"cost table : {os.path.basename(cost_path)}")

    max_node, dep_g, arr_g = _table_geometry(cost_path)
    if depot_idx is None:
        depot_idx = max_node
    n_sat = depot_idx - 1                      # sat indices are 1..depot_idx-1
    if verbose:
        print(f"  nodes 1..{max_node} (depot = {depot_idx}), "
              f"dep grid {dep_g.start:g}+{dep_g.step:g}*k (n={dep_g.n}), "
              f"arr grid {arr_g.start:g}+{arr_g.step:g}*k (n={arr_g.n})")

    sat_num = np.array([int(s.split("_")[1]) for s in dem["sat_identifiers"]])
    in_table = sat_num <= n_sat
    dropped = int((~in_table).sum())
    keep = np.flatnonzero(in_table)
    if n_clients is not None:
        keep = keep[:n_clients]
    if keep.size == 0:
        raise ValueError("no demand in this scenario is covered by the cost table")

    nC, nD = len(keep), n_depots
    node_tbl = np.concatenate([sat_num[keep], np.full(nD, depot_idx)])

    flat = load_dense_costs(cost_path, node_tbl, dep_g, arr_g, verbose)
    dv_cap = 5.0 * dv_max                       # keeps exp() in Tsiolkovsky finite
    dv_cost = make_dv_cost(flat, node_tbl, dep_g, arr_g, dv_cap)

    inst = Instance(
        n_clients=nC,
        n_depots=nD,
        n_vehicles=n_vehicles,
        service=np.asarray(dem["service_times"], float)[keep],
        deadline=np.asarray(dem["demand_deadlines"], float)[keep],
        release=np.zeros(nC),
        depot_service=np.full(nD, depot_service),
        fee=fee_fraction * np.asarray(dem["asset_values"], float)[keep],
        dv_max=dv_max,
        horizon=float(horizon if horizon is not None
                      else arr_g.start + (arr_g.n - 1) * arr_g.step),
        min_tof=float(min_tof if min_tof is not None else dep_g.step),
        depot_cost=depot_cost,
        vehicle_cost=vehicle_cost,
        prop_cost=prop_cost,
        dry_mass=dry_mass,
        isp=isp,
    )

    meta = {
        "demand_index": keep,                   # row in the scenario file
        "sat_number": sat_num[keep],            # table index of each client
        "sat_name": [dem["sat_identifiers"][k] for k in keep],
        "depot_idx": depot_idx,
        "node_tbl": node_tbl,
        "dep_grid": dep_g,
        "arr_grid": arr_g,
        "dv_cap": dv_cap,
        "dropped_out_of_table": dropped,
        "dv_cost_flat": flat,
    }

    if verbose:
        rev = inst.fee.sum()
        print(f"  clients {nC}, depots {nD}, vehicles {n_vehicles}"
              + (f"  ({dropped} demands dropped: sat index > {n_sat})" if dropped else ""))
        print(f"  deadlines {inst.deadline.min():.0f}-{inst.deadline.max():.0f} d, "
              f"service {inst.service.min():.1f}-{inst.service.max():.1f} d, "
              f"min_tof {inst.min_tof:g} d, horizon {inst.horizon:g} d")
        m_prop = dry_mass * (math.exp(dv_max / (9.80665 * isp)) - 1.0)
        print(f"  revenue if all served ${rev:,.0f} vs fixed cost of one "
              f"depot + one vehicle ${depot_cost + vehicle_cost:,.0f}")
        print(f"  a full {dv_max:g} m/s tank is {m_prop:.0f} kg of propellant "
              f"= ${m_prop * prop_cost:,.0f}")
        _report_reachability(inst, dv_cost, nC, meta)
    return inst, dv_cost, meta


def _report_reachability(inst, dv_cost, nC, meta):
    """How many clients the depot can reach at all, given the deadlines."""
    depot = nC                                   # first depot's global id
    ok = 0
    for c in range(nC):
        arr = inst.deadline[c] - inst.service[c]
        if dv_cost(depot, c, 0.0, 0.0, arr) <= inst.dv_max:
            ok += 1
    print(f"  reachable direct from depot within budget & deadline: {ok}/{nC}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from oos_hexaly import OOSModel

    scenario = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "outputs", "exp_demands", "tight_normal_01.jld2"))
    inst, dv_cost, meta = build_instance(scenario, n_clients=25, n_vehicles=3)

    with OOSModel(inst, dv_cost) as model:
        sol = model.solve(time_limit=60, verbosity=1)

    print(f"\nprofit           : ${sol['profit']:,.0f}")
    print(f"propellant       : {sol['propellant_kg']:,.1f} kg")
    print(f"active depots    : {sol['n_depots_active']}")
    print(f"active vehicles  : {sol['n_vehicles_active']}")
    print(f"unserved clients : {len(sol['unserved'])}/{inst.n_clients}")
    for r in sol["routes"]:
        legs = " -> ".join(
            (f"D@{t:.0f}d" if kind == "depot" else f"{meta['sat_name'][idx]}@{t:.0f}d")
            for kind, idx, t in r["stops"])
        print(f"  v{r['vehicle']}: {legs}")
