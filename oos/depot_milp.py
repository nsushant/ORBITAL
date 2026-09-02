"""Choose the fewest depot locations that hold unserved client value below a bound.

Input is one column per candidate depot location, produced by the (a, i) sweep:
the set of clients that location serves at the **maximum-coverage** point of its
own Pareto front, together with the delta-V and active fleet size that point
implies (decision D13a).

Model
-----
    minimise    sum_d y_d
    subject to  sum_c value_c (1 - z_c)  <=  R
                z_c  <=  sum_{d : d covers c} y_d          for every client c
                y_d, z_c binary

y_d selects a depot location, z_c records whether client c ends up covered. The
second constraint is the only link between them: a client may be counted as
covered only if some selected location covers it. Nothing pushes z_c down, so at
the optimum it is as large as the selection allows.

Sweeping R traces the curve of depot count against tolerated unserved value,
which is the figure the design study wants.

What the answer is and is not
-----------------------------
It is the smallest number of depots **among the locations sampled**, with every
depot operated at the maximum-coverage point of its own front. Two caveats have
to travel with any number this produces, because they pull in opposite
directions:

  * Maximum coverage is the most expensive point on each front. The depot count
    is therefore the smallest that the sampled grid admits when cost is not
    constrained, and it must be reported next to the delta-V and fleet size the
    selected columns imply, or a coverage result reads as a cost result.
  * The columns come from a finite grid and from a heuristic, so this optimises
    over what MDLS happened to find where the sweep happened to look.

It is not "the optimal number of depots".

Usage
-----
    python3 -m oos.depot_milp --sweep outputs/depot_sweep.h5 \
                              --out outputs/depot_selection.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def load_columns(path):
    """Read the sweep output. Returns locations, coverage, value, dv, fleet."""
    import h5py

    with h5py.File(path, "r") as f:
        cover = f["coverage"][:].astype(bool)      # (n_loc, n_client)
        value = f["client_value"][:]               # (n_client,)
        a_km = f["a_km"][:]
        incl_deg = f["incl_deg"][:]
        dv = f["dv_m_s"][:]
        fleet = f["fleet"][:]
        clients = [c.decode() if isinstance(c, bytes) else c
                   for c in f["client_names"][:]]
    return a_km, incl_deg, cover, value, dv, fleet, clients


def solve(cover, value, bound, dv=None, fleet=None, time_limit=120.0,
          verbose=False):
    """Fewest depot locations leaving at most `bound` of client value unserved.

    Returns (selected indices, unserved value) or (None, None) if infeasible.
    """
    import gurobipy as gp
    from gurobipy import GRB

    n_loc, n_client = cover.shape
    total = float(value.sum())

    m = gp.Model("depot_selection")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit

    y = m.addVars(n_loc, vtype=GRB.BINARY, name="y")
    z = m.addVars(n_client, vtype=GRB.BINARY, name="z")

    for c in range(n_client):
        serving = np.flatnonzero(cover[:, c])
        if serving.size == 0:
            m.addConstr(z[c] == 0, name=f"unreachable_{c}")
        else:
            m.addConstr(z[c] <= gp.quicksum(y[int(d)] for d in serving),
                        name=f"link_{c}")

    m.addConstr(gp.quicksum(float(value[c]) * (1 - z[c])
                            for c in range(n_client)) <= bound,
                name="replacement_bound")

    m.setObjective(gp.quicksum(y[d] for d in range(n_loc)), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return None, None
    chosen = [d for d in range(n_loc) if y[d].X > 0.5]
    unserved = total - sum(float(value[c]) for c in range(n_client)
                           if z[c].X > 0.5)
    return chosen, unserved


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="outputs/depot_sweep.h5")
    p.add_argument("--out", default="outputs/depot_selection.csv")
    p.add_argument("--fractions", default="0.0,0.05,0.10,0.20,0.30,0.40,0.50",
                   help="unserved-value bounds, as fractions of total client value")
    p.add_argument("--time-limit", type=float, default=120.0)
    args = p.parse_args()

    a_km, incl_deg, cover, value, dv, fleet, clients = load_columns(args.sweep)
    total = float(value.sum())
    reach = cover.sum(axis=1)
    print(f"{cover.shape[0]} candidate locations, {cover.shape[1]} clients, "
          f"total client value {total:,.0f}")
    print(f"best single location covers {reach.max()} clients "
          f"({100.0 * reach.max() / cover.shape[1]:.1f} %) at "
          f"a = {a_km[reach.argmax()]:.0f} km, i = {incl_deg[reach.argmax()]:.2f} deg")
    unreachable = np.flatnonzero(cover.sum(axis=0) == 0)
    if unreachable.size:
        lost = float(value[unreachable].sum())
        print(f"{unreachable.size} clients are reached by no sampled location "
              f"({100.0 * lost / total:.1f} % of value); no bound below that is feasible")

    rows = []
    for frac in [float(x) for x in args.fractions.split(",")]:
        bound = frac * total
        chosen, unserved = solve(cover, value, bound, time_limit=args.time_limit)
        if chosen is None:
            print(f"  bound {100 * frac:5.1f} % : infeasible")
            rows.append(dict(bound_fraction=frac, bound_value=bound,
                             n_depots="", unserved_value="", unserved_fraction="",
                             total_dv_m_s="", total_fleet="", locations=""))
            continue
        sel_dv = float(np.sum(dv[chosen]))
        sel_fleet = float(np.sum(fleet[chosen]))
        locs = ";".join(f"{a_km[d]:.0f}/{incl_deg[d]:.2f}" for d in chosen)
        print(f"  bound {100 * frac:5.1f} % : {len(chosen)} depot(s), "
              f"unserved {100 * unserved / total:5.1f} %, "
              f"delta-V {sel_dv:,.0f} m/s, fleet {sel_fleet:.0f}  [{locs}]")
        rows.append(dict(bound_fraction=frac, bound_value=bound,
                         n_depots=len(chosen), unserved_value=unserved,
                         unserved_fraction=unserved / total,
                         total_dv_m_s=sel_dv, total_fleet=sel_fleet,
                         locations=locs))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
