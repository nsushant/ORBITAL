"""Build the pairwise transfer cost table.

One entry per (departure node, arrival node, departure epoch, time of flight):
the delta-V of the Edelbaum transfer of Section 3.2 plus the in-plane phasing
that closes the remaining along-track gap, and the extra time that phasing
takes. The scheduler reads the table; the optimiser never solves a transfer.

What each entry depends on
--------------------------
The transfer delta-V depends on the two orbits (a, i) and on the RAAN change
the servicer must absorb. The orbits are fixed per pair, so only the RAAN
change varies across the epoch grid — which is why every entry for a pair goes
into one `transfer_cost` call and shares its ring bundles. That sharing is what
makes the table cheap.

Epochs are analytic, not looked up. `oos.nodes` fits each node's secular node
and argument of latitude, so RAAN and phase are available at any epoch in
closed form. The Julia had to wrap departure epochs modulo the 400-day
propagation to reuse one file for longer missions; here there is nothing to
wrap, and epochs beyond the propagated horizon are as valid as those inside it.

Speed, and what it cost
-----------------------
Two changes took a full table from hours to under an hour. The first is that
the ring ladder is now shared: `oos.edelbaum.transfer_cost_grid` computes each
ring once and tests every entry against it, where the older `transfer_cost`
refined each entry's growth bracket separately and rebuilt the 181-sample
bundle on every bisection step -- measured at 8617 bundle-equivalents per object
pair against a shared ladder of 48. That alone is about 12x, and it *improves*
the error tail: against a high-resolution reference the shared ladder is high by
0.87 % at the 99th percentile where the old settings were high by 4.1 %, and it
reports a transfer cheaper than it truly is on 0.1 % of entries where the old
settings did so on 0.4 %.

The second is optional and off by default. Nodes whose (a, i) agree to within a
tolerance share every ring, so the ladder can be built once per group of orbits
rather than once per object pair -- 8.8x fewer ladders on the 225-node instance,
worth about 2.6x in wall-clock on top of the first change. It is an
approximation, and the measured cost on a 20-node probe against an ungrouped
build is:

    tolerance          median      99th pct    worst      feasibility flips
    1 km / 0.01 deg    0.025 %     0.74 %      37.7 %     15 in 270,400
    0.25 km / 0.0025   0.019 %     0.73 %      36.1 %     11 in 270,400
    0.05 km / 0.0005   0.003 %     0.58 %      10.6 %     10 in 270,400

The tail does not close as the tolerance tightens, because it is not really a
tolerance effect: closure is not monotone in ring growth, so an entry sitting on
a feasibility boundary can jump to a quite different ring under an arbitrarily
small perturbation of the orbit. Since grouping buys 2.6x and can move a single
entry by 350 m/s, it is off unless asked for. Use it for exploration -- a depot
location sweep, where the table is rebuilt many times -- and leave it off for
any table a published number rests on.

Same-plane shortcut
-------------------
When the two planes coincide at the epoch in question — within 0.01 rad in both
inclination and node — there is no plane change to buy and the entry is phasing
alone. This is not a rare case: the Planet Labs population clusters heavily in
RAAN.

Output
------
`outputs/cost_table.h5` with `dv` and `phasing_days`, both
(n_nodes, n_nodes, n_dep, n_tof) in float32, m/s and days, NaN where no drift
orbit closes the node within the time of flight. Node order, the epoch grids
and the servicer are stored alongside as attributes.
"""

from __future__ import annotations

import argparse
import os
import time

import math

import numpy as np
from numba import njit, prange

from . import servicer
from . import edelbaum
from .edelbaum import MIN_GROWTH, transfer_cost_grid
from .nodes import load_nodes
from .phasing import phasing

DAY = 86400.0
GRID_STEP_DAYS = 15.0
HORIZON_DAYS = 400.0

SAME_PLANE_TOL = 0.01      # rad, on both inclination and node


@njit(cache=True)
def _wrap_pi(x):
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


def plane_groups(a, incl, da_tol=1.0, di_tol=1e-4):
    """Group nodes whose orbits are the same to within a tolerance.

    The ring geometry the transfer search works on depends only on the two
    orbits' (a, i). Nodes sharing those share every ring, so the ladder can be
    built once per group pair instead of once per object pair. RAAN is *not*
    part of the grouping: it enters only through the required nodal change,
    which stays per object and per epoch, so grouping cannot perturb it.

    At 1 km and 0.01 degrees the 225-node instance collapses to 76 groups, which
    is 8.8 times fewer distinct pairs. The residual error is the difference
    between a member's own semi-major axis and its group's mean, at most half
    the tolerance: about 0.5 km, or 0.5 m/s on a velocity-plane endpoint.

    Returns (group index per node, representative a, representative inclination).
    """
    n = a.shape[0]
    gid = np.full(n, -1, dtype=np.int64)
    reps = []
    for k in range(n):
        for g, (ra, ri) in enumerate(reps):
            if abs(a[k] - ra) < da_tol and abs(incl[k] - ri) < di_tol:
                gid[k] = g
                break
        else:
            gid[k] = len(reps)
            reps.append((a[k], incl[k]))
    # Use each group's mean rather than its first member, so the error is
    # centred on the group instead of biased towards whoever arrived first.
    a_rep = np.zeros(len(reps))
    i_rep = np.zeros(len(reps))
    for g in range(len(reps)):
        sel = gid == g
        a_rep[g] = a[sel].mean()
        i_rep[g] = incl[sel].mean()
    return gid, a_rep, i_rep


@njit(cache=True, parallel=True)
def build(a, incl, raan0, raan_rate, u0, u_rate, dep_s, tof_s,
          mass, isp, thrust, n_scan, n_ladder, max_growth, min_growth,
          gid, a_rep, i_rep, memb_start, memb_count, memb_idx,
          pair_gi, pair_gj):
    """Fill the whole table, one shared ladder per group pair."""
    n = a.shape[0]
    n_dep = dep_s.shape[0]
    n_tof = tof_s.shape[0]

    dv_out = np.full((n, n, n_dep, n_tof), np.nan, dtype=np.float32)
    ph_out = np.zeros((n, n, n_dep, n_tof), dtype=np.float32)

    for t in prange(pair_gi.shape[0]):
        gi = pair_gi[t]
        gj = pair_gj[t]

        s1 = np.empty(6)
        s2 = np.empty(6)
        s1[0] = a_rep[gi]; s1[1] = i_rep[gi]; s1[2] = 0.0
        s2[0] = a_rep[gj]; s2[1] = i_rep[gj]; s2[2] = 0.0
        for c in range(3, 6):
            s1[c] = (mass, isp, thrust)[c - 3]
            s2[c] = (mass, isp, thrust)[c - 3]

        ni = memb_count[gi]
        nj = memb_count[gj]
        coplanar_incl = abs(i_rep[gi] - i_rep[gj]) < SAME_PLANE_TOL

        n_row = ni * nj * n_dep
        d_raans = np.empty((n_row, n_tof))

        for u in range(ni):
            i = memb_idx[memb_start[gi] + u]
            for v in range(nj):
                j = memb_idx[memb_start[gj] + v]
                base = (u * nj + v) * n_dep
                for p in range(n_dep):
                    dep = dep_s[p]
                    raan_i = raan0[i] + raan_rate[i] * dep
                    for q in range(n_tof):
                        arr = dep + tof_s[q]
                        d_raans[base + p, q] = _wrap_pi(
                            (raan0[j] + raan_rate[j] * arr) - raan_i)

        dv = transfer_cost_grid(s1, s2, tof_s, d_raans, n_scan, n_ladder,
                                max_growth, min_growth)

        for u in range(ni):
            i = memb_idx[memb_start[gi] + u]
            for v in range(nj):
                j = memb_idx[memb_start[gj] + v]
                if i == j:
                    continue
                base = (u * nj + v) * n_dep
                for p in range(n_dep):
                    dep = dep_s[p]
                    ui = u0[i] + u_rate[i] * dep
                    for q in range(n_tof):
                        arr = dep + tof_s[q]
                        # The servicer carries the departure object's phase
                        # through the transfer, as cost/cost_functions.jl does.
                        dv_phase, dur = phasing(a[j], ui + u_rate[i] * tof_s[q],
                                                u0[j] + u_rate[j] * arr)
                        same_plane = (coplanar_incl
                                      and abs(d_raans[base + p, q]) < SAME_PLANE_TOL)
                        if same_plane:
                            dv_out[i, j, p, q] = np.float32(dv_phase * 1000.0)
                            ph_out[i, j, p, q] = np.float32(dur / DAY)
                        elif not np.isnan(dv[base + p, q]):
                            dv_out[i, j, p, q] = np.float32(
                                (dv[base + p, q] + dv_phase) * 1000.0)
                            ph_out[i, j, p, q] = np.float32(dur / DAY)

    return dv_out, ph_out


def group_arrays(gid, n_groups):
    """CSR-style member lists, so the njit builder needs no ragged structures."""
    counts = np.array([(gid == g).sum() for g in range(n_groups)], dtype=np.int64)
    start = np.zeros(n_groups, dtype=np.int64)
    start[1:] = np.cumsum(counts)[:-1]
    idx = np.concatenate([np.flatnonzero(gid == g) for g in range(n_groups)])
    return start, counts, idx.astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default="outputs/simulation.h5")
    p.add_argument("--out", default="outputs/cost_table.h5")
    p.add_argument("--step", type=float, default=GRID_STEP_DAYS)
    p.add_argument("--horizon", type=float, default=HORIZON_DAYS)
    p.add_argument("--limit", type=int, default=0,
                   help="build only the first N nodes, for timing")
    p.add_argument("--group-da", type=float, default=0.0,
                   help="semi-major-axis tolerance for plane grouping [km]. "
                        "0 (the default) means no grouping: every node keeps its "
                        "own orbit and the table is exact. Grouping is roughly "
                        "2.6x faster and costs accuracy -- see the module "
                        "docstring for measured numbers")
    p.add_argument("--group-di", type=float, default=0.0,
                   help="inclination tolerance for plane grouping [deg]")
    p.add_argument("--n-scan", type=int, default=180)
    p.add_argument("--n-ladder", type=int, default=1200,
                   help="rungs on the shared growth ladder; the delta-V returned "
                        "is high by at most one spacing and never low")
    p.add_argument("--max-growth", type=float, default=edelbaum.MAX_GROWTH)
    p.add_argument("--min-growth", type=float, default=MIN_GROWTH)
    args = p.parse_args()

    import h5py

    nd = load_nodes(args.sim)
    grid = np.arange(args.step, args.horizon + 1e-9, args.step)
    sl = slice(0, args.limit) if args.limit else slice(None)
    names = nd.names[sl]
    n = len(names)
    print(f"nodes {n}   grid {len(grid)} departures x {len(grid)} times of flight "
          f"= {n * (n - 1) * len(grid) ** 2:,} entries")
    print(f"servicer {servicer.MASS:.0f} kg, {servicer.THRUST_N * 1e3:.0f} mN, "
          f"Isp {servicer.ISP:.0f} s")

    # A zero tolerance puts every node in its own group, which is the exact
    # table; the grouped path is then a no-op wrapper around the same call.
    gid, a_rep, i_rep = plane_groups(nd.a[sl], nd.incl[sl],
                                     max(args.group_da, 0.0),
                                     max(math.radians(args.group_di), 0.0))
    n_groups = len(a_rep)
    memb_start, memb_count, memb_idx = group_arrays(gid, n_groups)
    gi, gj = np.meshgrid(np.arange(n_groups), np.arange(n_groups), indexing="ij")
    pair_gi, pair_gj = gi.reshape(-1), gj.reshape(-1)
    if args.group_da > 0.0 or args.group_di > 0.0:
        print(f"plane groups: {n_groups} at {args.group_da:g} km / "
              f"{args.group_di:g} deg -> {n_groups ** 2:,} group pairs instead of "
              f"{n * (n - 1):,} object pairs "
              f"({n * (n - 1) / max(n_groups ** 2, 1):.1f}x fewer ladders). "
              f"APPROXIMATE -- see the module docstring")
    else:
        print(f"no plane grouping: exact, one ladder per orbit pair")

    t0 = time.time()
    dv, ph = build(nd.a[sl], nd.incl[sl], nd.raan0[sl], nd.raan_rate[sl],
                   nd.u0[sl], nd.u_rate[sl], grid * DAY, grid * DAY,
                   servicer.MASS, servicer.ISP, servicer.THRUST,
                   args.n_scan, args.n_ladder, args.max_growth, args.min_growth,
                   gid, a_rep, i_rep, memb_start, memb_count, memb_idx,
                   pair_gi, pair_gj)
    elapsed = time.time() - t0
    pairs = n * (n - 1)
    print(f"built in {elapsed:.1f} s  ({1e3 * elapsed / pairs:.2f} ms per ordered pair)")

    finite = np.isfinite(dv)
    n_off = pairs * len(grid) ** 2

    # A table of NaNs reads downstream as "no transfer exists" rather than
    # "never computed", so an empty or near-empty build has to announce itself
    # rather than be inferred from a percentage in a log nobody reads. This is
    # what a numba parallel build did when the ring search hit a degenerate
    # drift orbit: it returned all NaN instead of raising.
    if finite.sum() == 0:
        raise SystemExit(
            "ABORT: not one entry is feasible. That is a numerical failure, not "
            "a result -- a table of NaNs is indistinguishable downstream from a "
            "table saying no transfer exists. Run tests/test_guards.py and "
            "tests/test_edelbaum.py before trusting anything built from this.")
    dead_rows = int((~finite).all(axis=(1, 2, 3)).sum())
    if dead_rows:
        print(f"WARNING: {dead_rows} departure node(s) have no feasible transfer "
              f"to anywhere, at any epoch. Check they are real orbits.")
    print(f"feasible entries: {finite.sum():,} of {n_off:,} "
          f"({100.0 * finite.sum() / n_off:.1f} %)")
    if finite.any():
        v = dv[finite]
        print(f"delta-V m/s: min {v.min():.1f}  median {np.median(v):.1f}  "
              f"p90 {np.percentile(v, 90):.1f}  max {v.max():.1f}")
        print(f"within the {servicer.DV_BUDGET * 1000:.0f} m/s budget: "
              f"{100.0 * (v <= servicer.DV_BUDGET * 1000).mean():.1f} % of feasible entries")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("dv", data=dv, compression="lzf")
        f.create_dataset("phasing_days", data=ph, compression="lzf")
        f.create_dataset("names", data=np.array(names, dtype="S32"))
        f.create_dataset("departure_days", data=grid)
        f.create_dataset("tof_days", data=grid)
        f.attrs["units"] = "dv m/s, phasing_days days"
        f.attrs["servicer_mass_kg"] = servicer.MASS
        f.attrs["servicer_isp_s"] = servicer.ISP
        f.attrs["servicer_thrust_N"] = servicer.THRUST_N
        f.attrs["dv_budget_m_s"] = servicer.DV_BUDGET * 1000.0
        f.attrs["source_sim"] = os.path.basename(args.sim)
        f.attrs["n_ladder"] = args.n_ladder
        f.attrs["group_da_km"] = args.group_da
        f.attrs["group_di_deg"] = args.group_di
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
