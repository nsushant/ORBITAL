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

import numpy as np
from numba import njit, prange

from . import servicer
from . import edelbaum
from .edelbaum import transfer_cost
from .nodes import load_nodes
from .phasing import phasing

DAY = 86400.0
GRID_STEP_DAYS = 15.0
HORIZON_DAYS = 400.0

SAME_PLANE_TOL = 0.01      # rad, on both inclination and node


@njit(cache=True)
def _wrap_pi(x):
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


@njit(cache=True, parallel=True)
def build(a, incl, raan0, raan_rate, u0, u_rate, dep_s, tof_s,
          mass, isp, thrust, n_scan, n_growth, max_growth, growth_rtol,
          i_from, i_to):
    """Fill the whole table. Angles in rad, times in seconds, delta-V in km/s."""
    n = a.shape[0]
    n_dep = dep_s.shape[0]
    n_tof = tof_s.shape[0]
    n_entry = n_dep * n_tof

    n_rows = i_to - i_from
    dv_out = np.full((n_rows, n, n_dep, n_tof), np.nan, dtype=np.float32)
    ph_out = np.zeros((n_rows, n, n_dep, n_tof), dtype=np.float32)

    for row in prange(n_rows):
        i = i_from + row
        state1 = np.empty(6)
        state2 = np.empty(6)
        state1[0] = a[i]
        state1[1] = incl[i]
        state1[2] = 0.0
        state1[3] = mass
        state1[4] = isp
        state1[5] = thrust
        state2[3] = mass
        state2[4] = isp
        state2[5] = thrust

        tofs = np.empty(n_entry)
        d_raans = np.empty(n_entry)
        same_plane = np.empty(n_entry, dtype=np.bool_)
        u_from = np.empty(n_entry)
        u_to = np.empty(n_entry)

        for j in range(n):
            if i == j:
                continue
            state2[0] = a[j]
            state2[1] = incl[j]
            state2[2] = 0.0

            d_incl = abs(incl[i] - incl[j])
            coplanar_incl = d_incl < SAME_PLANE_TOL

            for p in range(n_dep):
                dep = dep_s[p]
                raan_i = raan0[i] + raan_rate[i] * dep
                ui = u0[i] + u_rate[i] * dep
                for q in range(n_tof):
                    k = p * n_tof + q
                    tof = tof_s[q]
                    arr = dep + tof
                    raan_j = raan0[j] + raan_rate[j] * arr
                    tofs[k] = tof
                    d_raans[k] = _wrap_pi(raan_j - raan_i)
                    same_plane[k] = (coplanar_incl and
                                     abs(d_raans[k]) < SAME_PLANE_TOL)
                    # The servicer carries the departure object's phase through
                    # the transfer, as in cost/cost_functions.jl.
                    u_from[k] = ui + u_rate[i] * tof
                    u_to[k] = u0[j] + u_rate[j] * arr

            dv = transfer_cost(state1, state2, tofs, d_raans, n_scan, n_growth,
                               max_growth, growth_rtol)

            for k in range(n_entry):
                p = k // n_tof
                q = k - p * n_tof
                dv_phase, dur = phasing(a[j], u_from[k], u_to[k])
                if same_plane[k]:
                    dv_out[row, j, p, q] = np.float32(dv_phase * 1000.0)
                    ph_out[row, j, p, q] = np.float32(dur / DAY)
                elif not np.isnan(dv[k]):
                    dv_out[row, j, p, q] = np.float32((dv[k] + dv_phase) * 1000.0)
                    ph_out[row, j, p, q] = np.float32(dur / DAY)

    return dv_out, ph_out


def merge(pattern, out):
    """Concatenate node slices, written by --from-node/--to-node, into one table.

    Slices must tile [0, n) exactly: a gap would leave a band of the table
    silently NaN, which reads downstream as "no transfer exists" rather than as
    "never computed", so it is checked rather than assumed.
    """
    import glob

    import h5py

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern!r}")

    pieces = []
    for path in paths:
        with h5py.File(path, "r") as f:
            pieces.append((int(f.attrs["node_from"]), int(f.attrs["node_to"]), path))
    pieces.sort()

    n = None
    with h5py.File(pieces[0][2], "r") as f:
        n = f["dv"].shape[1]
    cursor = 0
    for lo, hi, path in pieces:
        if lo != cursor:
            raise SystemExit(
                f"slices do not tile: expected a slice starting at {cursor}, "
                f"{os.path.basename(path)} starts at {lo}")
        cursor = hi
    if cursor != n:
        raise SystemExit(f"slices cover [0, {cursor}) but the table has {n} nodes")

    with h5py.File(pieces[0][2], "r") as f0, h5py.File(out, "w") as g:
        shape = (n,) + f0["dv"].shape[1:]
        dv = g.create_dataset("dv", shape=shape, dtype="f4", compression="lzf")
        ph = g.create_dataset("phasing_days", shape=shape, dtype="f4",
                              compression="lzf")
        for key in ("names", "departure_days", "tof_days"):
            g.create_dataset(key, data=f0[key][:])
        for key, val in f0.attrs.items():
            g.attrs[key] = val
        g.attrs["node_from"] = 0
        g.attrs["node_to"] = n
        for lo, hi, path in pieces:
            with h5py.File(path, "r") as f:
                dv[lo:hi] = f["dv"][:]
                ph[lo:hi] = f["phasing_days"][:]
            print(f"  merged [{lo}, {hi}) from {os.path.basename(path)}")
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default="outputs/simulation.h5")
    p.add_argument("--out", default="outputs/cost_table.h5")
    p.add_argument("--step", type=float, default=GRID_STEP_DAYS)
    p.add_argument("--horizon", type=float, default=HORIZON_DAYS)
    p.add_argument("--limit", type=int, default=0,
                   help="build only the first N nodes, for timing")
    p.add_argument("--from-node", type=int, default=0,
                   help="first departure node of this slice")
    p.add_argument("--to-node", type=int, default=0,
                   help="one past the last departure node; 0 means all of them")
    p.add_argument("--merge", default="",
                   help="glob of slice files to concatenate into --out, "
                        "instead of building anything")
    p.add_argument("--n-scan", type=int, default=180)
    p.add_argument("--n-growth", type=int, default=48)
    p.add_argument("--max-growth", type=float, default=edelbaum.MAX_GROWTH)
    p.add_argument("--growth-rtol", type=float, default=1e-3)
    args = p.parse_args()

    import h5py

    if args.merge:
        merge(args.merge, args.out)
        return

    nd = load_nodes(args.sim)
    grid = np.arange(args.step, args.horizon + 1e-9, args.step)
    sl = slice(0, args.limit) if args.limit else slice(None)
    names = nd.names[sl]
    n = len(names)
    print(f"nodes {n}   grid {len(grid)} departures x {len(grid)} times of flight "
          f"= {n * (n - 1) * len(grid) ** 2:,} entries")
    print(f"servicer {servicer.MASS:.0f} kg, {servicer.THRUST_N * 1e3:.0f} mN, "
          f"Isp {servicer.ISP:.0f} s")

    i_from = args.from_node
    i_to = args.to_node if args.to_node else n
    if not (0 <= i_from < i_to <= n):
        raise SystemExit(f"bad node slice [{i_from}, {i_to}) against {n} nodes")
    if (i_from, i_to) != (0, n):
        print(f"slice: departure nodes [{i_from}, {i_to})")

    t0 = time.time()
    dv, ph = build(nd.a[sl], nd.incl[sl], nd.raan0[sl], nd.raan_rate[sl],
                   nd.u0[sl], nd.u_rate[sl], grid * DAY, grid * DAY,
                   servicer.MASS, servicer.ISP, servicer.THRUST,
                   args.n_scan, args.n_growth, args.max_growth, args.growth_rtol,
                   i_from, i_to)
    elapsed = time.time() - t0
    pairs = (i_to - i_from) * (n - 1)
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
        f.attrs["node_from"] = i_from
        f.attrs["node_to"] = i_to
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
