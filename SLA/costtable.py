"""Build the pairwise transfer cost table, via constellation planes.

The table the optimiser reads is indexed [from, to, departure epoch, arrival
epoch]. It is not computed that way, because most of those entries are the same
calculation repeated.

The flow
--------
1. Collapse the clients to one representative per constellation plane. Members
   of a plane share a, inclination and RAAN, and therefore share a nodal rate,
   so they share their node at *every* epoch.
2. Compute the transfer table on plane representatives: for each ordered pair of
   planes and each (departure, time of flight), one Edelbaum solve.
3. Expand to every client pair by copying the plane entry and adding the phasing
   manoeuvre, which is the only part that differs between members of a plane.
4. The depot is a plane of one, so its legs come from the same machinery.

Why this is exact, not an approximation. The transfer delta-V depends on the two
orbits through (a, i) and on the required nodal change

    dOmega = Omega_Q(dep + tof) - Omega_P(dep)

and nothing else. Every member of plane P has the same Omega_P at every epoch,
so every member pair of (P, Q) needs an identical transfer. The per-member
difference is entirely in the along-track phase, which the phasing manoeuvre
handles and which step 3 computes per client. An earlier version of this file
grouped on (a, i) with a tolerance and kept RAAN per object; that merged
satellites which were *not* plane-mates and could move an entry by 350 m/s.
Keying on the full plane removes the approximation rather than tightening it.

What the collapse does and does not buy here. Writing the required change out,

    dOmega(dep, tof) = [Omega_Q(0) - Omega_P(0)]
                     + (Omega_dot_Q - Omega_dot_P) * dep
                     + Omega_dot_Q * tof

the departure epoch enters only through the *difference* of the two nodal rates.
For two planes at the same altitude and inclination those rates are equal, the
term vanishes, and every departure epoch gives the same answer. For cross-shell
pairs it does not, so the departure axis is kept in the plane table rather than
assumed away -- collapsing it there would silently produce a wrong table.

The size of the win is set by how many clients share a plane, and for this
instance that is capped at two: `fetch_and_sample.jl` samples at most
`SATS_PER_PLANE = 2` per plane and `build_instance.py` reproduces it. On the
full constellation, where a plane holds twenty-odd satellites, the same code
pays far better.

Output
------
`outputs/cost_table.h5` with `dv` and `phasing_s`, both
(n_nodes, n_nodes, n_dep, n_tof) in float32, m/s and seconds, NaN where no drift
orbit closes the node in time.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
from numba import njit, prange

from . import edelbaum, servicer
from .edelbaum import MIN_GROWTH, transfer_cost_grid
from .nodes import load_nodes
from .phasing import phasing

GRID_STEP_S = 1296000.0        # fine step, both axes
FINE_UNTIL_S = 33696000.0      # fine/coarse transition
COARSE_STEP_S = 2592000.0
DEP_HORIZON_S = 157766400.0    # five-year departure horizon
TOF_HORIZON_S = 72592000.0     # two-year single-leg horizon


def two_rate_grid(fine_step, fine_until, coarse_step, horizon):
    """Fine spacing out to `fine_until`, coarse spacing beyond it.

    The departure and time-of-flight axes are epoch grids, and neither
    snap_departure nor snap_tof (oos/schedule.py) assumes uniform spacing --
    both only need the grid sorted. That buys the resolution where it is worth
    paying for: relative RAAN moves fastest in the first year, and a leg's
    useful times of flight are short, so both axes keep 15-day steps early and
    drop to 30-day steps once the geometry is changing slowly.
    """
    fine = np.arange(fine_step, fine_until + 1e-9, fine_step)
    if horizon <= fine_until:
        return fine[fine <= horizon + 1e-9]
    coarse = np.arange(fine_until + coarse_step, horizon + 1e-9, coarse_step)
    return np.concatenate([fine, coarse])

SAME_PLANE_TOL = 0.01      # rad, on both inclination and node


@njit(cache=True)
def _wrap_pi(x):
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


def planes_from_instance(pid, a, incl, raan0):
    """Representative elements for each plane the instance declares.

    There is no clustering and no tolerance here. Plane membership is a property
    of the instance -- `build_instance.py` builds each Starlink plane by
    sampling one propagated trajectory at evenly spaced times, so its members
    are on the same orbit by construction, and every Planet Labs satellite is
    its own plane because that fleet has no lattice. Membership arrives through
    the ephemeris and is used as given.

    Earlier versions rediscovered planes here by clustering (a, i) within a
    tolerance. That was wrong twice over: it merged satellites that were not
    plane-mates, and it made an exact structural property into a knob.

    The representative is the member mean. Members agree on mean semi-major axis
    to 0.4 m and on nodal rate to 2e-8 deg/day, so the only quantity where the
    choice is visible is the osculating node, which carries a J2 short-period
    offset of up to 0.14 deg across a plane; averaging centres that.
    """
    n_p = int(pid.max()) + 1
    a_p = np.zeros(n_p)
    i_p = np.zeros(n_p)
    w_p = np.zeros(n_p)
    spread = dict(a=0.0, incl=0.0, raan=0.0)
    for g in range(n_p):
        sel = pid == g
        a_p[g] = a[sel].mean()
        i_p[g] = incl[sel].mean()
        w_p[g] = math.atan2(np.sin(raan0[sel]).mean(), np.cos(raan0[sel]).mean())
        if sel.sum() > 1:
            spread["a"] = max(spread["a"], float(np.ptp(a[sel])))
            spread["incl"] = max(spread["incl"],
                                 math.degrees(float(np.ptp(incl[sel]))))
            spread["raan"] = max(
                spread["raan"],
                math.degrees(np.abs(_wrap_pi(raan0[sel] - w_p[g])).max()))
    return a_p, i_p, w_p, spread


def plane_members(pid, n_planes):
    """CSR member lists, so the njit expansion needs no ragged structures."""
    counts = np.array([(pid == g).sum() for g in range(n_planes)], dtype=np.int64)
    start = np.zeros(n_planes, dtype=np.int64)
    start[1:] = np.cumsum(counts)[:-1]
    idx = np.concatenate([np.flatnonzero(pid == g) for g in range(n_planes)])
    return start, counts, idx.astype(np.int64)


@njit(cache=True, parallel=True)
def build_plane_table(a_p, incl_p, raan0_p, rate_p, dep_s, tof_s,
                      mass, isp, thrust, n_scan, n_ladder, max_growth,
                      min_growth):
    """Transfer delta-V between planes: (n_plane, n_plane, n_dep, n_tof), km/s.

    One Edelbaum solve per ordered plane pair, covering the whole epoch grid in
    a single call so the ring ladder is shared across it.
    """
    n_p = a_p.shape[0]
    n_dep = dep_s.shape[0]
    n_tof = tof_s.shape[0]
    out = np.full((n_p, n_p, n_dep, n_tof), np.nan)

    for t in prange(n_p * n_p):
        p = t // n_p
        q = t - p * n_p
        if p == q:
            continue                      # same plane: phasing only, no transfer

        s1 = np.empty(6)
        s2 = np.empty(6)
        s1[0] = a_p[p]; s1[1] = incl_p[p]; s1[2] = 0.0
        s2[0] = a_p[q]; s2[1] = incl_p[q]; s2[2] = 0.0
        for c in range(3, 6):
            s1[c] = (mass, isp, thrust)[c - 3]
            s2[c] = (mass, isp, thrust)[c - 3]

        d_raans = np.empty((n_dep, n_tof))
        for k in range(n_dep):
            dep = dep_s[k]
            node_p = raan0_p[p] + rate_p[p] * dep
            for m in range(n_tof):
                arr = dep + tof_s[m]
                d_raans[k, m] = _wrap_pi((raan0_p[q] + rate_p[q] * arr) - node_p)

        dv = transfer_cost_grid(s1, s2, tof_s, d_raans, n_scan, n_ladder,
                                max_growth, min_growth)
        for k in range(n_dep):
            for m in range(n_tof):
                out[p, q, k, m] = dv[k, m]
    return out


@njit(cache=True, parallel=True)
def expand_to_clients(plane_dv, pid, a, u0, u_rate, dep_s, tof_s,
                      incl_p, raan0_p, rate_p):
    """Copy plane entries onto every client pair and add the phasing manoeuvre.

    No transfer is solved here. The plane entry is the whole plane change; what
    is added is the along-track closure, which is the only thing two members of
    one plane do not share.
    """
    n = a.shape[0]
    n_dep = dep_s.shape[0]
    n_tof = tof_s.shape[0]
    dv_out = np.full((n, n, n_dep, n_tof), np.nan, dtype=np.float32)
    ph_out = np.zeros((n, n, n_dep, n_tof), dtype=np.float32)

    for i in prange(n):
        p = pid[i]
        for j in range(n):
            if i == j:
                continue
            q = pid[j]
            coplanar = abs(incl_p[p] - incl_p[q]) < SAME_PLANE_TOL
            for k in range(n_dep):
                dep = dep_s[k]
                ui = u0[i] + u_rate[i] * dep
                node_p = raan0_p[p] + rate_p[p] * dep
                for m in range(n_tof):
                    arr = dep + tof_s[m]
                    # The servicer carries the departure object's phase through
                    # the transfer, as cost/cost_functions.jl does.
                    dv_phase, dur = phasing(a[j], ui + u_rate[i] * tof_s[m],
                                            u0[j] + u_rate[j] * arr)
                    ph_out[i, j, k, m] = np.float32(dur)

                    if p == q:
                        dv_out[i, j, k, m] = np.float32(dv_phase * 1000.0)
                        continue

                    gap = _wrap_pi((raan0_p[q] + rate_p[q] * arr) - node_p)
                    if coplanar and abs(gap) < SAME_PLANE_TOL:
                        dv_out[i, j, k, m] = np.float32(dv_phase * 1000.0)
                    else:
                        t_dv = plane_dv[p, q, k, m]
                        if not np.isnan(t_dv):
                            dv_out[i, j, k, m] = np.float32(
                                (t_dv + dv_phase) * 1000.0)
    return dv_out, ph_out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", default="outputs/simulation.h5")
    p.add_argument("--out", default="outputs/cost_table.h5")
    p.add_argument("--step-s", type=float, default=GRID_STEP_S,
                   help="fine grid step [s], both axes")
    p.add_argument("--fine-until-s", type=float, default=FINE_UNTIL_S,
                   help="fine step applies out to here, coarse step beyond")
    p.add_argument("--coarse-step-s", type=float, default=COARSE_STEP_S)
    p.add_argument("--dep-horizon-s", type=float, default=DEP_HORIZON_S,
                   help="last departure epoch [s] relative to as_of")
    p.add_argument("--tof-horizon-s", type=float, default=TOF_HORIZON_S,
                   help="longest single leg [s]")
    p.add_argument("--limit", type=int, default=0,
                   help="build only the first N nodes, for timing")
    p.add_argument("--n-scan", type=int, default=180)
    p.add_argument("--n-ladder", type=int, default=1200)
    p.add_argument("--max-growth", type=float, default=edelbaum.MAX_GROWTH)
    p.add_argument("--min-growth", type=float, default=MIN_GROWTH)
    args = p.parse_args()

    import h5py

    nd = load_nodes(args.sim)
    dep_grid = two_rate_grid(args.step_s, args.fine_until_s, args.coarse_step_s,
                             args.dep_horizon_s)
    tof_grid = two_rate_grid(args.step_s, args.fine_until_s, args.coarse_step_s,
                             args.tof_horizon_s)
    sl = slice(0, args.limit) if args.limit else slice(None)
    names = nd.names[sl]
    n = len(names)
    a, incl = nd.a[sl], nd.incl[sl]
    raan0, rate = nd.raan0[sl], nd.raan_rate[sl]
    u0, u_rate = nd.u0[sl], nd.u_rate[sl]

    pid = nd.plane[sl]
    if (pid < 0).any():
        raise SystemExit(
            "the ephemeris carries no plane assignment. Rebuild the instance "
            "and the ephemeris: planes are declared by build_instance.py, not "
            "rediscovered here.")
    pid = pid - pid.min()
    a_p, i_p, w_p, spread = planes_from_instance(pid, a, incl, raan0)
    n_p = len(a_p)
    # Every member of a plane shares (a, i), so they share a nodal rate exactly;
    # take the representative's rather than averaging a derived quantity.
    rate_p = np.array([rate[np.flatnonzero(pid == g)[0]] for g in range(n_p)])

    print(f"nodes {n}   grid {len(dep_grid)} departures "
          f"({dep_grid[0]:.0f}-{dep_grid[-1]:.0f} s) x {len(tof_grid)} times of "
          f"flight ({tof_grid[0]:.0f}-{tof_grid[-1]:.0f} s)   "
          f"{len(dep_grid) * len(tof_grid):,} cells per pair")
    print(f"planes {n_p}  ->  {n_p * (n_p - 1):,} plane solves instead of "
          f"{n * (n - 1):,} client pairs ({n * (n - 1) / max(n_p * (n_p - 1), 1):.1f}x fewer)")
    sizes = np.bincount(pid)
    print(f"  plane occupancy: {sizes.max()} at most, "
          f"{(sizes > 1).sum()} planes with more than one member")
    print(f"  worst spread inside a plane: {spread['a']:.4f} km, "
          f"{spread['incl']:.5f} deg inclination, {spread['raan']:.5f} deg RAAN")
    print(f"servicer {servicer.MASS:.0f} kg, {servicer.THRUST_N * 1e3:.0f} mN, "
          f"Isp {servicer.ISP:.0f} s")

    t0 = time.time()
    plane_dv = build_plane_table(a_p, i_p, w_p, rate_p, dep_grid,
                                 tof_grid,
                                 servicer.MASS, servicer.ISP, servicer.THRUST,
                                 args.n_scan, args.n_ladder, args.max_growth,
                                 args.min_growth)
    t_plane = time.time() - t0
    print(f"plane table built in {t_plane:.1f} s "
          f"({1e3 * t_plane / max(n_p * (n_p - 1), 1):.1f} ms per plane pair)")

    t0 = time.time()
    dv, ph = expand_to_clients(plane_dv, pid, a, u0, u_rate, dep_grid,
                               tof_grid, i_p, w_p, rate_p)
    print(f"expanded to {n * (n - 1):,} client pairs in {time.time() - t0:.1f} s")

    finite = np.isfinite(dv)
    n_off = n * (n - 1) * len(dep_grid) * len(tof_grid)
    if finite.sum() == 0:
        raise SystemExit(
            "ABORT: not one entry is feasible. That is a numerical failure, not "
            "a result -- a table of NaNs is indistinguishable downstream from a "
            "table saying no transfer exists.")
    dead = int((~finite).all(axis=(1, 2, 3)).sum())
    if dead:
        print(f"WARNING: {dead} departure node(s) reach nothing, at any epoch.")

    print(f"feasible entries: {finite.sum():,} of {n_off:,} "
          f"({100.0 * finite.sum() / n_off:.1f} %)")
    v = dv[finite]
    print(f"delta-V m/s: min {v.min():.1f}  median {np.median(v):.1f}  "
          f"p90 {np.percentile(v, 90):.1f}  max {v.max():.1f}")
    print(f"within the {servicer.DV_BUDGET * 1000:.0f} m/s budget: "
          f"{100.0 * (v <= servicer.DV_BUDGET * 1000).mean():.1f} % of feasible")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("dv", data=dv, compression="lzf")
        f.create_dataset("phasing_s", data=ph, compression="lzf")
        f.create_dataset("names", data=np.array(names, dtype="S32"))
        f.create_dataset("plane_of_node", data=pid)
        f.create_dataset("departure_s", data=dep_grid)
        f.create_dataset("tof_s", data=tof_grid)
        f.attrs["units"] = "dv m/s, all time seconds relative to as_of"
        f.attrs["servicer_mass_kg"] = servicer.MASS
        f.attrs["servicer_isp_s"] = servicer.ISP
        f.attrs["servicer_thrust_N"] = servicer.THRUST_N
        f.attrs["dv_budget_m_s"] = servicer.DV_BUDGET * 1000.0
        f.attrs["source_sim"] = os.path.basename(args.sim)
        f.attrs["n_planes"] = n_p
        f.attrs["n_ladder"] = args.n_ladder
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
