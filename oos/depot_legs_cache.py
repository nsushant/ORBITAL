"""One-time build and random access of the (a, i) grid's depot legs.

`depot_legs` depends only on the depot orbit (a, i), the fixed client nodes,
and the servicer -- never on the scenario (S1/S2/S3) or its demands. The three
scenario sweeps therefore share one cached copy of every grid location's legs
instead of recomputing the identical transfer ladders three times.

The cache also skips what no budget can reach. Edelbaum's chord is a strict
lower bound on the transfer cost between two orbits: every ring (growth >= 0)
costs 2*(half_separation + growth) >= 2*half_separation = chord. So a client
whose chord from the depot exceeds the delta-V budget can never appear in a
feasible schedule, and its legs are all-NaN by construction -- no ladder to
scan. On the 391-location study grid only the shell-aligned lanes pass the
filter (~140-170 of 391), and within a live location only the 48-160 clients
on the matching shell keep working entries.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import h5py
import numpy as np

from .constants import MU
from .depot_legs import depot_legs

_CT = None
_NODES = None
_A_KM = None
_INCL = None
_BUDGET = None


def reachable_node_ids(a_km, incl_rad, nodes, budget_m_s):
    """Client nodes whose Edelbaum chord to the depot lies within the budget.

    Any feasible transfer costs at least the chord, so this is an exact upper
    bound on the set of nodes `depot_legs` can return a finite entry for.
    """
    v0 = np.sqrt(MU / a_km)
    x0 = v0 * np.cos(0.5 * np.pi * incl_rad)
    y0 = v0 * np.sin(0.5 * np.pi * incl_rad)
    v1 = np.sqrt(MU / nodes.a)
    x1 = v1 * np.cos(0.5 * np.pi * nodes.incl)
    y1 = v1 * np.sin(0.5 * np.pi * nodes.incl)
    chord = np.hypot(x1 - x0, y1 - y0)
    return np.flatnonzero(chord <= budget_m_s / 1000.0)


def _build_one(idx):
    a = float(_A_KM[idx])
    inc = float(_INCL[idx])
    inc_r = np.radians(inc)
    live = reachable_node_ids(a, inc_r, _NODES, _BUDGET)
    if live.size == 0:
        return None
    sub = SimpleNamespace(
        a=_NODES.a[live], incl=_NODES.incl[live],
        raan0=_NODES.raan0[live], raan_rate=_NODES.raan_rate[live],
        u0=_NODES.u0[live], u_rate=_NODES.u_rate[live])
    dv, ph = depot_legs(sub, a, inc_r, _CT.dep_days,
                        dep_days=_CT.dep_days, tof_days=_CT.tof_days)
    n = len(_NODES)
    n_dep = dv.shape[-2]
    n_tof = dv.shape[-1]
    full_dv = np.full((2, n, n_dep, n_tof), np.nan, dtype=np.float32)
    full_ph = np.zeros((2, n, n_dep, n_tof), dtype=np.float32)
    full_dv[:, live] = dv
    full_ph[:, live] = ph
    return (idx, a, inc, full_dv, full_ph)


def build_cache(ct, nodes, a_km, incl_deg, budget_m_s, out_path, jobs=4):
    """Build `out_path` with the depot legs of every live grid location.

    Rows are the live locations, in grid order: `a_km`, `incl_deg`, then
    `dv`/`ph` of shape (2, n_nodes, n_dep, n_tof) float32, gzip-compressed.
    """
    global _CT, _NODES, _A_KM, _INCL, _BUDGET
    _CT, _NODES, _A_KM, _INCL, _BUDGET = ct, nodes, a_km, incl_deg, float(budget_m_s)

    seats = list(range(len(a_km)))
    results = []
    if jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(jobs) as pool:
            for r in pool.imap_unordered(_build_one, seats, chunksize=4):
                if r is not None:
                    results.append(r)
    else:
        for i in seats:
            r = _build_one(i)
            if r is not None:
                results.append(r)
    results.sort(key=lambda r: r[0])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = len(results)
    with h5py.File(out_path, "w") as f:
        f.attrs["n_nodes"] = len(nodes)
        f.attrs["dep_days"] = np.asarray(ct.dep_days)
        f.attrs["tof_days"] = np.asarray(ct.tof_days)
        f.attrs["dv_budget_m_s"] = float(budget_m_s)
        f.create_dataset("grid_a_km", data=np.asarray(a_km, dtype=np.float64))
        f.create_dataset("grid_incl_deg", data=np.asarray(incl_deg, dtype=np.float64))
        f.create_dataset("a_km", data=np.array([r[1] for r in results]))
        f.create_dataset("incl_deg", data=np.array([r[2] for r in results]))
        if n:
            shape = (n, 2, len(nodes), len(ct.dep_days), len(ct.tof_days))
            chunk = (1, 2, 1, len(ct.dep_days), len(ct.tof_days))
            dv = f.create_dataset("dv", shape, dtype=np.float32,
                                  chunks=chunk, compression="gzip")
            ph = f.create_dataset("ph", shape, dtype=np.float32,
                                  chunks=chunk, compression="gzip")
            for k, r in enumerate(results):
                dv[k] = r[3]
                ph[k] = r[4]
    return out_path


class LegsCache:
    """Read-only random access to a built legs cache.

    The H5 handle is opened lazily on first read so a pool of forked workers
    each gets its own, rather than inheriting a shared one.
    """

    def __init__(self, path):
        self.path = path
        self._fh = None
        with h5py.File(path, "r") as f:
            self.a_km = np.asarray(f["a_km"][...])
            self.incl_deg = np.asarray(f["incl_deg"][...])
            self.n_nodes = int(f.attrs["n_nodes"])
            self.dep_days = np.asarray(f.attrs["dep_days"])
            self.tof_days = np.asarray(f.attrs["tof_days"])
            self.dv_budget = float(f.attrs["dv_budget_m_s"])
            grid_a = np.asarray(f["grid_a_km"][...]).tolist()
            grid_i = np.asarray(f["grid_incl_deg"][...]).tolist()
            self._grid = set(zip(grid_a, grid_i))
        self._index = {
            (float(a), float(i)): r
            for r, (a, i) in enumerate(zip(self.a_km, self.incl_deg))}

    def check(self, n_nodes, dep_days, tof_days, budget_m_s):
        return (self.n_nodes == n_nodes
                and np.array_equal(self.dep_days, np.asarray(dep_days))
                and np.array_equal(self.tof_days, np.asarray(tof_days))
                and self.dv_budget == float(budget_m_s))

    def covers(self, a_km, incl_deg):
        """True iff every requested grid point exists in the cache's grid."""
        return all((float(a), float(i)) in self._grid
                   for a, i in zip(np.asarray(a_km), np.asarray(incl_deg)))

    def _open(self):
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def get(self, a_km, incl_deg):
        """(dv, ph) for a grid location, or None when that location is dead."""
        r = self._index.get((float(a_km), float(incl_deg)))
        if r is None:
            return None
        f = self._open()
        return f["dv"][r], f["ph"][r]