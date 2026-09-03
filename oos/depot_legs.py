"""Rebuild only the depot legs of the cost table for a trial depot location.

The (a, i) sweep of Section 5 moves the depot and leaves the clients alone, so
the client-to-client block of the table never changes. Only the 2*(N-1) ordered
pairs that touch the depot do, and rebuilding those is a couple of minutes
rather than the hours the full table costs. This is what makes a grid of depot
locations affordable at all.

The trial depot is a circular orbit at the given (a, i) with the nominal node.
Its secular node rate and argument-of-latitude rate come from the closed forms
in `oos.nodes` rather than from a fresh propagation: for a circular orbit those
are exact to the same order as the Edelbaum model that consumes them, and
re-propagating 225 nodes per grid point to move one of them would be waste.

Usage
-----
    from oos.depot_legs import depot_legs
    dv, ph = depot_legs(nodes, a_km, incl_rad, grid_days)

`dv` and `ph` are (2, N, n_dep, n_tof): index 0 is depot -> client, index 1 is
client -> depot, in the node order of `nodes`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import servicer
from .constants import J2, MU, R_E
from .costtable import SAME_PLANE_TOL, _wrap_pi
from .edelbaum import MAX_GROWTH, MIN_GROWTH, transfer_cost_grid
from .nodes import draconitic_rate
from .phasing import phasing

DAY = 86400.0


def depot_secular(a, incl, raan0=0.0, u0=0.0):
    """(raan0, raan_rate, u0, u_rate) for a circular depot orbit."""
    n = np.sqrt(MU / a**3)
    raan_rate = -1.5 * J2 * (R_E / a) ** 2 * n * np.cos(incl)
    return raan0, raan_rate, u0, float(draconitic_rate(a, incl))


@njit(cache=True, parallel=True)
def _legs(a, incl, raan0, raan_rate, u0, u_rate,
          a_d, incl_d, raan0_d, raan_rate_d, u0_d, u_rate_d,
          dep_s, tof_s, mass, isp, thrust,
          n_scan, n_ladder, max_growth, min_growth):
    n = a.shape[0]
    n_dep = dep_s.shape[0]
    n_tof = tof_s.shape[0]

    dv_out = np.full((2, n, n_dep, n_tof), np.nan, dtype=np.float32)
    ph_out = np.zeros((2, n, n_dep, n_tof), dtype=np.float32)

    for j in prange(n):
        s_dep = np.empty(6)
        s_cli = np.empty(6)
        s_dep[0] = a_d;    s_dep[1] = incl_d;  s_dep[2] = 0.0
        s_cli[0] = a[j];   s_cli[1] = incl[j]; s_cli[2] = 0.0
        for c in range(3, 6):
            s_dep[c] = (mass, isp, thrust)[c - 3]
            s_cli[c] = (mass, isp, thrust)[c - 3]

        coplanar = abs(incl_d - incl[j]) < SAME_PLANE_TOL

        dr_out = np.empty((n_dep, n_tof))
        dr_ret = np.empty((n_dep, n_tof))
        for p in range(n_dep):
            dep = dep_s[p]
            for q in range(n_tof):
                arr = dep + tof_s[q]
                dr_out[p, q] = _wrap_pi((raan0[j] + raan_rate[j] * arr)
                                        - (raan0_d + raan_rate_d * dep))
                dr_ret[p, q] = _wrap_pi((raan0_d + raan_rate_d * arr)
                                        - (raan0[j] + raan_rate[j] * dep))

        dv_o = transfer_cost_grid(s_dep, s_cli, tof_s, dr_out,
                                  n_scan, n_ladder, max_growth, min_growth)
        dv_r = transfer_cost_grid(s_cli, s_dep, tof_s, dr_ret,
                                  n_scan, n_ladder, max_growth, min_growth)

        for p in range(n_dep):
            dep = dep_s[p]
            for q in range(n_tof):
                arr = dep + tof_s[q]

                # depot -> client: the servicer carries the depot's phase across
                dvp, dur = phasing(a[j], u0_d + u_rate_d * arr,
                                   u0[j] + u_rate[j] * arr)
                if coplanar and abs(dr_out[p, q]) < SAME_PLANE_TOL:
                    dv_out[0, j, p, q] = np.float32(dvp * 1000.0)
                    ph_out[0, j, p, q] = np.float32(dur / DAY)
                elif not np.isnan(dv_o[p, q]):
                    dv_out[0, j, p, q] = np.float32((dv_o[p, q] + dvp) * 1000.0)
                    ph_out[0, j, p, q] = np.float32(dur / DAY)

                # client -> depot
                dvp, dur = phasing(a_d, u0[j] + u_rate[j] * arr,
                                   u0_d + u_rate_d * arr)
                if coplanar and abs(dr_ret[p, q]) < SAME_PLANE_TOL:
                    dv_out[1, j, p, q] = np.float32(dvp * 1000.0)
                    ph_out[1, j, p, q] = np.float32(dur / DAY)
                elif not np.isnan(dv_r[p, q]):
                    dv_out[1, j, p, q] = np.float32((dv_r[p, q] + dvp) * 1000.0)
                    ph_out[1, j, p, q] = np.float32(dur / DAY)

    return dv_out, ph_out


def depot_legs(nodes, a_km, incl_rad, grid_days, raan0=0.0, u0=0.0,
               n_scan=180, n_ladder=1200, max_growth=MAX_GROWTH,
               min_growth=MIN_GROWTH):
    """Depot-to-client and client-to-depot legs for one trial depot location."""
    r0, rrate, uu0, urate = depot_secular(a_km, incl_rad, raan0, u0)
    grid_s = np.asarray(grid_days, dtype=float) * DAY
    return _legs(nodes.a, nodes.incl, nodes.raan0, nodes.raan_rate,
                 nodes.u0, nodes.u_rate,
                 a_km, incl_rad, r0, rrate, uu0, urate,
                 grid_s, grid_s,
                 servicer.MASS, servicer.ISP, servicer.THRUST,
                 n_scan, n_ladder, max_growth, min_growth)
