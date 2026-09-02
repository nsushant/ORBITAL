"""In-plane phasing after the transfer, ported from `cost/lu_transfer.jl`.

The Edelbaum arc delivers the servicer to the client's *orbit*, not to the
client. Closing the remaining along-track gap costs a two-burn phasing
manoeuvre: drop into a slightly different orbit, let the phase difference wash
out over k revolutions, then circularise again.

k is the smallest revolution count for which the phasing orbit stays within
`DA_MAX` of the client's, so the delta-V is essentially pinned at that cap and
the phase gap is paid for in *time* rather than in propellant. This is why the
phase-angle defect of F11 shows up in arrival epochs and barely at all in the
delta-V budget.

The one substantive change from the Julia is the phase variable. The Julia
passed true anomalies; on orbits whose eccentricity is the J2 forced term the
perigee they are measured from is arbitrary and precesses at about 30 deg/day,
so the gap was wrong by the difference of the two arguments of perigee. The
argument of latitude, measured from the ascending node, is well defined however
small the eccentricity is. See F11.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import MU

DA_MAX = 50.0        # km, cap on the phasing orbit's semi-major-axis offset
K_MAX = 10000        # safety cap on the revolution count


@njit(cache=True)
def phasing(a0, u_from, u_to):
    """Return (delta-V [km/s], duration [s]) to close an along-track gap.

    `u_from` is where the servicer arrives in the target plane and `u_to` is
    where the client is; both are arguments of latitude in radians.
    """
    n0 = np.sqrt(MU / a0**3)
    period = 2.0 * np.pi / n0
    gap = (u_to - u_from) % (2.0 * np.pi)
    if gap < 1e-10:
        return 0.0, 0.0

    k = 1
    while k <= K_MAX:
        t_orb = period + gap / (n0 * k)
        a_p = (MU * (t_orb / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
        if abs(a_p - a0) <= DA_MAX:
            break
        k += 1

    t_orb = period + gap / (n0 * k)
    a_p = (MU * (t_orb / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    dv = 2.0 * abs(np.sqrt(MU * (2.0 / a0 - 1.0 / a_p)) - np.sqrt(MU / a0))
    return dv, k * t_orb


@njit(cache=True)
def phasing_batch(a0, u_from, u_to):
    """`phasing` over arrays of arrival phases. Returns (dv, duration) arrays."""
    n = u_from.shape[0]
    dv = np.empty(n)
    dur = np.empty(n)
    for i in range(n):
        dv[i], dur[i] = phasing(a0, u_from[i], u_to[i])
    return dv, dur
