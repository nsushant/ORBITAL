"""
Fast analytical transfer model: Edelbaum arcs either side of a ballistic J2 coast.

A transfer runs in three phases. A first low-thrust arc takes the servicer from
its orbit (a0, I0, RAAN0) to a drift orbit (a_d, I_d). The servicer then coasts,
and the RAAN separation closes under J2 nodal regression. A second arc puts it
on the target orbit (af, If). Only the two arcs cost propellant; the coast buys
RAAN for free, which is what makes the drift orbit worth searching for.

Arc cost is Edelbaum's relation for low-thrust transfer between circular orbits
with a plane change. Written in the velocity plane

    (x, y) = V * (cos(pi/2 * I), sin(pi/2 * I)),     V = sqrt(mu / a)

an arc is a straight chord and its delta-V is the Euclidean length of that
chord, which is exactly

    dV = sqrt(V1^2 + V2^2 - 2 V1 V2 cos(pi/2 * dI)).

The factor pi/2 on the inclination term presumes the optimal yaw programme over
the arc, so no explicit steering law is needed.

Choosing the drift orbit is therefore a search over points in the velocity
plane. Candidates are taken along the chord joining the two endpoints (no free
RAAN, cheapest arcs) and on ellipses with the endpoints as foci (progressively
more delta-V spent to reach a faster-drifting orbit). A candidate is admissible
only if the RAAN accumulated over the two arcs plus the coast closes the gap to
the target, modulo 2*pi. Among admissible candidates the cheapest is returned.

This is the model the optimiser runs on; `oos.lu` is the reference it is
validated against.

Units: km, s, kg, rad. Thrust is in kg*km/s^2, so 1 N = 1e-3.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import G0, J2, MU, R_E

# State vector layout, used throughout: [a, incl, raan, mass, isp, thrust].
A, INCL, RAAN, MASS, ISP, THRUST = 0, 1, 2, 3, 4, 5

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)

# Sentinel for "no per-entry RAAN gaps given"; numba needs a concrete array type.
EMPTY = np.empty(0)

# A drift orbit has to be an orbit. In the velocity plane a point at radius v
# maps to a = MU / v^2, so v -> 0 maps to a -> infinity and the origin itself is
# a division by zero; rings of large growth sweep straight through it. These
# bounds reject such points where drift orbits are evaluated, in `ring_bundles`
# and `drift_orbit_residual`. `from_velocity_plane` itself stays a pure
# coordinate map, so the geometric ring identity still holds everywhere.
# The bounds bite only on rings far outside anything affordable: delta-V on a
# ring is 2*(c + growth), so a growth of even 1 km/s already costs 2 km/s, four
# times the servicer's whole budget.
A_DRIFT_MIN = R_E + 150.0     # km
A_DRIFT_MAX = 10.0 * R_E      # km

# Ceiling on the ring search. Circular-orbit speeds in this problem are about
# 7.6 km/s, so a growth of 8 km/s already reaches the origin of the velocity
# plane; nothing beyond it is an orbit. It is also 32 times the largest growth
# the servicer could pay for.
MAX_GROWTH = 8.0              # km/s


@njit(cache=True)
def to_velocity_plane(a, incl):
    v = np.sqrt(MU / a)
    return v * np.cos(0.5 * np.pi * incl), v * np.sin(0.5 * np.pi * incl)


@njit(cache=True)
def from_velocity_plane(x, y):
    v2 = x * x + y * y
    if v2 <= 0.0:                      # the origin maps to an infinite orbit
        return np.inf, 0.0
    a = MU / v2
    incl = 2.0 / np.pi * np.arctan2(y, x)
    return a, incl


@njit(cache=True)
def wrap_pi(x):
    """Wrap an angle onto (-pi, pi]."""
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


@njit(cache=True)
def j2_raan_rate(a, incl):
    """Secular J2 nodal regression rate [rad/s] for a circular orbit."""
    n = np.sqrt(MU / a**3)
    return -1.5 * J2 * (R_E / a) ** 2 * n * np.cos(incl)


@njit(cache=True)
def arc_cost(a0, incl0, af, inclf, mass, isp, thrust):
    """Edelbaum arc: delta-V [km/s], burn duration [s], and final mass [kg]."""
    x0, y0 = to_velocity_plane(a0, incl0)
    xf, yf = to_velocity_plane(af, inclf)
    dv = np.hypot(xf - x0, yf - y0)
    v_exhaust = isp * G0
    duration = v_exhaust * (mass / thrust) * (1.0 - np.exp(-dv / v_exhaust))
    return dv, duration, mass * np.exp(-dv / v_exhaust)


@njit(cache=True)
def _raan_during_arc(a0, incl0, mass, isp, thrust, x0, y0, xf, yf, dv):
    """RAAN accumulated while thrusting along an arc [rad].

    The arc is parameterised by expended delta-V s in [0, dV]. Along it
    dOmega/ds = Omega_dot(a(s), I(s)) / f(s), with f = thrust/m(s) the current
    acceleration. Integrated by 16-point Gauss-Legendre, which is exact enough
    here because the integrand is smooth in s.
    """
    if dv <= 1e-15:
        return 0.0
    v_exhaust = isp * G0
    total = 0.0
    for k in range(_GL_NODES.shape[0]):
        s = dv * 0.5 * (_GL_NODES[k] + 1.0)
        weight = dv * 0.5 * _GL_WEIGHTS[k]
        frac = s / dv
        x = x0 + frac * (xf - x0)
        y = y0 + frac * (yf - y0)
        a_s, incl_s = from_velocity_plane(x, y)
        accel = thrust / (mass * np.exp(-s / v_exhaust))
        total += j2_raan_rate(a_s, incl_s) / accel * weight
    return total


@njit(cache=True)
def ring_point(growth, x1, y1, x2, y2, t):
    """One point on a candidate ring, at ring parameter t.

    `growth` = 0 is the chord joining the endpoints, parameterised by t in
    [0, 1]. `growth` > 0 is an ellipse with the endpoints as foci, parameterised
    by t in [0, 2*pi].
    """
    if growth == 0.0:
        return (1.0 - t) * x1 + t * x2, (1.0 - t) * y1 + t * y2

    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    dx, dy = x2 - x1, y2 - y1
    d = np.hypot(dx, dy)
    # Endpoints differing only in RAAN share a velocity-plane point. Grow a
    # circle about it rather than a degenerate ellipse.
    if d < 1e-12:
        ux, uy, c = 1.0, 0.0, 0.0
    else:
        ux, uy, c = dx / d, dy / d, 0.5 * d
    px, py = -uy, ux

    semi_major = c + growth
    semi_minor = np.sqrt(max(semi_major * semi_major - c * c, 0.0))
    return (cx + semi_major * np.cos(t) * ux + semi_minor * np.sin(t) * px,
            cy + semi_major * np.cos(t) * uy + semi_minor * np.sin(t) * py)


@njit(cache=True)
def ellipse_points(growth, x1, y1, x2, y2, num_points):
    """`num_points` samples of the ring, for inspection and plotting."""
    out = np.empty((num_points, 2))
    t_max = 1.0 if growth == 0.0 else 2.0 * np.pi
    for k in range(num_points):
        t = 0.0 if num_points == 1 else t_max * k / (num_points - 1.0)
        out[k, 0], out[k, 1] = ring_point(growth, x1, y1, x2, y2, t)
    return out


@njit(cache=True)
def drift_orbit_residual(xd, yd, state1, state2, tof, d_raan):
    """RAAN closure residual for a drift orbit, and the coast it implies.

    Returns (residual [rad], coast duration [s]). The residual is NaN when the
    two arcs do not fit inside the time of flight, which is the one hard
    feasibility limit; everything else is a matter of hitting the right node.
    """
    a_d, incl_d = from_velocity_plane(xd, yd)

    x1, y1 = to_velocity_plane(state1[A], state1[INCL])
    dv1, dt1, mass_d = arc_cost(state1[A], state1[INCL], a_d, incl_d,
                                state1[MASS], state1[ISP], state1[THRUST])
    if tof - dt1 <= 0.0:
        return np.nan, np.nan
    raan1 = _raan_during_arc(state1[A], state1[INCL], state1[MASS],
                             state1[ISP], state1[THRUST], x1, y1, xd, yd, dv1)

    x2, y2 = to_velocity_plane(state2[A], state2[INCL])
    dv2, dt2, _ = arc_cost(a_d, incl_d, state2[A], state2[INCL],
                           mass_d, state1[ISP], state1[THRUST])
    raan2 = _raan_during_arc(a_d, incl_d, mass_d, state1[ISP], state1[THRUST],
                             xd, yd, x2, y2, dv2)

    t_coast = tof - dt1 - dt2
    if t_coast < 0.0:
        return np.nan, np.nan

    delivered = raan1 + j2_raan_rate(a_d, incl_d) * t_coast + raan2
    return wrap_pi(delivered - d_raan), t_coast


@njit(cache=True)
def ring_bundles(growth, x1, y1, x2, y2, state1, state2, n_scan):
    """Arc quantities for each sampled drift orbit on a ring.

    None of this depends on the time of flight - only the coast between the two
    arcs does - so a ring is evaluated once and then tested against every time
    of flight. Columns: total delta-V, first-arc duration, both-arc duration,
    RAAN accrued while thrusting, and the drift orbit's nodal rate.
    """
    out = np.empty((n_scan + 1, 5))
    xs1, ys1 = to_velocity_plane(state1[A], state1[INCL])
    xs2, ys2 = to_velocity_plane(state2[A], state2[INCL])
    t_max = 1.0 if growth == 0.0 else 2.0 * np.pi

    for k in range(n_scan + 1):
        t = t_max * k / n_scan
        xd, yd = ring_point(growth, x1, y1, x2, y2, t)
        a_d, incl_d = from_velocity_plane(xd, yd)

        if a_d < A_DRIFT_MIN or a_d > A_DRIFT_MAX:
            out[k, 0] = np.nan
            out[k, 1] = np.nan
            out[k, 2] = np.nan
            out[k, 3] = np.nan
            out[k, 4] = np.nan
            continue

        dv1, dt1, mass_d = arc_cost(state1[A], state1[INCL], a_d, incl_d,
                                    state1[MASS], state1[ISP], state1[THRUST])
        raan1 = _raan_during_arc(state1[A], state1[INCL], state1[MASS],
                                 state1[ISP], state1[THRUST], xs1, ys1, xd, yd, dv1)
        dv2, dt2, _ = arc_cost(a_d, incl_d, state2[A], state2[INCL],
                               mass_d, state1[ISP], state1[THRUST])
        raan2 = _raan_during_arc(a_d, incl_d, mass_d, state1[ISP], state1[THRUST],
                                 xd, yd, xs2, ys2, dv2)

        out[k, 0] = dv1 + dv2
        out[k, 1] = dt1
        out[k, 2] = dt1 + dt2
        out[k, 3] = raan1 + raan2
        out[k, 4] = j2_raan_rate(a_d, incl_d)
    return out


@njit(cache=True)
def _residual_from_bundle(bundles, k, tof, d_raan):
    """Closure residual of sampled drift orbit k at this time of flight."""
    if tof - bundles[k, 1] <= 0.0:      # the first arc alone overruns the horizon
        return np.nan
    t_coast = tof - bundles[k, 2]
    if t_coast < 0.0:
        return np.nan
    return wrap_pi(bundles[k, 3] + bundles[k, 4] * t_coast - d_raan)


@njit(cache=True)
def _closes_from_bundles(bundles, growth, x1, y1, x2, y2, state1, state2,
                         tof, d_raan, n_scan):
    """Does some drift orbit on this ring close RAAN exactly at this horizon?

    Sign changes of the residual bracket a root, except where the residual steps
    by 2*pi, which is the wrap rather than a crossing. Brackets are bisected on
    the ring parameter, so closure is solved for rather than tested against a
    tolerance.
    """
    t_max = 1.0 if growth == 0.0 else 2.0 * np.pi
    prev_r = np.nan
    prev_t = 0.0
    for k in range(n_scan + 1):
        t = t_max * k / n_scan
        r = _residual_from_bundle(bundles, k, tof, d_raan)

        if not np.isnan(r) and not np.isnan(prev_r):
            if r == 0.0:
                return True
            if prev_r * r < 0.0 and np.abs(r - prev_r) < np.pi:
                lo_t, lo_r = prev_t, prev_r
                hi_t = t
                converged = True
                for _ in range(60):
                    mid_t = 0.5 * (lo_t + hi_t)
                    mx, my = ring_point(growth, x1, y1, x2, y2, mid_t)
                    mr, _ = drift_orbit_residual(mx, my, state1, state2, tof, d_raan)
                    if np.isnan(mr):
                        converged = False
                        break
                    if (mr < 0.0) == (lo_r < 0.0):
                        lo_t, lo_r = mid_t, mr
                    else:
                        hi_t = mid_t
                if converged:
                    return True
        prev_r, prev_t = r, t
    return False


@njit(cache=True)
def ring_closes(growth, x1, y1, x2, y2, state1, state2, tof, d_raan, n_scan):
    """Single-horizon wrapper, kept for tests and for one-off queries."""
    bundles = ring_bundles(growth, x1, y1, x2, y2, state1, state2, n_scan)
    return _closes_from_bundles(bundles, growth, x1, y1, x2, y2, state1, state2,
                                tof, d_raan, n_scan)


@njit(cache=True)
def transfer_cost(state1, state2, tofs, d_raans=EMPTY, n_scan=180, n_growth=48,
                  max_growth=MAX_GROWTH, growth_rtol=1e-3):
    """Minimum delta-V [km/s] for one orbit pair over several times of flight.

    Every point on a ring costs the same: the ring is an ellipse with the two
    endpoints as foci, and the two arcs are the distances from the point to each
    focus, so their sum is exactly 2*(c + growth). The transfer delta-V depends
    only on which ring the drift orbit lies on, so the search is for the
    *smallest* ring carrying a drift orbit that closes RAAN - the method
    Section 3.2.4 describes. The chord (growth = 0) is the direct transfer.

    Feasibility is not monotone in growth: too small a ring cannot buy enough
    nodal drift, too large a one spends so long thrusting that the arcs no
    longer fit the horizon. So the ladder is scanned outward for the first ring
    that closes, then bisected against the last that did not.

    Each ring is evaluated once and tested against every time of flight, since
    the arcs do not depend on the horizon.

    `tofs` in seconds. States are [a, incl, raan, mass, isp, thrust].
    `d_raans`, if given, is the required RAAN change for each entry of `tofs`;
    it defaults to `state2[RAAN] - state1[RAAN]` for all of them. A cost table
    needs one entry per (departure epoch, arrival epoch), and the nodes of both
    objects have drifted by different amounts at each departure, so the required
    RAAN change differs entry by entry while the orbits do not. The ring bundles
    depend only on the orbits, so passing every entry for an object pair in one
    call shares that work across all of them.
    NaN where no drift orbit closes RAAN within the time of flight.
    """
    n_tofs = tofs.shape[0]
    best = np.full(n_tofs, np.nan)
    per_entry = d_raans.shape[0] == n_tofs
    default_raan = state2[RAAN] - state1[RAAN]

    x1, y1 = to_velocity_plane(state1[A], state1[INCL])
    x2, y2 = to_velocity_plane(state2[A], state2[INCL])
    half_separation = 0.5 * np.hypot(x2 - x1, y2 - y1)

    ladder = 10.0 ** np.linspace(-2.0, np.log10(max_growth), n_growth)

    # Largest growth known not to close, and smallest known to close, per horizon.
    lo = np.zeros(n_tofs)
    hi = np.full(n_tofs, -1.0)
    resolved = np.zeros(n_tofs, dtype=np.bool_)

    chord = ring_bundles(0.0, x1, y1, x2, y2, state1, state2, n_scan)
    for k in range(n_tofs):
        gap = d_raans[k] if per_entry else default_raan
        if _closes_from_bundles(chord, 0.0, x1, y1, x2, y2, state1, state2,
                                tofs[k], gap, n_scan):
            best[k] = 2.0 * half_separation      # the direct transfer already closes
            resolved[k] = True

    for g in range(ladder.shape[0]):
        if np.all(resolved):
            break
        bundles = ring_bundles(ladder[g], x1, y1, x2, y2, state1, state2, n_scan)
        for k in range(n_tofs):
            if resolved[k] or hi[k] > 0.0:
                continue
            gap = d_raans[k] if per_entry else default_raan
            if _closes_from_bundles(bundles, ladder[g], x1, y1, x2, y2, state1,
                                    state2, tofs[k], gap, n_scan):
                hi[k] = ladder[g]
            else:
                lo[k] = ladder[g]

    for k in range(n_tofs):
        if resolved[k] or hi[k] < 0.0:
            continue
        gap = d_raans[k] if per_entry else default_raan
        lo_k, hi_k = lo[k], hi[k]
        # growth_rtol <= 0 skips refinement, leaving the ladder to set the
        # resolution. The iteration cap is a guard, not a tuning knob: once the
        # midpoint rounds to an endpoint the interval stops shrinking, and a
        # tolerance of zero would otherwise spin forever.
        refinements = 0
        while (growth_rtol > 0.0 and hi_k - lo_k > growth_rtol * hi_k
               and refinements < 200):
            refinements += 1
            mid = 0.5 * (lo_k + hi_k)
            bundles = ring_bundles(mid, x1, y1, x2, y2, state1, state2, n_scan)
            if _closes_from_bundles(bundles, mid, x1, y1, x2, y2, state1, state2,
                                    tofs[k], gap, n_scan):
                hi_k = mid
            else:
                lo_k = mid
        best[k] = 2.0 * (half_separation + hi_k)

    return best


def transfer_dv(a0, incl0, raan0, af, inclf, raanf, tof_days,
                mass=335.0, isp=2800.0, thrust=1e-5, **kwargs):
    """Convenience single-transfer wrapper. Returns delta-V [m/s], or None.

    Defaults are the Otter-class servicer of the paper: 335 kg, Isp 2800 s,
    10 mN (1e-5 kg*km/s^2).
    """
    s1 = np.array([a0, incl0, raan0, mass, isp, thrust])
    s2 = np.array([af, inclf, raanf, mass, isp, thrust])
    dv = transfer_cost(s1, s2, np.array([tof_days * 86400.0]), **kwargs)[0]
    return None if np.isnan(dv) else dv * 1000.0
