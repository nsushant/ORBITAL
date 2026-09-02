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

# A drift orbit is admissible when the RAAN it delivers matches the required gap
# to within this tolerance. The candidate set is discrete, so exact closure is
# not attainable; 0.01 rad (0.57 deg) is tight against a nodal gap of tens of
# degrees and loose enough for a 30-point ring to contain a hit.
RAAN_CLOSURE_TOL = 1e-2

# State vector layout, used throughout: [a, incl, raan, mass, isp, thrust].
A, INCL, RAAN, MASS, ISP, THRUST = 0, 1, 2, 3, 4, 5

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


@njit(cache=True)
def to_velocity_plane(a, incl):
    v = np.sqrt(MU / a)
    return v * np.cos(0.5 * np.pi * incl), v * np.sin(0.5 * np.pi * incl)


@njit(cache=True)
def from_velocity_plane(x, y):
    a = MU / (x * x + y * y)
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
def evaluate_drift_orbit(xd, yd, state1, state2, tof, d_raan):
    """Total delta-V [km/s] through drift orbit (xd, yd), or NaN if inadmissible.

    Inadmissible means either the two arcs do not fit inside the time of flight,
    or the RAAN delivered over arc-coast-arc does not close the gap.
    """
    a_d, incl_d = from_velocity_plane(xd, yd)

    x1, y1 = to_velocity_plane(state1[A], state1[INCL])
    dv1, dt1, mass_d = arc_cost(state1[A], state1[INCL], a_d, incl_d,
                                state1[MASS], state1[ISP], state1[THRUST])
    if tof - dt1 <= 0.0:
        return np.nan
    raan1 = _raan_during_arc(state1[A], state1[INCL], state1[MASS],
                             state1[ISP], state1[THRUST], x1, y1, xd, yd, dv1)

    x2, y2 = to_velocity_plane(state2[A], state2[INCL])
    dv2, dt2, _ = arc_cost(a_d, incl_d, state2[A], state2[INCL],
                           mass_d, state1[ISP], state1[THRUST])
    raan2 = _raan_during_arc(a_d, incl_d, mass_d, state1[ISP], state1[THRUST],
                             xd, yd, x2, y2, dv2)

    t_coast = tof - dt1 - dt2
    if t_coast < 0.0:
        return np.nan

    delivered = raan1 + j2_raan_rate(a_d, incl_d) * t_coast + raan2
    if np.abs(wrap_pi(delivered - d_raan)) > RAAN_CLOSURE_TOL:
        return np.nan
    return dv1 + dv2


@njit(cache=True)
def ellipse_points(growth, x1, y1, x2, y2, num_points):
    """Points on an ellipse with foci (x1, y1) and (x2, y2).

    `growth` is how far the semi-major axis exceeds the half focal separation,
    so growth = 0 degenerates to the chord between the foci, sampled uniformly.
    Larger growth reaches drift orbits further from both endpoints: more
    delta-V, but a faster nodal drift.
    """
    out = np.empty((num_points, 2))
    if growth == 0.0:
        for k in range(num_points):
            t = 0.0 if num_points == 1 else k / (num_points - 1.0)
            out[k, 0] = (1.0 - t) * x1 + t * x2
            out[k, 1] = (1.0 - t) * y1 + t * y2
        return out

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
    for k in range(num_points):
        t = 0.0 if num_points == 1 else 2.0 * np.pi * k / (num_points - 1.0)
        out[k, 0] = cx + semi_major * np.cos(t) * ux + semi_minor * np.sin(t) * px
        out[k, 1] = cy + semi_major * np.cos(t) * uy + semi_minor * np.sin(t) * py
    return out


@njit(cache=True)
def _candidates(x1, y1, x2, y2, num_points, growths):
    out = np.empty((growths.shape[0] * num_points, 2))
    for g in range(growths.shape[0]):
        ring = ellipse_points(growths[g], x1, y1, x2, y2, num_points)
        out[g * num_points:(g + 1) * num_points, :] = ring
    return out


@njit(cache=True)
def transfer_cost(state1, state2, tofs, num_points=30, n_rings=20, max_growth=4096.0):
    """Minimum delta-V [km/s] for one orbit pair over several times of flight.

    Drift-orbit candidates are expensive to evaluate and independent of the time
    of flight only up to the coast duration, so the chord and the first rings are
    tried for every time of flight before the rings are widened. Times of flight
    with no admissible drift orbit come back as NaN.

    `tofs` in seconds. `state1`, `state2` are [a, incl, raan, mass, isp, thrust].
    """
    n_tofs = tofs.shape[0]
    best = np.full(n_tofs, np.nan)
    d_raan = state2[RAAN] - state1[RAAN]

    x1, y1 = to_velocity_plane(state1[A], state1[INCL])
    x2, y2 = to_velocity_plane(state2[A], state2[INCL])

    growth_limit = 8.0
    points = ellipse_points(0.0, x1, y1, x2, y2, num_points)

    while True:
        for k in range(n_tofs):
            if not np.isnan(best[k]):
                continue
            local = np.inf
            for c in range(points.shape[0]):
                dv = evaluate_drift_orbit(points[c, 0], points[c, 1],
                                          state1, state2, tofs[k], d_raan)
                if not np.isnan(dv) and dv < local:
                    local = dv
            if local < np.inf:
                best[k] = local

        if not np.any(np.isnan(best)) or growth_limit > max_growth:
            break

        growths = 10.0 ** np.linspace(np.log10(growth_limit / 8.0),
                                      np.log10(growth_limit), n_rings)
        points = _candidates(x1, y1, x2, y2, num_points, growths)
        growth_limit *= 8.0

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
