"""
Low-thrust transfer between circular orbits, after

    S. Lu, C. Zhang, et al., "Low-thrust transfer solution between circular
    orbits based on yaw switch steering and analytical propagation",
    Acta Astronautica 245 (2026) 924-936.

The transfer is a three-phase trajectory: a thrusting arc to a coasting orbit,
a ballistic J2 coast during which RAAN drifts to the target, and a second
thrusting arc onto the final orbit.

Two levels are implemented, matching the paper's two sections:

  Section 4.2  `coasting_orbit_estimate` — impulsive estimate. The coasting
               orbit is found in closed form (Case 1, pure RAAN change) or by a
               one-dimensional minimisation under the RAAN-closure constraint
               (Case 2, general). Cheap; benchmarked against the paper's Table 2.

  Section 4.3  `nlp_transfer_cost` — continuous-thrust refinement. Given the
               coasting orbit, the yaw angles and phase durations are optimised
               by SLSQP against the averaged dynamics of Section 3. Expensive;
               benchmarked against the paper's Tables 4, 6 and 8.

This module is the *reference* model. It validates the fast analytical model in
`oos.edelbaum`, which is what the optimiser actually runs on.

Angles are radians, semi-major axes km, times seconds unless a name says days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .constants import DAY, J2, K_J2, MU, R_E

# Below ~200 km altitude a multi-day J2 drift is meaningless (drag dominates),
# and the unconstrained estimator will happily return sub-surface coasting
# orbits with a plausible-looking delta-V.
A_MIN = R_E + 200.0

# Paper value for the maximum thrust acceleration, km/s^2.
FMAX_DEFAULT = 3.5e-6

# Returned when no admissible transfer exists. Far above any usable budget, so
# a caller that sums costs still gets a finite number that cannot be selected.
INFEASIBLE = 2e7


def wrap_pi(x: float) -> float:
    """Wrap an angle onto (-pi, pi].

    RAAN closure is only defined mod 2*pi, so an unwrapped difference can demand
    a 340-degree plane change where 20 degrees would do.
    """
    return math.remainder(x, 2.0 * math.pi)


def raan_drift_rate(a: float, incl: float) -> float:
    """Secular J2 nodal regression rate [rad/s]."""
    return -1.5 * J2 * R_E**2 * math.sqrt(MU / a**7) * math.cos(incl)


def circular_velocity(a: float) -> float:
    return math.sqrt(MU / a)


# ---------------------------------------------------------------------------
# Section 4.2 — impulsive coasting-orbit estimate
# ---------------------------------------------------------------------------


@dataclass
class CoastingOrbit:
    """Result of the Section 4.2 estimate. Velocity increments in km/s."""

    a_c: float          # coasting semi-major axis [km]
    incl_c: float       # coasting inclination [rad]
    jt: float           # transfer-arc velocity increment
    ja: float           # adjustment-arc velocity increment

    @property
    def j_total(self) -> float:
        return self.jt + self.ja


def _arc_increments(a0, incl0, af, inclf, da_transfer, di_transfer):
    """Velocity increments of the two thrusting arcs (Eqs. 52-53)."""
    a_bar = 0.5 * (a0 + af)
    v_bar = 0.5 * (circular_velocity(a0) + circular_velocity(af))
    da_adjust = (af - a0) - da_transfer
    di_adjust = (inclf - incl0) - di_transfer
    jt = v_bar * math.hypot(da_transfer / (2.0 * a_bar), di_transfer)
    ja = v_bar * math.hypot(da_adjust / (2.0 * a_bar), di_adjust)
    return jt, ja


def _case1_raan_only(a0, incl0, d_raan, tf_sec) -> CoastingOrbit:
    """Closed form for a pure RAAN change, a0 == af and incl0 == inclf (Eq. 51)."""
    v0 = circular_velocity(a0)
    omega_dot = raan_drift_rate(a0, incl0)

    denom = (
        49.0 / 2.0
        + math.tan(incl0) ** 2 / 2.0
        + 2.0 / (math.sin(incl0) ** 2 * (tf_sec * omega_dot) ** 2)
    )
    lam = -d_raan / (tf_sec * omega_dot * denom)

    x1 = 7.0 * lam
    x2 = 0.5 * math.tan(incl0) * lam
    x3 = -2.0 / (math.sin(incl0) ** 2 * tf_sec * omega_dot) * lam

    da = x1 * a0
    di = x2
    jt = v0 * math.sqrt(
        (da / (2.0 * a0)) ** 2 + di**2 + (x3 * math.sin(incl0) / 2.0) ** 2
    )
    # The two arcs are symmetric for a pure RAAN change.
    return CoastingOrbit(a_c=a0 + da, incl_c=incl0 + di, jt=jt, ja=jt)


# Eq. 44 linearises the nodal drift rate about a reference semi-major axis. The
# paper is ambiguous about which one: its reported Case-2 coasting orbit implies
# the initial value a0, while its reported Case-2 delta-V implies the mean value
# (a0+af)/2. Case 3 discriminates — a0 reproduces the published coasting orbit to
# 0.02 km against 2.45 km for the mean — so a0 is the default. See
# docs/lu_normalisation.md.
EQ44_NORMALISATION = "a0"     # or "a_bar"


def _case2_general(a0, incl0, af, inclf, d_raan, tf_sec, a_min=A_MIN,
                   eq44_normalisation=None) -> CoastingOrbit | None:
    """General transfer. Minimises Jt + Ja under RAAN closure (Eqs. 52-60).

    The RAAN-closure constraint (Eq. 59) slaves the semi-major-axis change to
    the inclination change, leaving one decision variable. The objective is
    convex in it, so a bracketed one-dimensional minimisation finds the global
    optimum. The coasting-altitude floor (Eq. 55) enters as a one-sided bound on
    the bracket: an interior optimum is the unconstrained solution, a boundary
    optimum is the active-floor solution, so both KKT cases are covered by a
    single solve.

    Returns None when no admissible coasting orbit exists.
    """
    di_total = inclf - incl0
    omega_dot_0 = raan_drift_rate(a0, incl0)
    omega_dot_f = raan_drift_rate(af, inclf)

    norm_choice = eq44_normalisation or EQ44_NORMALISATION
    a_ref = a0 if norm_choice == "a0" else 0.5 * (a0 + af)

    # Near-polar orbits have almost no J2 nodal drift; the method degenerates.
    if abs(omega_dot_0) < 1e-14:
        return None

    # Eq. 59:  -3.5*x1 - tan(I0)*x2 = R,  with x1 = da/a0 and x2 = di.
    # da is normalised by a0, not a_bar: Eq. 44 expands the drift rate about the
    # *initial* orbit. The arc increments above do use a_bar; different quantity.
    r_const = wrap_pi(d_raan + (omega_dot_f - omega_dot_0) * tf_sec) / (omega_dot_0 * tf_sec)

    def da_of(x2):
        return -a_ref * (r_const + math.tan(incl0) * x2) / 3.5

    def j_of(x2):
        jt, ja = _arc_increments(a0, incl0, af, inclf, da_of(x2), x2)
        return jt + ja

    # Generous physical bracket: total inclination change plus or minus 90 deg.
    pad = math.radians(90.0)
    lo = min(0.0, di_total) - pad
    hi = max(0.0, di_total) + pad

    # Altitude floor (Eq. 55): da >= a_min - a0, a one-sided bound on x2.
    tan_i = math.tan(incl0)
    if abs(tan_i) > 1e-12:
        x2_floor = (-3.5 * (a_min - a0) / a_ref - r_const) / tan_i
        if tan_i > 0:      # da decreases as x2 grows
            hi = min(hi, x2_floor)
        else:              # da increases as x2 grows
            lo = max(lo, x2_floor)
    if lo >= hi:
        return None

    interior = minimize_scalar(j_of, bounds=(lo, hi), method="bounded",
                               options={"xatol": 1e-12})
    # Test the endpoints too, so an active-floor optimum is exact.
    x2 = min([interior.x, lo, hi], key=j_of)

    da_transfer = da_of(x2)
    jt, ja = _arc_increments(a0, incl0, af, inclf, da_transfer, x2)
    return CoastingOrbit(a_c=a0 + da_transfer, incl_c=incl0 + x2, jt=jt, ja=ja)


def coasting_orbit_estimate(a0, incl0, raan0, af, inclf, raanf, tof_days,
                            a_min=A_MIN, eq44_normalisation=None) -> CoastingOrbit | None:
    """Section 4.2 estimate of the coasting orbit and the impulsive delta-V.

    Returns None when the transfer would require drifting below `a_min`, which
    is outside the model's domain — reported as unreachable rather than as a
    cheap transfer.
    """
    tf_sec = tof_days * DAY
    d_raan = wrap_pi(raanf - raan0)

    coplanar = abs(af - a0) < 1e-6 and abs(inclf - incl0) < 1e-6
    if coplanar:
        orbit = _case1_raan_only(a0, incl0, d_raan, tf_sec)
    else:
        orbit = _case2_general(a0, incl0, af, inclf, d_raan, tf_sec, a_min=a_min,
                               eq44_normalisation=eq44_normalisation)

    if orbit is None or orbit.a_c < a_min:
        return None
    return orbit


# ---------------------------------------------------------------------------
# Section 3 — averaged dynamics under yaw-switch steering
# ---------------------------------------------------------------------------


def propagate_phase(a0, incl0, raan0, beta1, beta2, theta, eps1, eps2, duration, fmax):
    """Propagate (a, i, RAAN) over one thrusting phase.

    `theta` is the yaw switch angle, `beta1`/`beta2` the yaw angles either side
    of the switch, `eps1`/`eps2` the throttle in each segment. Closed-form
    solutions of the averaged equations; the four branches below are the cases
    where the semi-major axis and/or the inclination are stationary, which make
    the general expressions singular.
    """
    f1, f2 = eps1 * fmax, eps2 * fmax
    fc1, fs1 = f1 * math.cos(beta1), f1 * math.sin(beta1)
    fc2, fs2 = f2 * math.cos(beta2), f2 * math.sin(beta2)

    v0 = math.sqrt(MU / a0)
    kappa = (4.0 * theta * fc1 + 2.0 * (math.pi - 2.0 * theta) * fc2) / math.pi

    tiny = 1e-20
    a_varies = abs(kappa) > tiny
    i_varies = abs(fs1) > tiny

    a = 4.0 * MU / (2.0 * v0 - kappa * duration) ** 2 if a_varies else a0

    if not i_varies:
        incl = incl0
    elif not a_varies:
        xi = 2.0 * fs1 * math.sin(theta) / (math.pi * v0)
        incl = incl0 + xi * duration
    else:
        z = math.log(2.0 * v0 - kappa * duration)
        z0 = math.log(2.0 * v0)
        xi_bar = -4.0 * fs1 * math.sin(theta) / (math.pi * kappa)
        incl = incl0 + xi_bar * (z - z0)

    if not a_varies and not i_varies:
        raan = raan0 + (
            2.0 * math.cos(theta) * fs2 / (math.pi * v0 * math.sin(incl0))
            - K_J2 * v0**7 * math.cos(incl0)
        ) * duration
    elif not a_varies and i_varies:
        xi = 2.0 * fs1 * math.sin(theta) / (math.pi * v0)
        incl_t = incl0 + xi * duration
        ratio = math.log(
            ((1 + math.cos(incl_t)) * (1 - math.cos(incl0)))
            / ((1 - math.cos(incl_t)) * (1 + math.cos(incl0)))
        )
        raan = (raan0
                - K_J2 * (v0**7 / xi) * (math.sin(incl_t) - math.sin(incl0))
                - math.cos(theta) * fs2 / (math.pi * xi * v0) * ratio)
    elif a_varies and not i_varies:
        z = math.log(2.0 * v0 - kappa * duration)
        z0 = math.log(2.0 * v0)
        raan = (raan0
                - 4.0 * math.cos(theta) * fs2 / (math.pi * kappa * math.sin(incl0)) * (z - z0)
                + K_J2 * math.cos(incl0) / (1024.0 * kappa) * (math.exp(8 * z) - math.exp(8 * z0)))
    else:
        z = math.log(2.0 * v0 - kappa * duration)
        z0 = math.log(2.0 * v0)
        xi_bar = -4.0 * fs1 * math.sin(theta) / (math.pi * kappa)
        incl_t = incl0 + xi_bar * (z - z0)
        ratio = math.log(
            ((1 + math.cos(incl_t)) * (1 - math.cos(incl0)))
            / ((1 - math.cos(incl_t)) * (1 + math.cos(incl0)))
        )

        def g(zz, ii):
            return math.exp(8 * zz) * (8 * math.cos(ii) + xi_bar * math.sin(ii)) / (64 + xi_bar**2)

        raan = (raan0
                + 2.0 * math.cos(theta) * fs2 / (math.pi * xi_bar * kappa) * ratio
                + K_J2 / (128.0 * kappa) * (g(z, incl_t) - g(z0, incl0)))

    return a, incl, raan


# ---------------------------------------------------------------------------
# Section 4.3 — continuous-thrust NLP
# ---------------------------------------------------------------------------

# Decision vector: [beta_t1, theta_t, T_transfer_days, beta_a1, theta_a, T_adjust_end_days]
# This is the paper's xi_1 = [1, 0, 1, 0] integer choice — segment 1 only —
# which is optimal for all three published cases.


def _trajectory(x, a0, incl0, raan0, tf_sec, fmax):
    """Propagate transfer arc, ballistic J2 coast, adjustment arc."""
    beta_t1, theta_t, t_transfer_d, beta_a1, theta_a, t_adjust_end_d = x
    t_transfer = t_transfer_d * DAY
    t_adjust_end = t_adjust_end_d * DAY

    a1, i1, raan1 = propagate_phase(
        a0, incl0, raan0, beta_t1, 0.0, theta_t, 1.0, 0.0, t_transfer, fmax)
    raan2 = raan1 + raan_drift_rate(a1, i1) * (t_adjust_end - t_transfer)
    a3, i3, raan3 = propagate_phase(
        a1, i1, raan2, beta_a1, 0.0, theta_a, 1.0, 0.0, tf_sec - t_adjust_end, fmax)
    return a1, i1, a3, i3, raan3


def _objective(x, tf_sec, fmax):
    """Continuous-thrust delta-V, Eq. 36 for xi_1 [m/s]."""
    return (2.0 * fmax / math.pi) * (x[1] * x[2] * DAY + x[4] * (tf_sec - x[5] * DAY)) * 1000.0


def _defects(x, a0, incl0, af, inclf, a_c, incl_c, raan0, raan_target, tf_sec, fmax):
    """Equality constraints: hit the coasting orbit, then the target orbit.

    The two semi-major-axis residuals are divided by 100 so that all five
    constraints are of comparable magnitude, which SLSQP needs.
    """
    a1, i1, a3, i3, raan3 = _trajectory(x, a0, incl0, raan0, tf_sec, fmax)
    return np.array([(a1 - a_c) / 100.0, i1 - incl_c,
                     (a3 - af) / 100.0, i3 - inclf,
                     raan3 - raan_target])


def nlp_transfer_cost(a0, incl0, af, inclf, a_c, incl_c, raan0, raan_target,
                      tof_days, seeds, fmax=FMAX_DEFAULT,
                      bounds=None, feasibility_tol=1e-4):
    """Section 4.3 continuous-thrust delta-V [m/s], or None if no seed converges.

    `a_c`, `incl_c` are the coasting orbit from Section 4.2. `seeds` is a list of
    starting decision vectors; the paper's published solutions are close enough
    to the optimum that one good seed suffices, but SLSQP on this problem is
    seed-sensitive, so several are tried and the best feasible one is returned.
    """
    tf_sec = tof_days * DAY
    if bounds is None:
        bounds = [(math.radians(-179), math.radians(179)),
                  (math.radians(0.3), math.radians(89)),
                  (0.05, 60.0),
                  (math.radians(-179), math.radians(179)),
                  (math.radians(0.3), math.radians(89)),
                  (40.0, 99.95)]

    args = (a0, incl0, af, inclf, a_c, incl_c, raan0, raan_target, tf_sec, fmax)
    constraints = {"type": "eq", "fun": lambda x: _defects(x, *args)}

    best = None
    for seed in seeds:
        res = minimize(_objective, np.asarray(seed, dtype=float),
                       args=(tf_sec, fmax), method="SLSQP",
                       bounds=bounds, constraints=[constraints],
                       options={"maxiter": 3000, "ftol": 1e-12})
        residual = np.max(np.abs(_defects(res.x, *args)))
        if residual < feasibility_tol and (best is None or res.fun < best):
            best = float(res.fun)
    return best
