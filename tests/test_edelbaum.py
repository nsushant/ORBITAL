"""Physics checks on the Edelbaum transfer model.

The model is not validated against a published table, so it is pinned instead
against closed-form results it must reproduce exactly, and against an
independent numerical evaluation of the one quantity computed by quadrature.

Run:  python3 -m pytest tests/test_edelbaum.py -q     (or: python3 tests/test_edelbaum.py)
"""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oos.constants import G0, J2, MU, R_E
from oos import edelbaum as eb


def test_arc_matches_closed_form_edelbaum():
    """Velocity-plane chord length must equal Edelbaum's relation."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        a0 = 6378.137 + rng.uniform(300, 1500)
        af = 6378.137 + rng.uniform(300, 1500)
        i0 = math.radians(rng.uniform(0, 100))
        i_f = math.radians(rng.uniform(0, 100))
        dv, _, _ = eb.arc_cost(a0, i0, af, i_f, 335.0, 2800.0, 1e-5)
        v1, v2 = math.sqrt(MU / a0), math.sqrt(MU / af)
        closed = math.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * math.cos(0.5 * math.pi * (i_f - i0)))
        worst = max(worst, abs(dv - closed) / closed)
    assert worst < 1e-12, f"worst relative error {worst:.2e}"
    return worst


def test_coplanar_arc_is_velocity_difference():
    """With no plane change the arc costs exactly |V2 - V1|."""
    a0, af = 6378.137 + 500.0, 6378.137 + 900.0
    incl = math.radians(53.0)
    dv, _, _ = eb.arc_cost(a0, incl, af, incl, 335.0, 2800.0, 1e-5)
    expected = abs(math.sqrt(MU / af) - math.sqrt(MU / a0))
    assert abs(dv - expected) < 1e-12
    return abs(dv - expected)


def test_pure_plane_change_arc():
    """At fixed semi-major axis the arc costs 2 V sin(pi/4 * dI)."""
    a = 6378.137 + 550.0
    i0, i_f = math.radians(53.0), math.radians(70.0)
    dv, _, _ = eb.arc_cost(a, i0, a, i_f, 335.0, 2800.0, 1e-5)
    v = math.sqrt(MU / a)
    expected = 2 * v * math.sin(0.25 * math.pi * (i_f - i0))
    assert abs(dv - expected) < 1e-12
    return abs(dv - expected)


def test_burn_duration_matches_rocket_equation():
    """Burn time must satisfy m(t) = m0 exp(-dV/ve) at constant thrust."""
    mass, isp, thrust = 335.0, 2800.0, 1e-5
    a0, af = 6378.137 + 500.0, 6378.137 + 800.0
    incl = math.radians(53.0)
    dv, duration, m_end = eb.arc_cost(a0, incl, af, incl, mass, isp, thrust)
    ve = isp * G0
    assert abs(m_end - mass * math.exp(-dv / ve)) < 1e-12
    # constant thrust: dm/dt = -T/ve, so duration = (m0 - m_end) * ve / T
    assert abs(duration - (mass - m_end) * ve / thrust) < 1e-6
    return duration


def test_sun_synchronous_drift_rate():
    """A sun-synchronous orbit must regress at about one degree per day."""
    rate = eb.j2_raan_rate(6378.137 + 800.0, math.radians(98.6))
    deg_per_day = math.degrees(rate) * 86400.0
    assert abs(deg_per_day - 0.9856) < 0.002, deg_per_day
    return deg_per_day


def test_raan_quadrature_against_fine_integration():
    """The 16-point Gauss-Legendre arc integral must match a fine trapezoid."""
    mass, isp, thrust = 335.0, 2800.0, 1e-5
    a0, af = 6378.137 + 500.0, 6378.137 + 1100.0
    i0, i_f = math.radians(53.0), math.radians(60.0)
    x0, y0 = eb.to_velocity_plane(a0, i0)
    xf, yf = eb.to_velocity_plane(af, i_f)
    dv, _, _ = eb.arc_cost(a0, i0, af, i_f, mass, isp, thrust)
    quad = eb._raan_during_arc(a0, i0, mass, isp, thrust, x0, y0, xf, yf, dv)

    ve = isp * G0
    s = np.linspace(0.0, dv, 200_001)
    frac = s / dv
    x = x0 + frac * (xf - x0)
    y = y0 + frac * (yf - y0)
    a_s = MU / (x * x + y * y)
    i_s = 2.0 / np.pi * np.arctan2(y, x)
    n = np.sqrt(MU / a_s**3)
    rate = -1.5 * J2 * (R_E / a_s) ** 2 * n * np.cos(i_s)
    accel = thrust / (mass * np.exp(-s / ve))
    fine = np.trapezoid(rate / accel, s)
    rel = abs(quad - fine) / abs(fine)
    assert rel < 1e-9, f"relative error {rel:.2e}"
    return rel


def test_ring_delta_v_identity():
    """Every point on a ring costs the same: dv1 + dv2 = 2 * (c + growth).

    The ring is an ellipse with the two endpoints as foci, and the arcs are the
    distances from the drift orbit to each focus. This identity is what lets the
    search minimise over rings instead of over points.
    """
    mass, isp, thrust = 335.0, 2800.0, 1e-5
    a0, af = 6378.137 + 550.0, 6378.137 + 700.0
    i0, i_f = math.radians(53.0), math.radians(56.0)
    x1, y1 = eb.to_velocity_plane(a0, i0)
    x2, y2 = eb.to_velocity_plane(af, i_f)
    c = 0.5 * math.hypot(x2 - x1, y2 - y1)

    worst = 0.0
    for growth in (0.0, 0.05, 0.2, 1.0, 5.0, 50.0):
        t_max = 1.0 if growth == 0.0 else 2 * math.pi
        for t in np.linspace(0.0, t_max, 37):
            px, py = eb.ring_point(growth, x1, y1, x2, y2, t)
            a_d, i_d = eb.from_velocity_plane(px, py)
            dv1, _, m_d = eb.arc_cost(a0, i0, a_d, i_d, mass, isp, thrust)
            dv2, _, _ = eb.arc_cost(a_d, i_d, af, i_f, m_d, isp, thrust)
            worst = max(worst, abs((dv1 + dv2) - 2 * (c + growth)))
    assert worst < 1e-12, f"worst deviation {worst:.2e} km/s"
    return worst


def test_returned_cost_corresponds_to_a_closing_drift_orbit():
    """Each delta-V returned must sit on a ring that genuinely closes RAAN."""
    mass, isp, thrust = 335.0, 2800.0, 1e-5
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(30):
        a0 = 6378.137 + rng.uniform(500, 600)
        af = a0 + rng.uniform(-60, 60)
        i0 = math.radians(53.0 + rng.uniform(-0.5, 0.5))
        i_f = i0 + math.radians(rng.uniform(-2, 2))
        gap = rng.uniform(-math.pi, math.pi)
        s1 = np.array([a0, i0, 0.0, mass, isp, thrust])
        s2 = np.array([af, i_f, gap, mass, isp, thrust])
        tofs = np.array([180.0, 365.0]) * 86400.0
        out = eb.transfer_cost(s1, s2, tofs)

        x1, y1 = eb.to_velocity_plane(a0, i0)
        x2, y2 = eb.to_velocity_plane(af, i_f)
        c = 0.5 * math.hypot(x2 - x1, y2 - y1)
        for k, dv in enumerate(out):
            if np.isnan(dv):
                continue
            growth = 0.5 * dv - c
            assert growth > -1e-9, "returned cost below the direct transfer"
            assert eb.ring_closes(max(growth, 0.0), x1, y1, x2, y2, s1, s2,
                                  tofs[k], gap, 180), "no closing drift orbit on that ring"
            checked += 1
    assert checked > 0, "no feasible transfers generated; test is vacuous"
    return checked


def test_search_finds_the_transfers_that_exist():
    """Regression guard on docs/raan_closure_defect.md.

    The old sampling search found 7.8 % of the transfers that provably exist on
    this population. Root-finding must find essentially all of them.
    """
    mass, isp, thrust = 335.0, 2800.0, 1e-5
    rng = np.random.default_rng(11)
    exists = 0
    found = 0
    for _ in range(60):
        a0 = 6378.137 + rng.uniform(500, 600)
        af = a0 + rng.uniform(-60, 60)
        i0 = math.radians(53.0 + rng.uniform(-0.5, 0.5))
        i_f = i0 + math.radians(rng.uniform(-2, 2))
        gap = rng.uniform(-math.pi, math.pi)
        tof = rng.uniform(120, 365) * 86400.0
        s1 = np.array([a0, i0, 0.0, mass, isp, thrust])
        s2 = np.array([af, i_f, gap, mass, isp, thrust])

        x1, y1 = eb.to_velocity_plane(a0, i0)
        x2, y2 = eb.to_velocity_plane(af, i_f)
        ladder = [0.0] + list(10.0 ** np.linspace(-2, np.log10(4096.0), 48))
        any_ring = any(eb.ring_closes(g, x1, y1, x2, y2, s1, s2, tof, gap, 360)
                       for g in ladder)
        exists += any_ring
        found += (not np.isnan(eb.transfer_cost(s1, s2, np.array([tof]))[0]))
    assert exists > 0, "test population has no feasible transfers; not informative"
    rate = found / exists
    assert rate > 0.95, f"only {100*rate:.1f} % of existing transfers found"
    return exists, found, rate


if __name__ == "__main__":
    print(f"closed-form Edelbaum agreement       worst rel. err = {test_arc_matches_closed_form_edelbaum():.2e}")
    print(f"coplanar arc = |V2 - V1|             abs. err       = {test_coplanar_arc_is_velocity_difference():.2e}")
    print(f"pure plane change = 2V sin(pi dI/4)  abs. err       = {test_pure_plane_change_arc():.2e}")
    print(f"burn duration vs rocket equation     duration       = {test_burn_duration_matches_rocket_equation():.1f} s")
    print(f"sun-synchronous drift rate           = {test_sun_synchronous_drift_rate():.4f} deg/day (want 0.9856)")
    print(f"RAAN quadrature vs fine trapezoid    rel. err       = {test_raan_quadrature_against_fine_integration():.2e}")
    print(f"ring delta-V identity                worst dev      = {test_ring_delta_v_identity():.2e} km/s")
    print(f"returned costs close RAAN            checked        = {test_returned_cost_corresponds_to_a_closing_drift_orbit()} solutions")
    e, f, r = test_search_finds_the_transfers_that_exist()
    print(f"transfers found vs transfers existing = {f}/{e}  ({100*r:.1f} %)")
    print("\nall checks passed")
