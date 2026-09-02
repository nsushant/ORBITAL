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


def test_returned_transfer_actually_closes_raan():
    """Every delta-V the search returns must correspond to a closing transfer."""
    mass, isp, thrust = 335.0, 2800.0, 1e-5
    checked = 0
    rng = np.random.default_rng(7)
    for _ in range(40):
        a0 = 6378.137 + rng.uniform(400, 600)
        af = 6378.137 + rng.uniform(400, 600)
        i0 = math.radians(rng.uniform(50, 56))
        i_f = math.radians(rng.uniform(50, 56))
        raan_gap = rng.uniform(-1.0, 1.0)
        s1 = np.array([a0, i0, 0.0, mass, isp, thrust])
        s2 = np.array([af, i_f, raan_gap, mass, isp, thrust])
        tofs = np.array([30.0, 60.0, 120.0]) * 86400.0
        out = eb.transfer_cost(s1, s2, tofs)
        for k, dv in enumerate(out):
            if np.isnan(dv):
                continue
            # Re-derive closure independently of the search bookkeeping.
            found = False
            for growth in [0.0] + list(10.0 ** np.linspace(-1, np.log10(8.0), 20)):
                pts = eb.ellipse_points(growth, *eb.to_velocity_plane(a0, i0),
                                        *eb.to_velocity_plane(af, i_f), 30)
                for p in pts:
                    v = eb.evaluate_drift_orbit(p[0], p[1], s1, s2, tofs[k], raan_gap)
                    if not np.isnan(v) and abs(v - dv) < 1e-12:
                        found = True
                        break
                if found:
                    break
            assert found, "returned delta-V not reproducible from an admissible drift orbit"
            checked += 1
    assert checked > 0, "no feasible transfers generated; test is vacuous"
    return checked


if __name__ == "__main__":
    print(f"closed-form Edelbaum agreement       worst rel. err = {test_arc_matches_closed_form_edelbaum():.2e}")
    print(f"coplanar arc = |V2 - V1|             abs. err       = {test_coplanar_arc_is_velocity_difference():.2e}")
    print(f"pure plane change = 2V sin(pi dI/4)  abs. err       = {test_pure_plane_change_arc():.2e}")
    print(f"burn duration vs rocket equation     duration       = {test_burn_duration_matches_rocket_equation():.1f} s")
    print(f"sun-synchronous drift rate           = {test_sun_synchronous_drift_rate():.4f} deg/day (want 0.9856)")
    print(f"RAAN quadrature vs fine trapezoid    rel. err       = {test_raan_quadrature_against_fine_integration():.2e}")
    print(f"returned transfers close RAAN        checked        = {test_returned_transfer_actually_closes_raan()} solutions")
    print("\nall checks passed")
