"""Nothing in the package may raise on a degenerate input.

The ring search once reached a drift orbit of 24 m, where the rocket equation
underflowed the servicer's mass to zero and dividing thrust by it raised
ZeroDivisionError -- and inside a numba parallel build the same condition came
back as a silently all-NaN table instead. So these checks assert two things,
and the second matters more than the first:

  nothing raises, and
  a degenerate input produces NaN or a plainly absurd number, never a
  plausible-looking one.
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos import edelbaum as eb
from oos import servicer
from oos.constants import R_E
from oos.phasing import phasing
from oos.propagate import _accel, eci_from_oelem, mean_to_true, oelem_from_rv, propagate

fails = []
def check(name, fn):
    try:
        out = fn()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  FAIL  {name:<52} raised {type(exc).__name__}: {exc}")
        fails.append(name)
        return None
    print(f"  PASS  {name:<52} -> {out}")
    return out

M, I, T = servicer.MASS, servicer.ISP, servicer.THRUST

print("1. velocity-plane map at and near the origin")
check("from_velocity_plane(0, 0)", lambda: eb.from_velocity_plane(0.0, 0.0))
check("from_velocity_plane(1e-300, 0)", lambda: eb.from_velocity_plane(1e-300, 0.0))
check("from_velocity_plane(1e8, 1e8)", lambda: eb.from_velocity_plane(1e8, 1e8))

print("\n2. nodal rate at degenerate semi-major axes")
for a in (0.0, 1e-300, -1.0, 1e30):
    check(f"j2_raan_rate(a={a:g})", lambda a=a: eb.j2_raan_rate(a, 0.9))

print("\n3. rocket equation with a vanishing or absurd vehicle")
check("arc_cost, zero thrust", lambda: eb.arc_cost(6900.0, 0.9, 7000.0, 0.95, M, I, 0.0))
check("arc_cost, zero Isp", lambda: eb.arc_cost(6900.0, 0.9, 7000.0, 0.95, M, 0.0, T))
check("arc_cost, zero mass", lambda: eb.arc_cost(6900.0, 0.9, 7000.0, 0.95, 0.0, I, T))
check("arc_cost onto a 24 m orbit (the old failure)",
      lambda: eb.arc_cost(6900.0, 0.9, 0.0238, 0.5, M, I, T))

print("\n4. RAAN quadrature along a pathological arc -- the division that raised")
x0, y0 = eb.to_velocity_plane(6900.0, 0.9)
check("_raan_during_arc across 4000 km/s",
      lambda: eb._raan_during_arc(6900.0, 0.9, M, I, T, x0, y0, 4000.0, 100.0, 4088.0))
check("_raan_during_arc with zero mass",
      lambda: eb._raan_during_arc(6900.0, 0.9, 0.0, I, T, x0, y0, x0 + 1, y0, 1.0))

print("\n5. the whole ring search on the geometry that used to fail")
s1 = np.array([6907.14, math.radians(53.141), 0.0, M, I, T])
s2 = np.array([6910.64, math.radians(53.141), 0.0, M, I, T])
for g in (4096.0, 7.5947, 1e6):
    check(f"transfer_cost with max_growth={g:g}",
          lambda g=g: np.nansum(eb.transfer_cost(
              s1, s2, np.array([60.0 * 86400.0]), np.array([1.0]), 180, 48, g, 1e-3)))
check("transfer_cost, identical orbits",
      lambda: eb.transfer_cost(s1, s1.copy(), np.array([60.0 * 86400.0]),
                               np.array([0.0]))[0])
check("transfer_cost, zero time of flight",
      lambda: eb.transfer_cost(s1, s2, np.array([0.0]), np.array([1.0]))[0])

print("\n6. phasing at degenerate orbits and gaps")
check("phasing(a=0)", lambda: phasing(0.0, 0.0, 1.0))
check("phasing(a=1e-300)", lambda: phasing(1e-300, 0.0, 1.0))
check("phasing, zero gap", lambda: phasing(6900.0, 1.0, 1.0))

print("\n7. propagation kernels at the singularity")
check("_accel at the origin", lambda: _accel(0.0, 0.0, 0.0, True))
check("oelem_from_rv, all zeros", lambda: oelem_from_rv(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
check("oelem_from_rv, radial (h = 0)",
      lambda: oelem_from_rv(7000.0, 0.0, 0.0, 1.0, 0.0, 0.0))
check("oelem_from_rv, equatorial circular",
      lambda: oelem_from_rv(7000.0, 0.0, 0.0, 0.0, 7.546, 0.0))
check("eci_from_oelem at ecc = 1, nu = pi",
      lambda: eci_from_oelem(7000.0, 1.0, 0.9, 0.0, 0.0, math.pi))
check("mean_to_true at ecc = 1", lambda: mean_to_true(0.0, 1.0))
check("propagate a node starting at the origin",
      lambda: float(np.nan_to_num(propagate(np.zeros((1, 3)), np.zeros((1, 3)),
                                            10.0, 20, 10, True)[2]).sum()))

print("\n8. a degenerate input must not look plausible")
dv, dur = phasing(0.0, 0.0, 1.0)
ok = not np.isfinite(dv) or dv > 1e3
print(f"  {'PASS' if ok else 'FAIL'}  phasing at a=0 gives {dv!r}, not a usable delta-V")
if not ok:
    fails.append("phasing at a=0 looks plausible")
a_deg, _ = eb.from_velocity_plane(0.0, 0.0)
ok = not np.isfinite(a_deg)
print(f"  {'PASS' if ok else 'FAIL'}  origin maps to {a_deg!r}, not a usable orbit")
if not ok:
    fails.append("origin looks plausible")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed -- no degenerate input raises")
