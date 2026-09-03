"""Gate on the RK4 + J2 propagation port.

Checks the port against closed-form results rather than against the Julia,
so it stands on its own if the Julia is retired.
"""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos.constants import J2, MU, R_E
from oos.propagate import (eci_from_oelem, oelem_from_rv, propagate,
                           mean_to_true)

fails = []
def check(name, got, want, tol, unit=""):
    err = abs(got - want)
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {got:.9g} vs {want:.9g} "
          f"(err {err:.3g} {unit})")
    if not ok:
        fails.append(name)

print("1. element round-trip, eccentric and inclined")
a, e, i, W, w, nu = 7000.0, 0.01, math.radians(53.0), math.radians(120.0), math.radians(40.0), math.radians(210.0)
r, v = eci_from_oelem(a, e, i, W, w, nu)
a2, i2, W2, nu2, u2 = oelem_from_rv(r[0], r[1], r[2], v[0], v[1], v[2])
check("semi-major axis", a2, a, 1e-9, "km")
check("inclination", i2, i, 1e-12, "rad")
check("RAAN", W2, W, 1e-12, "rad")
check("true anomaly", nu2, nu, 1e-12, "rad")
check("argument of latitude", u2, (w + nu) % (2*math.pi), 1e-12, "rad")

print("\n2. Kepler solver: M -> nu -> M")
for ecc in (0.0, 1e-4, 0.01, 0.2):
    for M in (0.3, 2.0, 5.5):
        nu_ = mean_to_true(M, ecc)
        E = 2.0*math.atan2(math.sqrt(1-ecc)*math.sin(0.5*nu_), math.sqrt(1+ecc)*math.cos(0.5*nu_))
        M_back = (E - ecc*math.sin(E)) % (2*math.pi)
        check(f"e={ecc} M={M}", M_back, M % (2*math.pi), 1e-11, "rad")

print("\n3. two-body only: i and RAAN invariant; a drifts only by RK4 truncation")
a0 = R_E + 550.0
r, v = eci_from_oelem(a0, 0.0, math.radians(53.0), 0.0, 0.0, 0.0)
r0 = np.array([r]); v0 = np.array([v])
n_steps = int(5*86400/60)
pos, vel, oe, times, _sec = propagate(r0, v0, 60.0, n_steps, 60, False)
check("max |di|", float(np.abs(oe[0,:,1]-math.radians(53.0)).max()), 0.0, 1e-11, "rad")
raan = np.unwrap(oe[0,:,2])
check("max |dRAAN|", float(np.abs(raan-raan[0]).max()), 0.0, 1e-10, "rad")
# RK4 is 4th order, so halving dt should cut the semi-major-axis drift by ~16.
d60 = float(np.abs(oe[0,:,0]-a0).max())
_,_,oe30,_,_ = propagate(r0, v0, 30.0, 2*n_steps, 120, False)
d30 = float(np.abs(oe30[0,:,0]-a0).max())
print(f"    |da| at dt=60 s: {d60:.4e} km;  at dt=30 s: {d30:.4e} km;  ratio {d60/d30:.1f}")
# The point is that the drift converges away with dt, so it is truncation and
# not a defect. RK4 is 4th order globally; on a closed orbit the leading terms
# largely cancel over a period, so the observed order for the secular energy
# drift comes out near 5. Gate on convergence, not on a particular order.
order = math.log2(d60/d30)
print(f"    observed convergence order of the drift: {order:.2f}")
check("drift converges at 4th order or better", min(order, 4.0), 4.0, 1e-9, "orders")
check("|da| at dt=60 s over 5 days", d60, 0.0, 0.5, "km")

print("\n4. J2: secular nodal rate matches the closed form, 20 days")
for inc_deg, alt in ((53.0, 550.0), (97.6, 560.0), (70.0, 570.0)):
    a0 = R_E + alt
    inc = math.radians(inc_deg)
    r, v = eci_from_oelem(a0, 0.0, inc, 0.0, 0.0, 0.0)
    n_steps = int(20*86400/60)
    pos, vel, oe, times, _sec = propagate(np.array([r]), np.array([v]), 60.0, n_steps, 60, True)
    raan = np.unwrap(oe[0,:,2])
    rate = np.polyfit(times, raan, 1)[0]          # rad/s
    n = math.sqrt(MU/a0**3)
    analytic = -1.5*J2*(R_E/a0)**2*n*math.cos(inc)
    got = math.degrees(rate)*86400
    want = math.degrees(analytic)*86400
    check(f"nodal rate i={inc_deg} deg, rel. to mean-element form",
          got/want, 1.0, 0.01, "(1 = exact; offset of order J2 expected)")

print("\n5. J2: mean semi-major axis and inclination stay bounded, 20 days")
check("max |da| about its mean", float(np.abs(oe[0,:,0]-oe[0,:,0].mean()).max()), 0.0, 20.0, "km")
check("max |di| about its mean [deg]",
      float(np.degrees(np.abs(oe[0,:,1]-oe[0,:,1].mean()).max())), 0.0, 0.05, "deg")

print("\n6. orbital period from the argument of latitude, two-body")
a0 = R_E + 550.0
r, v = eci_from_oelem(a0, 0.0, math.radians(53.0), 0.0, 0.0, 0.0)
pos, vel, oe, times, _sec = propagate(np.array([r]), np.array([v]), 10.0, int(3*86400/10), 6, False)
u = np.unwrap(oe[0,:,4])
period = 2*math.pi/(np.polyfit(times, u, 1)[0])
check("period [s]", period, 2*math.pi*math.sqrt(a0**3/MU), 1e-4, "s")


print("\n7. argument of latitude is a usable phase where the true anomaly is not")
# Two co-planar near-circular orbits differing only in where they sit along the
# track. Their phase difference must equal the along-track separation; measured
# through the true anomaly it does not, because each has its own perigee.
a0 = R_E + 550.0
inc = math.radians(53.0)
sep = math.radians(37.0)
r1, v1 = eci_from_oelem(a0, 5e-4, inc, 0.0, math.radians(215.0), 0.0)
r2, v2 = eci_from_oelem(a0, 5e-4, inc, 0.0, math.radians(154.0), sep + math.radians(215.0-154.0))
_,_,_, nu1, u1 = oelem_from_rv(*r1, *v1)
_,_,_, nu2_, u2 = oelem_from_rv(*r2, *v2)
d_u  = (u2 - u1) % (2*math.pi)
d_nu = (nu2_ - nu1) % (2*math.pi)
check("phase gap via argument of latitude [deg]", math.degrees(d_u), 37.0, 1e-8, "deg")
print(f"    same gap via the true anomaly: {math.degrees(d_nu):.2f} deg "
      f"-- off by {math.degrees(d_nu)-37.0:+.2f} deg")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
