"""Node ephemerides: RK4 on the Cartesian state under two-body plus J2.

A port of `sim/orbital_mechanics.jl` and `sim/propagator.jl`. Section 3.4 of the
paper describes this integrator explicitly (Eq. eq:a), so the port reproduces it
rather than substituting the analytic mean-element rates that the transfer model
itself uses. See decision D15 in the plan.

Two deliberate differences from the Julia, both documented rather than silent:

constants   The port uses `oos/constants.py` (R_E = 6378.137 km,
            J2 = 1.0825267e-3) where the Julia sim used R_E = 6371.0 and
            J2 = 1.08263e-3. One constant set now holds across the whole stack;
            the effect on transfer cost was measured in
            docs/edelbaum_port_audit.md at 0.79 % mean, 7.80 % worst.

record grid The Julia recorded the state *before* taking the step whose index
            was a multiple of WRITE_EVERY, so its first record sat at
            t = 3540 s rather than 0 and every record was one step early. The
            port records at t = 0, 3600, 7200, ... exactly. Over 400 days this
            shifts each sample by 60 s.

Output layout matches what h5py sees when reading the Julia file, so existing
readers are unaffected: `metadata/names`, `metadata/times`, and per node
`positions` (n_rec, 3), `velocities` (n_rec, 3), `orbital_elements`
(n_rec, **5**) holding (a, inclination, RAAN, true anomaly, argument of
latitude) in km and radians. The first four columns are the Julia's, so a
reader that slices `[:4]` is unaffected.

The fifth column is new, and it exists because the fourth is not usable as a
phase angle here. `cost/cost_functions.jl` feeds the true anomaly to
`phasing()` as the in-plane phase of both nodes. On these near-circular orbits
(e ~ 5e-4, dominated by the J2 forced eccentricity) the perigee that the true
anomaly is measured from is essentially arbitrary: it differs by tens of
degrees between two nodes in the same shell and precesses at about 30 deg/day.
The common phase variable is the argument of latitude u = argp + nu, measured
from the ascending node, which is well defined however small the eccentricity
gets. Phasing must use column 4, not column 3. See F11 in the plan.

Usage
-----
    python3 -m oos.propagate --population outputs/instance_population.csv \
                             --out outputs/simulation.h5 --days 400
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
from numba import njit, prange

from .constants import J2, MU, R_E
from .guards import R_FLOOR, den

# The Julia used dt = 60 s. Over the 400-day horizon that integrates to a
# spurious secular loss of up to 11.4 km in the mean semi-major axis and 4.4 deg
# in RAAN, measured against a converged dt = 10 s run -- large enough to matter,
# since RAAN closure is what binds the transfer geometry. At dt = 10 s the run
# takes 9 s and the residual is under 0.05 deg. See F12 in the plan.
DT = 10.0                 # s
WRITE_EVERY = 360         # integrator steps between records -> hourly
T_END_DAYS = 400.0


@njit(cache=True)
def eci_from_oelem(sma, ecc, inc, raan, aop, nu):
    """Cartesian state from classical elements. Angles in radians, km and km/s."""
    p = sma * (1.0 - ecc * ecc)
    rmag = p / den(1.0 + ecc * math.cos(nu))
    h = den(math.sqrt(MU * abs(p)))

    rp0 = rmag * math.cos(nu)
    rp1 = rmag * math.sin(nu)
    vp0 = -MU / h * math.sin(nu)
    vp1 = MU / h * (ecc + math.cos(nu))

    co, so = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(aop), math.sin(aop)

    r00 = co * cw - so * sw * ci
    r01 = -co * sw - so * cw * ci
    r10 = so * cw + co * sw * ci
    r11 = -so * sw + co * cw * ci
    r20 = sw * si
    r21 = cw * si

    r = np.empty(3)
    v = np.empty(3)
    r[0] = r00 * rp0 + r01 * rp1
    r[1] = r10 * rp0 + r11 * rp1
    r[2] = r20 * rp0 + r21 * rp1
    v[0] = r00 * vp0 + r01 * vp1
    v[1] = r10 * vp0 + r11 * vp1
    v[2] = r20 * vp0 + r21 * vp1
    return r, v


@njit(cache=True)
def oelem_from_rv(rx, ry, rz, vx, vy, vz):
    """Return (a, inclination, RAAN, true anomaly, argument of latitude).

    The argument of latitude is taken straight from the geometry rather than as
    argp + nu, so it stays well conditioned as the eccentricity goes to zero.
    """
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h = den(math.sqrt(hx * hx + hy * hy + hz * hz), R_FLOOR)

    c = hz / h
    c = min(1.0, max(-1.0, c))
    inc = math.acos(c)

    # n = z_hat x h
    nx, ny = -hy, hx
    n = math.hypot(nx, ny)
    raan = 0.0 if n < 1e-10 else math.atan2(ny, nx)

    rn = den(math.sqrt(rx * rx + ry * ry + rz * rz), R_FLOOR)
    vn2 = vx * vx + vy * vy + vz * vz
    rdotv = rx * vx + ry * vy + rz * vz
    k = vn2 - MU / rn
    ex = (k * rx - rdotv * vx) / MU
    ey = (k * ry - rdotv * vy) / MU
    ez = (k * rz - rdotv * vz) / MU
    ecc = math.sqrt(ex * ex + ey * ey + ez * ez)

    if ecc < 1e-8:
        # Argument of latitude stands in for the true anomaly on a circular
        # orbit, which is what the phasing calculation actually wants.
        if n < 1e-10:
            nu = math.atan2(ry, rx)
        else:
            cu = (nx * rx + ny * ry) / den(n * rn)
            cu = min(1.0, max(-1.0, cu))
            nu = math.acos(cu)
            if rz < 0.0:
                nu = 2.0 * math.pi - nu
    else:
        cn = (ex * rx + ey * ry + ez * rz) / den(ecc * rn)
        cn = min(1.0, max(-1.0, cn))
        nu = math.acos(cn)
        if rdotv < 0.0:
            nu = 2.0 * math.pi - nu

    # Argument of latitude, from the geometry: the angle at the centre from the
    # ascending node to the position vector, positive along the motion.
    si = math.sin(inc)
    if si < 1e-12:
        arglat = math.atan2(ry, rx) - raan
    else:
        arglat = math.atan2(rz / den(si), rx * math.cos(raan) + ry * math.sin(raan))

    energy = 0.5 * vn2 - MU / rn
    sma = math.inf if abs(energy) < 1e-10 else -MU / (2.0 * energy)
    return (sma, inc, raan % (2.0 * math.pi), nu % (2.0 * math.pi),
            arglat % (2.0 * math.pi))


@njit(cache=True, inline="always")
def _accel(rx, ry, rz, use_j2):
    r2 = den(rx * rx + ry * ry + rz * rz, R_FLOOR * R_FLOOR)
    rmag = math.sqrt(r2)
    f = -MU / (rmag * r2)
    ax, ay, az = f * rx, f * ry, f * rz
    if use_j2:
        z2 = rz * rz
        c = -3.0 * J2 * MU * R_E * R_E / (2.0 * r2 * r2 * rmag)
        q = 5.0 * z2 / r2
        ax += c * (1.0 - q) * rx
        ay += c * (1.0 - q) * ry
        az += c * (3.0 - q) * rz
    return ax, ay, az


@njit(cache=True, parallel=True)
def propagate(r0, v0, dt, n_steps, write_every, use_j2):
    """RK4-propagate every node and record every `write_every` steps.

    r0, v0 are (N, 3). Returns positions (N, n_rec, 3), velocities
    (N, n_rec, 3), elements (N, n_rec, 4) and times (n_rec,), with record 0 at
    t = 0.
    """
    n_nodes = r0.shape[0]
    n_rec = n_steps // write_every
    pos = np.empty((n_nodes, n_rec, 3))
    vel = np.empty((n_nodes, n_rec, 3))
    oe = np.empty((n_nodes, n_rec, 5))

    for i in prange(n_nodes):
        rx, ry, rz = r0[i, 0], r0[i, 1], r0[i, 2]
        vx, vy, vz = v0[i, 0], v0[i, 1], v0[i, 2]
        rec = 0
        for step in range(n_steps):
            if step % write_every == 0 and rec < n_rec:
                pos[i, rec, 0] = rx
                pos[i, rec, 1] = ry
                pos[i, rec, 2] = rz
                vel[i, rec, 0] = vx
                vel[i, rec, 1] = vy
                vel[i, rec, 2] = vz
                a, inc, raan, nu, u = oelem_from_rv(rx, ry, rz, vx, vy, vz)
                oe[i, rec, 0] = a
                oe[i, rec, 1] = inc
                oe[i, rec, 2] = raan
                oe[i, rec, 3] = nu
                oe[i, rec, 4] = u
                rec += 1

            k1rx, k1ry, k1rz = vx, vy, vz
            k1vx, k1vy, k1vz = _accel(rx, ry, rz, use_j2)

            k2rx = vx + 0.5 * dt * k1vx
            k2ry = vy + 0.5 * dt * k1vy
            k2rz = vz + 0.5 * dt * k1vz
            k2vx, k2vy, k2vz = _accel(rx + 0.5 * dt * k1rx,
                                      ry + 0.5 * dt * k1ry,
                                      rz + 0.5 * dt * k1rz, use_j2)

            k3rx = vx + 0.5 * dt * k2vx
            k3ry = vy + 0.5 * dt * k2vy
            k3rz = vz + 0.5 * dt * k2vz
            k3vx, k3vy, k3vz = _accel(rx + 0.5 * dt * k2rx,
                                      ry + 0.5 * dt * k2ry,
                                      rz + 0.5 * dt * k2rz, use_j2)

            k4rx = vx + dt * k3vx
            k4ry = vy + dt * k3vy
            k4rz = vz + dt * k3vz
            k4vx, k4vy, k4vz = _accel(rx + dt * k3rx,
                                      ry + dt * k3ry,
                                      rz + dt * k3rz, use_j2)

            s = dt / 6.0
            rx += s * (k1rx + 2.0 * k2rx + 2.0 * k3rx + k4rx)
            ry += s * (k1ry + 2.0 * k2ry + 2.0 * k3ry + k4ry)
            rz += s * (k1rz + 2.0 * k2rz + 2.0 * k3rz + k4rz)
            vx += s * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
            vy += s * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
            vz += s * (k1vz + 2.0 * k2vz + 2.0 * k3vz + k4vz)

    times = np.arange(n_rec) * (dt * write_every)
    return pos, vel, oe, times


def mean_to_true(mean_anom, ecc, tol=1e-12, itmax=50):
    """Kepler's equation, then the true anomaly."""
    e_anom = mean_anom if ecc < 0.8 else math.pi
    for _ in range(itmax):
        f = e_anom - ecc * math.sin(e_anom) - mean_anom
        fp = 1.0 - ecc * math.cos(e_anom)
        step = f / (fp if abs(fp) > 1e-14 else math.copysign(1e-14, fp or 1.0))
        e_anom -= step
        if abs(step) < tol:
            break
    return 2.0 * math.atan2(math.sqrt(1.0 + ecc) * math.sin(0.5 * e_anom),
                            math.sqrt(1.0 - ecc) * math.cos(0.5 * e_anom))


def load_population(path):
    names, states = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ecc = float(row["ecc"])
            nu = mean_to_true(math.radians(float(row["mean_anom_deg"])), ecc)
            r, v = eci_from_oelem(float(row["a_km"]), ecc,
                                  math.radians(float(row["incl_deg"])),
                                  math.radians(float(row["raan_deg"])),
                                  math.radians(float(row["argp_deg"])), nu)
            names.append(row["name"])
            states.append((r, v))
    r0 = np.array([s[0] for s in states])
    v0 = np.array([s[1] for s in states])
    return names, r0, v0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", default="outputs/instance_population.csv")
    p.add_argument("--out", default="outputs/simulation.h5")
    p.add_argument("--days", type=float, default=T_END_DAYS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--write-every", type=int, default=WRITE_EVERY)
    p.add_argument("--no-j2", action="store_true")
    args = p.parse_args()

    import h5py

    names, r0, v0 = load_population(args.population)
    n_steps = int(round(args.days * 86400.0 / args.dt))
    n_rec = n_steps // args.write_every
    print(f"nodes {len(names)}   steps {n_steps}   records {n_rec}   "
          f"record spacing {args.dt * args.write_every / 3600:.3f} h")

    t0 = time.time()
    pos, vel, oe, times = propagate(r0, v0, args.dt, n_steps,
                                    args.write_every, not args.no_j2)
    print(f"propagation took {time.time() - t0:.1f} s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with h5py.File(args.out, "w") as f:
        g = f.create_group("metadata")
        g.create_dataset("names", data=np.array(names, dtype="S32"))
        g.create_dataset("times", data=times)
        g.attrs["J2"] = np.uint8(0 if args.no_j2 else 1)
        g.attrs["dt"] = args.dt
        g.attrs["t_end"] = args.days * 86400.0
        g.attrs["source"] = os.path.basename(args.population)
        g.attrs["R_E"] = R_E
        g.attrs["J2_value"] = J2
        for i, name in enumerate(names):
            sg = f.create_group(name)
            sg.create_dataset("positions", data=pos[i], compression="lzf")
            sg.create_dataset("velocities", data=vel[i], compression="lzf")
            sg.create_dataset("orbital_elements", data=oe[i], compression="lzf")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({size_mb:.1f} MB)")

    # Conservation check: a two-body-plus-J2 RK4 integration should hold the
    # semi-major axis and inclination essentially fixed over the horizon.
    da = np.abs(oe[:, :, 0] - oe[:, :1, 0]).max(axis=1)
    di = np.degrees(np.abs(oe[:, :, 1] - oe[:, :1, 1])).max(axis=1)
    print(f"max |da| over all nodes and time : {da.max():.4f} km")
    print(f"max |di| over all nodes and time : {di.max():.5f} deg")


if __name__ == "__main__":
    main()
