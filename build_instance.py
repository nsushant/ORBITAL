"""Build the instance population: simulated Starlink planes plus the real Planet fleet.

Two populations, built differently, because the two constellations *are*
different and pretending otherwise was costing us.

Starlink is a maintained lattice. In the CelesTrak catalogue its plane structure
is unambiguous: satellites in a plane sit within 0.006-0.03 degrees of each
other in RAAN while planes are 1.4-8.8 degrees apart, a separation of one to
three orders of magnitude, and the 53-degree shell resolves to 22 satellites per
plane, which is Starlink's published design. So its planes are taken from the
catalogue and each is *populated* with evenly phased satellites. Members of a
plane then share a, inclination and RAAN to machine precision rather than to a
tolerance, which is what makes the cost table exact: they share a nodal rate,
hence their node at every epoch, hence every transfer. Only the along-track
phase differs, and the phasing manoeuvre handles that.

Planet Labs is not a lattice, and is kept as it flies. Its satellites are
rideshare-deployed and decaying, spread over 352-604 km, and objects that are
close in RAAN differ by 15-66 km in altitude -- so they do not share a nodal
rate and are not plane-mates in any sense that helps. Each keeps its own orbit
and counts as its own plane.

Asset values are carried through, because recovery potential is an objective of
the study and it has to mean something. A simulated Starlink satellite inherits
a type and a launch epoch sampled from the real members of the plane it sits in,
so the mix of V1.5 and V2 hardware, and the age distribution that depreciates
it, both match the catalogue.

Usage
-----
    python3 build_instance.py --out outputs/instance_population.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

from oos.constants import MU, R_E
from oos.propagate import eci_from_oelem, propagate

HERE = os.path.dirname(os.path.abspath(__file__))
STARLINK_CATALOG = os.path.join(HERE, "starlink", "starlink_catalog.json")
PLANET_CATALOG = os.path.join(HERE, "starlink", "planet_catalog.json")

# (inclination [deg], altitude [km]) of the Starlink shells, as Table 2 has them.
SHELLS = ((53.0, 550.0), (53.2, 540.0), (70.0, 570.0), (97.6, 560.0), (43.0, 530.0))
INC_TOL = 1.5
ALT_TOL = 30.0

# Splitting RAAN at 0.5 degrees sits between the two scales by a wide margin:
# plane-mates are within 0.03 degrees, planes are 1.4 degrees apart at worst.
# Anything from about 0.1 to 1.0 gives the same planes, so this is not a knob.
PLANE_SPLIT_DEG = 0.5
MIN_PLANE_MEMBERS = 4     # a "plane" seen with fewer real members is not evidence of one

N_PLANES = 14
SATS_PER_PLANE = 16
SEED = 42

# Depot, as in starlink/run_mixed_fleet.jl: sun-synchronous, co-planar with the
# Planet population and Starlink's SSO shell.
DEPOT_ALT_KM = 560.0
DEPOT_INC_DEG = 97.6
DEPOT_RAAN_DEG = 0.0

# Asset values, pre-depreciation. Starlink: manufacture at $1000/kg plus launch
# at SpaceX's internal $1850/kg, against DAS-filed masses. Planet: flat figures
# at Planet's production scale.
INT_LAUNCH_RATE = 1850.0
V1_MASS_KG, V2_MASS_KG = 303.0, 800.0
V1_VALUE = (1000.0 + INT_LAUNCH_RATE) * V1_MASS_KG      # $864,150
V2_VALUE = (1000.0 + INT_LAUNCH_RATE) * V2_MASS_KG      # $2,280,000
DOVE_VALUE = 200_000.0
SKYSAT_VALUE = 3_000_000.0

WEIBULL_LAMBDA, WEIBULL_K = 5.0, 1.5
SIM_START_YEAR = 2024.0

FIELDS = ("name", "object_name", "object_id", "group", "shell", "plane",
          "sat_type", "valued", "value_usd", "age_years",
          "a_km", "ecc", "incl_deg", "raan_deg", "argp_deg", "mean_anom_deg")


def sma_from_mean_motion(n_rev_per_day):
    n = n_rev_per_day * 2.0 * math.pi / 86400.0
    return (MU / n**2) ** (1.0 / 3.0)


def assign_shell(inc_deg, alt_km):
    for k, (inc, alt) in enumerate(SHELLS):
        if abs(inc_deg - inc) <= INC_TOL and abs(alt_km - alt) <= ALT_TOL:
            return k
    return None


def launch_year(object_id):
    try:
        return int(str(object_id)[:4])
    except (ValueError, TypeError):
        return None


def starlink_type(object_id):
    """V2 mini from 2024 on, and from launch 29 of 2023."""
    s = str(object_id)
    y = launch_year(s)
    if y is None:
        return "v1"
    try:
        launch_no = int(s[5:8])
    except (ValueError, IndexError):
        launch_no = 0
    if y >= 2024 or (y == 2023 and launch_no >= 29):
        return "v2"
    return "v1"


def weibull_depreciate(value, age_years):
    if age_years <= 0.0:
        return value
    return value * math.exp(-((age_years / WEIBULL_LAMBDA) ** WEIBULL_K))


def elements_from_rv(r, v):
    """Full classical elements (a, e, i, RAAN, argp, mean anomaly) from a state."""
    h = np.cross(r, v)
    n_vec = np.array([-h[1], h[0], 0.0])
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    e_vec = ((vn * vn - MU / rn) * r - np.dot(r, v) * v) / MU
    ecc = np.linalg.norm(e_vec)
    a = 1.0 / (2.0 / rn - vn * vn / MU)
    inc = math.acos(max(-1.0, min(1.0, h[2] / np.linalg.norm(h))))
    n_norm = np.linalg.norm(n_vec)
    raan = math.atan2(n_vec[1], n_vec[0]) % (2.0 * math.pi) if n_norm > 1e-12 else 0.0
    if ecc > 1e-12 and n_norm > 1e-12:
        argp = math.acos(max(-1.0, min(1.0, np.dot(n_vec, e_vec) / (n_norm * ecc))))
        if e_vec[2] < 0.0:
            argp = 2.0 * math.pi - argp
        nu = math.acos(max(-1.0, min(1.0, np.dot(e_vec, r) / (ecc * rn))))
        if np.dot(r, v) < 0.0:
            nu = 2.0 * math.pi - nu
    else:
        argp = 0.0
        nu = math.atan2(np.dot(r, np.cross(h / np.linalg.norm(h), n_vec / max(n_norm, 1e-12))),
                        np.dot(r, n_vec / max(n_norm, 1e-12))) % (2.0 * math.pi)
    e_anom = 2.0 * math.atan2(math.sqrt(max(1.0 - ecc, 0.0)) * math.sin(0.5 * nu),
                              math.sqrt(1.0 + ecc) * math.cos(0.5 * nu))
    mean_anom = (e_anom - ecc * math.sin(e_anom)) % (2.0 * math.pi)
    return a, ecc, inc, raan, argp, mean_anom


def phase_a_plane(a_km, incl_deg, raan_deg, n_sats, dt=1.0):
    """Elements for n_sats evenly phased on ONE orbit.

    Not n_sats copies of the same osculating elements at different anomalies:
    that is a different orbit each time. The J2 short-period correction between
    osculating and mean elements depends on where in the orbit you are, so
    stamping one osculating semi-major axis at n different phases produces n
    orbits whose *mean* semi-major axes differ by the short-period amplitude --
    measured at 19 km here, enough to give plane-mates nodal rates 0.03 deg/day
    apart and 13 degrees of relative drift over the horizon.

    Instead, one seed is propagated for a full revolution and sampled at n
    equally spaced times. Those states lie on the same trajectory by
    construction, so the satellites share every mean element exactly and differ
    only in phase, which is what a constellation plane is.
    """
    incl = math.radians(incl_deg)
    r, v = eci_from_oelem(a_km, 0.0, incl, math.radians(raan_deg), 0.0, 0.0)
    period = 2.0 * math.pi * math.sqrt(a_km**3 / MU)
    n_steps = int(round(period / dt))
    # Record often enough that the sampled phases land within a fraction of a
    # degree of the targets.
    write_every = max(1, n_steps // (n_sats * 64))
    pos, vel, _, times, _ = propagate(np.array([r]), np.array([v]), dt, n_steps,
                                      write_every, True)
    out = []
    for k in range(n_sats):
        target = period * k / n_sats
        j = int(np.argmin(np.abs(times - target)))
        a, ecc, inc, raan, argp, m = elements_from_rv(pos[0, j], vel[0, j])
        out.append((a, ecc, math.degrees(inc), math.degrees(raan) % 360.0,
                    math.degrees(argp) % 360.0, math.degrees(m) % 360.0))
    return out


def find_catalogue_planes(objects, split_deg=PLANE_SPLIT_DEG):
    """Split a shell's objects into planes on a RAAN gap.

    Returns a list of lists of indices into `objects`, largest plane first.
    """
    raan = np.array([float(o["RA_OF_ASC_NODE"]) % 360.0 for o in objects])
    order = np.argsort(raan)
    planes, current = [], [order[0]]
    for k in range(1, len(order)):
        if raan[order[k]] - raan[order[k - 1]] <= split_deg:
            current.append(order[k])
        else:
            planes.append(current)
            current = [order[k]]
    planes.append(current)
    # The circle wraps: a plane straddling 0 degrees appears at both ends.
    if len(planes) > 1 and (raan[order[0]] + 360.0 - raan[order[-1]]) <= split_deg:
        planes[0] = planes[0] + planes.pop()
    planes.sort(key=len, reverse=True)
    return planes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/instance_population.csv")
    p.add_argument("--n-planes", type=int, default=N_PLANES)
    p.add_argument("--sats-per-plane", type=int, default=SATS_PER_PLANE)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    with open(STARLINK_CATALOG) as fh:
        starlink = json.load(fh)
    with open(PLANET_CATALOG) as fh:
        planet = json.load(fh)

    # ---- Starlink: find real planes, then populate the chosen ones -----------
    by_shell = {k: [] for k in range(len(SHELLS))}
    for o in starlink:
        a = sma_from_mean_motion(float(o["MEAN_MOTION"]))
        shell = assign_shell(float(o["INCLINATION"]), a - R_E)
        if shell is not None:
            by_shell[shell].append(o)

    shell_planes = {}
    for k, objs in by_shell.items():
        if len(objs) < MIN_PLANE_MEMBERS:
            shell_planes[k] = []
            continue
        shell_planes[k] = [pl for pl in find_catalogue_planes(objs)
                           if len(pl) >= MIN_PLANE_MEMBERS]
    print("Starlink planes found in the catalogue "
          f"(split at {PLANE_SPLIT_DEG} deg of RAAN, >= {MIN_PLANE_MEMBERS} members):")
    for k, (inc, alt) in enumerate(SHELLS):
        n_pl = len(shell_planes[k])
        occ = [len(pl) for pl in shell_planes[k]]
        print(f"  shell {k} i={inc:5.1f} alt={alt:5.1f} : {len(by_shell[k]):5d} objects "
              f"-> {n_pl:4d} planes"
              + (f", {min(occ)}-{max(occ)} satellites each" if n_pl else ""))

    # Allocate the planes we take across shells in proportion to how many each
    # shell actually has, so the instance reflects where the constellation is.
    counts = np.array([len(shell_planes[k]) for k in range(len(SHELLS))], float)
    if counts.sum() == 0:
        raise SystemExit("no Starlink planes found; check the shell tolerances")
    alloc = np.floor(args.n_planes * counts / counts.sum()).astype(int)
    while alloc.sum() < args.n_planes:
        # Give the remainder to whichever shell is most under its exact share.
        deficit = args.n_planes * counts / counts.sum() - alloc
        deficit[counts == 0] = -1
        alloc[int(np.argmax(deficit))] += 1
    print(f"\ntaking {args.n_planes} planes: "
          + ", ".join(f"shell {k}: {alloc[k]}" for k in range(len(SHELLS)) if alloc[k]))

    rows = []
    plane_id = 0
    for k in range(len(SHELLS)):
        if alloc[k] == 0:
            continue
        chosen = shell_planes[k][:alloc[k]]
        for pl in chosen:
            members = [by_shell[k][i] for i in pl]
            a_km = float(np.mean([sma_from_mean_motion(float(o["MEAN_MOTION"]))
                                  for o in members]))
            incl = float(np.mean([float(o["INCLINATION"]) for o in members]))
            w = np.radians([float(o["RA_OF_ASC_NODE"]) % 360.0 for o in members])
            raan = math.degrees(math.atan2(np.sin(w).mean(), np.cos(w).mean())) % 360.0

            # Types and ages come from the real members, so the value mix and the
            # age distribution that depreciates it both match the catalogue.
            types = [starlink_type(o.get("OBJECT_ID", "")) for o in members]
            years = [launch_year(o.get("OBJECT_ID", "")) for o in members]
            years = [y for y in years if y is not None] or [2022]

            phased = phase_a_plane(a_km, incl, raan, args.sats_per_plane)
            for s in range(args.sats_per_plane):
                pa, pe, pi_, pw, pg, pm = phased[s]
                t = types[rng.integers(len(types))]
                yr = years[rng.integers(len(years))]
                age = max(0.0, SIM_START_YEAR - (yr + 0.5))
                base = V2_VALUE if t == "v2" else V1_VALUE
                rows.append(dict(
                    name=f"sat_{len(rows) + 1}",
                    object_name=f"SIM-SL-P{plane_id:02d}-{s:02d}",
                    object_id="", group="starlink", shell=k, plane=plane_id,
                    sat_type=t, valued=1,
                    value_usd=round(weibull_depreciate(base, age), 2),
                    age_years=round(age, 2),
                    a_km=round(pa, 9), ecc=round(pe, 12),
                    incl_deg=round(pi_, 9), raan_deg=round(pw, 9),
                    argp_deg=round(pg, 9),
                    # Phase is the only thing distinguishing members, and it is
                    # what the phasing manoeuvre pays for.
                    mean_anom_deg=round(pm, 9)))
            plane_id += 1

    n_starlink = len(rows)

    # ---- Planet Labs: kept exactly as it flies ------------------------------
    for o in planet:
        name = str(o.get("OBJECT_NAME", ""))
        is_skysat = "SKYSAT" in name.upper()
        yr = launch_year(o.get("OBJECT_ID", "")) or 2020
        age = max(0.0, SIM_START_YEAR - (yr + 0.5))
        base = SKYSAT_VALUE if is_skysat else DOVE_VALUE
        rows.append(dict(
            name=f"sat_{len(rows) + 1}",
            object_name=name, object_id=str(o.get("OBJECT_ID", "")),
            group="planet", shell="", plane=plane_id,
            sat_type="skysat" if is_skysat else "dove",
            valued=0 if "TANAGER" in name.upper() else 1,
            value_usd=round(weibull_depreciate(base, age), 2),
            age_years=round(age, 2),
            a_km=round(sma_from_mean_motion(float(o["MEAN_MOTION"])), 6),
            ecc=float(o["ECCENTRICITY"]), incl_deg=float(o["INCLINATION"]),
            raan_deg=float(o["RA_OF_ASC_NODE"]),
            argp_deg=float(o["ARG_OF_PERICENTER"]),
            mean_anom_deg=float(o["MEAN_ANOMALY"])))
        plane_id += 1     # each Planet satellite is its own plane

    n_planet = len(rows) - n_starlink

    rows.append(dict(name="depot_1", object_name="DEPOT", object_id="",
                     group="depot", shell="", plane=plane_id, sat_type="depot",
                     valued=0, value_usd=0.0, age_years=0.0,
                     a_km=round(R_E + DEPOT_ALT_KM, 6), ecc=0.0,
                     incl_deg=DEPOT_INC_DEG, raan_deg=DEPOT_RAAN_DEG,
                     argp_deg=0.0, mean_anom_deg=0.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)

    n_planes_total = plane_id + 1
    print(f"\nwrote {args.out}")
    print(f"  {n_starlink} simulated Starlink on {args.n_planes} real planes "
          f"({args.sats_per_plane} each)")
    print(f"  {n_planet} real Planet Labs orbits, one plane each")
    print(f"  1 depot")
    print(f"  {len(rows)} nodes on {n_planes_total} distinct planes -> "
          f"{n_planes_total * (n_planes_total - 1):,} plane pairs instead of "
          f"{len(rows) * (len(rows) - 1):,} node pairs "
          f"({len(rows) * (len(rows) - 1) / (n_planes_total * (n_planes_total - 1)):.1f}x fewer)")
    vals = np.array([r["value_usd"] for r in rows if r["valued"]])
    print(f"  recovery potential: ${vals.sum() / 1e6:.1f} M over {len(vals)} clients, "
          f"median ${np.median(vals) / 1e3:.0f} k, range "
          f"${vals.min() / 1e3:.0f} k - ${vals.max() / 1e6:.2f} M")


if __name__ == "__main__":
    main()
