"""Build the instance population: 100 Starlink + 124 Planet Labs + 1 depot.

Reproduces the sampling rule of `starlink/fetch_and_sample.jl` and
`starlink/fetch_planet_labs.jl` from the cached CelesTrak GP catalogues, and
writes the full osculating element set that the propagator needs.

The predecessor file `outputs/instance_population.csv` recorded only
(name, a, inclination, group), which was enough for the transfer-model
cross-validation but not for propagation: RK4 needs a Cartesian state, so
eccentricity, RAAN, argument of perigee and mean anomaly are all required.

Sampling rule (Starlink)
------------------------
1. Assign each catalogue object to one of five shells by (inclination,
   altitude) with tolerances 1.5 deg and 30 km. Unassigned objects are dropped.
2. Bin the survivors into 5-degree RAAN planes within each shell.
3. Allocate the 100 slots across shells in proportion to each shell's catalogue
   population, fixing the rounding residual on the largest shells first.
4. Within a shell, walk the planes in shuffled order taking at most two
   satellites per plane until the shell's allocation is filled.

Planet Labs takes the whole catalogue, so no sampling is involved. TANAGER-1
is *retained* here: Table 2 of the paper counts 124 Planet Labs clients, and the
sentence that excludes TANAGER-1 (Section 4.1) is about asset valuation, not
about the population. It is flagged in the `valued` column so the valuation step
can drop it without the population changing.

Reproducibility. The rule above is the Julia one; the *draw* is not, because
numpy's PCG64 and Julia's MersenneTwister cannot be made to agree. The seed is
fixed so this script is reproducible on its own terms. See F10 in the plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

from oos.constants import MU, R_E

HERE = os.path.dirname(os.path.abspath(__file__))

STARLINK_CATALOG = os.path.join(HERE, "starlink", "starlink_catalog.json")
PLANET_CATALOG = os.path.join(HERE, "starlink", "planet_catalog.json")

# (inclination [deg], altitude [km]) of the five Starlink shells.
SHELLS = ((53.0, 550.0), (53.2, 540.0), (70.0, 570.0), (97.6, 560.0), (43.0, 530.0))
INC_TOL = 1.5      # deg
ALT_TOL = 30.0     # km
PLANE_BIN_DEG = 5.0
SATS_PER_PLANE = 2

N_STARLINK = 100
SEED = 42

# Nominal depot, as in starlink/run_mixed_fleet.jl: sun-synchronous, co-planar
# with the Planet Labs population and the Starlink SSO shell. The (a, i) sweep
# of Section 5 moves this; RAAN is held at the nominal value (decision D12).
DEPOT_ALT_KM = 560.0
DEPOT_INC_DEG = 97.6
DEPOT_RAAN_DEG = 0.0

FIELDS = ("name", "object_name", "object_id", "group", "shell", "valued",
          "a_km", "ecc", "incl_deg", "raan_deg", "argp_deg", "mean_anom_deg")


def sma_from_mean_motion(n_rev_per_day):
    """Semi-major axis from the GP mean motion, in km."""
    n = n_rev_per_day * 2.0 * math.pi / 86400.0
    return (MU / n**2) ** (1.0 / 3.0)


def assign_shell(inc_deg, alt_km):
    """Index of the matching Starlink shell, or None."""
    for k, (inc, alt) in enumerate(SHELLS):
        if abs(inc_deg - inc) <= INC_TOL and abs(alt_km - alt) <= ALT_TOL:
            return k
    return None


def sample_starlink(catalog, n_targets, rng):
    shell_planes = [dict() for _ in SHELLS]
    n_unrecognised = 0
    for obj in catalog:
        inc_deg = float(obj["INCLINATION"])
        a = sma_from_mean_motion(float(obj["MEAN_MOTION"]))
        shell = assign_shell(inc_deg, a - R_E)
        if shell is None:
            n_unrecognised += 1
            continue
        plane = int(float(obj["RA_OF_ASC_NODE"]) // PLANE_BIN_DEG)
        shell_planes[shell].setdefault(plane, []).append(obj)

    counts = [sum(len(v) for v in planes.values()) for planes in shell_planes]
    total = sum(counts)
    if total == 0:
        raise SystemExit("no catalogue object fell in any shell; check the tolerances")

    if n_targets >= total:
        alloc = list(counts)
    else:
        alloc = [int(round(n_targets * c / total)) for c in counts]
        residual = n_targets - sum(alloc)
        if residual:
            order = np.argsort(counts)[::-1]
            for i in range(abs(residual)):
                alloc[order[i % len(order)]] += int(np.sign(residual))

    sampled = []
    for shell, planes in enumerate(shell_planes):
        if alloc[shell] <= 0:
            continue
        plane_keys = list(planes.keys())
        rng.shuffle(plane_keys)
        taken = 0
        for key in plane_keys:
            if taken >= alloc[shell]:
                break
            entries = list(planes[key])
            rng.shuffle(entries)
            take = min(SATS_PER_PLANE, len(entries), alloc[shell] - taken)
            for obj in entries[:take]:
                sampled.append((obj, shell))
            taken += take

    return sampled, n_unrecognised, counts, alloc


def row_from_gp(name, obj, group, shell):
    return dict(
        name=name,
        object_name=str(obj.get("OBJECT_NAME", "")),
        object_id=str(obj.get("OBJECT_ID", "")),
        group=group,
        shell="" if shell is None else shell,
        valued=0 if "TANAGER" in str(obj.get("OBJECT_NAME", "")).upper() else 1,
        a_km=round(sma_from_mean_motion(float(obj["MEAN_MOTION"])), 6),
        ecc=float(obj["ECCENTRICITY"]),
        incl_deg=float(obj["INCLINATION"]),
        raan_deg=float(obj["RA_OF_ASC_NODE"]),
        argp_deg=float(obj["ARG_OF_PERICENTER"]),
        mean_anom_deg=float(obj["MEAN_ANOMALY"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/instance_population.csv")
    p.add_argument("--n-starlink", type=int, default=N_STARLINK)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    with open(STARLINK_CATALOG) as fh:
        starlink = json.load(fh)
    with open(PLANET_CATALOG) as fh:
        planet = json.load(fh)

    rng = np.random.default_rng(args.seed)
    sampled, n_unrecognised, counts, alloc = sample_starlink(
        starlink, args.n_starlink, rng)

    print(f"Starlink catalogue          : {len(starlink)} objects, "
          f"{n_unrecognised} outside every shell")
    for k, (inc, alt) in enumerate(SHELLS):
        print(f"  shell {k} i={inc:5.1f} deg alt={alt:5.1f} km : "
              f"{counts[k]:5d} in catalogue -> {alloc[k]:3d} sampled")
    print(f"Starlink sampled            : {len(sampled)}")

    rows = []
    for k, (obj, shell) in enumerate(sampled, start=1):
        rows.append(row_from_gp(f"sat_{k}", obj, "starlink", shell))

    n_unvalued = sum(1 for o in planet
                     if "TANAGER" in str(o.get("OBJECT_NAME", "")).upper())
    print(f"Planet Labs catalogue       : {len(planet)} objects, all retained "
          f"({n_unvalued} flagged unvalued: TANAGER)")
    offset = len(rows)
    for k, obj in enumerate(planet, start=1):
        rows.append(row_from_gp(f"sat_{offset + k}", obj, "planet", None))

    rows.append(dict(name="depot_1", object_name="DEPOT", object_id="",
                     group="depot", shell="", valued=0,
                     a_km=round(R_E + DEPOT_ALT_KM, 6), ecc=0.0,
                     incl_deg=DEPOT_INC_DEG, raan_deg=DEPOT_RAAN_DEG,
                     argp_deg=0.0, mean_anom_deg=0.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)

    n_sl = sum(1 for r in rows if r["group"] == "starlink")
    n_pl = sum(1 for r in rows if r["group"] == "planet")
    print(f"\nwrote {args.out}: {len(rows)} nodes "
          f"({n_sl} starlink + {n_pl} planet + 1 depot)")
    ecc = np.array([r["ecc"] for r in rows])
    print(f"eccentricity: max {ecc.max():.5f}, mean {ecc.mean():.5f}")


if __name__ == "__main__":
    main()
