"""
Cross-validation of the fast Edelbaum model against the Lu et al. reference.

Second stage of the two-stage validation chain of Section 3.2.6. The first
stage established that our implementation of Lu et al. is faithful
(validate_lu.py reproduces their published cases to 0.41 %). This stage asks how
far the fast analytical model, the one the optimiser actually runs on, departs
from that reference over the transfer geometries the instance set contains.

Both models get the same servicer, so the comparison isolates the modelling
difference:

  same   thrust acceleration f = T/m, semi-major axes, inclinations, RAAN gaps,
         times of flight, and the constants in oos/constants.py
  differ Edelbaum tracks mass depletion along an arc, so its acceleration rises
         as propellant burns; the Lu averaged dynamics hold f fixed.

Masking. A geometry is compared only when *both* models return a transfer.
Cases where one finds a transfer and the other does not are counted and
reported separately, never dropped silently or filled with a sentinel - the
disagreement about feasibility is itself a result, and mixing sentinels into a
mean is how the earlier lu_edelbaum_compare.csv became unusable.

Usage
-----
    python3 crossvalidate_transfer_models.py --population outputs/instance_population.csv

The population CSV has columns name,a_km,incl_deg[,group] and lists the client
objects of the instance. Ordered pairs are drawn from it, so the sampled
(delta-a, delta-I) span exactly what the cost table contains.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

from oos import servicer
from oos.edelbaum import transfer_dv
from oos.lu import coasting_orbit_estimate, raan_drift_rate, transfer_cost_nlp

# The servicer of Section 3.2.1, read from oos.servicer rather than restated.
# It used to be restated here as an Otter-class 335 kg / 2800 s / 10 mN vehicle,
# and when Section 3.2.1 became the Exotrail spacevan this file went on
# comparing the old one at a third of the thrust acceleration -- quietly
# invalidating the numbers Section 3.2.6 quotes. The comparison is sensitive to
# it: the gap between the two models is set by how long the arcs take relative
# to the drift phase, and arc duration scales with 1/f.
MASS = servicer.MASS
ISP = servicer.ISP
THRUST = servicer.THRUST


def load_population(path):
    objects = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            objects.append((row["name"], float(row["a_km"]),
                            math.radians(float(row["incl_deg"]))))
    if len(objects) < 2:
        raise SystemExit(f"{path}: need at least two objects to form transfer pairs")
    return objects


def sample_geometries(objects, n, tof_range, rng):
    """Ordered pairs of distinct objects, with a uniform RAAN gap.

    The RAAN gap is uniform on the full circle because relative nodes drift
    through all values over a campaign of the length considered here, so every
    gap is encountered somewhere in the cost table.
    """
    out = []
    while len(out) < n:
        i, j = rng.integers(len(objects)), rng.integers(len(objects))
        if i == j:
            continue
        name_a, a0, incl0 = objects[i]
        name_b, af, inclf = objects[j]
        out.append(dict(from_obj=name_a, to_obj=name_b, a0=a0, incl0=incl0,
                        af=af, inclf=inclf,
                        raan_gap=rng.uniform(-math.pi, math.pi),
                        tof_days=rng.uniform(*tof_range)))
    return out


def run(geometries, mass, isp, thrust):
    """Evaluate both models on the same physical rendezvous.

    The two models take the target node in different conventions, and getting
    this wrong silently compares different problems. The physical statement is:
    the target starts at RAAN `raan_gap` and regresses under J2 at its own rate,
    and the servicer must match its node on arrival.

      Lu       is given the target's *initial* node and propagates it internally
               (`raan_target = raanf + raan_drift_rate(af, inclf) * Tf`).
      Edelbaum is given the required *absolute* RAAN change, since the cost
               table supplies nodes already propagated to the arrival epoch.

    So Edelbaum receives the propagated node and Lu the initial one. Passing the
    same number to both makes Lu solve for an extra `drift * Tf` of nodal
    motion - several radians over a year - and inflates the apparent
    disagreement by orders of magnitude.
    """
    fmax = thrust / mass
    rows = []
    for g in geometries:
        tof_sec = g["tof_days"] * 86400.0
        raan_arrival = g["raan_gap"] + raan_drift_rate(g["af"], g["inclf"]) * tof_sec

        eb = transfer_dv(g["a0"], g["incl0"], 0.0, g["af"], g["inclf"],
                         raan_arrival, g["tof_days"],
                         mass=mass, isp=isp, thrust=thrust)
        nlp = transfer_cost_nlp(g["a0"], g["incl0"], 0.0, g["af"], g["inclf"],
                                g["raan_gap"], g["tof_days"], fmax=fmax)
        orbit = coasting_orbit_estimate(g["a0"], g["incl0"], 0.0, g["af"],
                                        g["inclf"], g["raan_gap"], g["tof_days"])
        est = orbit.j_total * 1000.0 if orbit is not None else None
        rel = 100.0 * (eb - nlp) / nlp if (eb is not None and nlp) else None

        rows.append(dict(
            from_obj=g["from_obj"], to_obj=g["to_obj"],
            tof_days=round(g["tof_days"], 3),
            d_a_km=round(g["af"] - g["a0"], 4),
            d_incl_deg=round(math.degrees(g["inclf"] - g["incl0"]), 4),
            d_raan_deg=round(math.degrees(g["raan_gap"]), 4),
            edelbaum_m_s=eb, lu_nlp_m_s=nlp, lu_estimate_m_s=est,
            rel_diff_pct=rel))
    return rows


def summarise(rows):
    both = [r for r in rows if r["rel_diff_pct"] is not None]
    eb_only = [r for r in rows if r["edelbaum_m_s"] is not None and r["lu_nlp_m_s"] is None]
    lu_only = [r for r in rows if r["edelbaum_m_s"] is None and r["lu_nlp_m_s"] is not None]
    neither = [r for r in rows if r["edelbaum_m_s"] is None and r["lu_nlp_m_s"] is None]

    print(f"\nsampled geometries              : {len(rows)}")
    print(f"  both models found a transfer  : {len(both)}")
    print(f"  Edelbaum only                 : {len(eb_only)}")
    print(f"  Lu NLP only                   : {len(lu_only)}")
    print(f"  neither                       : {len(neither)}")
    if not both:
        print("\nno comparable geometries; nothing to report")
        return None

    diffs = np.array([r["rel_diff_pct"] for r in both])
    ad = np.abs(diffs)
    print(f"\nrelative difference of Edelbaum against the Lu NLP, over {len(both)} geometries")
    print(f"  mean |difference|             : {ad.mean():.2f} %")
    print(f"  median |difference|           : {np.median(ad):.2f} %")
    print(f"  worst |difference|            : {ad.max():.2f} %")
    print(f"  signed mean                   : {diffs.mean():+.2f} %  (positive = Edelbaum dearer)")

    print("\n  by inclination separation")
    edges = [0.0, 0.1, 0.5, 2.0, 10.0, 30.0, 90.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in both if lo <= abs(r["d_incl_deg"]) < hi]
        if sel:
            d = np.abs([r["rel_diff_pct"] for r in sel])
            print(f"    {lo:5.1f} - {hi:5.1f} deg : n = {len(sel):4d}   "
                  f"mean {d.mean():6.2f} %   max {d.max():6.2f} %")
    return dict(mean=float(ad.mean()), median=float(np.median(ad)),
                worst=float(ad.max()), n=len(both))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", default="outputs/instance_population.csv")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--tof-min", type=float, default=30.0)
    p.add_argument("--tof-max", type=float, default=365.0)
    p.add_argument("--thrust-mn", type=float, default=servicer.THRUST_N * 1e3,
                   help="servicer thrust [mN]; defaults to the vehicle of "
                        "Section 3.2.1, currently %(default)g")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/transfer_model_crossvalidation.csv")
    args = p.parse_args()

    thrust = args.thrust_mn * 1e-6      # mN -> kg*km/s^2
    objects = load_population(args.population)
    rng = np.random.default_rng(args.seed)
    geometries = sample_geometries(objects, args.n, (args.tof_min, args.tof_max), rng)

    print(f"population: {len(objects)} objects from {args.population}")
    print(f"servicer  : {MASS:.0f} kg, Isp {ISP:.0f} s, thrust {args.thrust_mn:g} mN "
          f"-> f = {thrust/MASS:.3e} km/s^2")
    print(f"times of flight: {args.tof_min:g} - {args.tof_max:g} days")

    rows = run(geometries, MASS, ISP, thrust)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summarise(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
