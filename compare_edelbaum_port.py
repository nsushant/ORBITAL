"""Place the Python Edelbaum port and the Julia original against a common truth.

This began as a numerical gate on the port, with the Julia as the reference.
That framing turned out to be wrong: the Julia is the less accurate of the two,
by a wide margin, so agreeing with it would have been evidence of a defect
rather than of fidelity. What the script does now is measure both against a
brute-force evaluation of the same model, and report what each gets wrong.

Why the Julia is coarse. `cost/edelbaum_transfer.jl` searches 20 rings spaced
geometrically from 0.1 km/s of growth to 8, samples each with 30 points, and
declares the node closed when the residual falls inside a tolerance of 1e-2 rad,
about 0.57 degrees. Three consequences follow. Its ladder has no rung below
0.1 km/s, which is 200 m/s of delta-V, so any transfer whose true growth is
smaller is quoted at the floor. Thirty samples miss narrow closure bands. And a
tolerance accepts near-misses that do not actually close while rejecting exact
closures that fall between samples.

What the port does instead: a ladder from 1e-4 km/s with a thousand-plus rungs,
180 samples, and closure solved for by root-finding rather than tested against a
tolerance.

On what grounds the port is trusted, then, if not agreement with the Julia:

  * the physics, by closed-form results in tests/test_edelbaum.py -- the
    Edelbaum relation itself to 2.4e-13, a pure plane change against
    2V sin(pi dI/4), the rocket equation, the sun-synchronous drift rate, and
    the ring identity that makes the search valid;
  * the resolution, by convergence -- the production settings are compared here
    against a deliberately over-resolved evaluation of the same model.

Agreement with an independent implementation would have been worth more than
either. This comparison cannot supply it, and the script says so rather than
implying otherwise.

Run `julia --project=. reference_edelbaum.jl` first, on a machine with Julia.
"""

import csv
import math
import os
import sys

import numpy as np

from oos import edelbaum as eb

REFERENCE = "outputs/edelbaum_reference_julia.csv"

# The vehicle reference_edelbaum.jl used. Arbitrary for an equivalence test, but
# both sides must agree, so it is not read from oos.servicer.
MASS, ISP, THRUST = 335.0, 2800.0, 1e-4

# Deliberately over-resolved: twice the samples, more than twice the rungs, and
# a refinement tolerance three orders tighter than production.
TRUTH = dict(n_scan=360, n_growth=3000, growth_rtol=1e-6)

JULIA_SENTINEL = 1e6      # the Julia writes 2e7 for "no transfer"


def nodal_rate(a, incl):
    return -1.5 * eb.J2 * (eb.R_E / a) ** 2 * math.sqrt(eb.MU / a**3) * math.cos(incl)


def summarise(name, values, truth, n_total):
    paired = [(v, t) for v, t in zip(values, truth) if v is not None and t is not None]
    found = sum(v is not None for v in values)
    missed = sum(1 for v, t in zip(values, truth) if v is None and t is not None)
    invented = sum(1 for v, t in zip(values, truth) if v is not None and t is None)
    print(f"\n--- {name}")
    print(f"  reports a transfer      : {found} / {n_total}")
    print(f"  misses one truth finds  : {missed}")
    print(f"  finds one truth does not: {invented}")
    if not paired:
        print("  no comparable geometries")
        return
    err = np.array([(v - t) / t for v, t in paired])
    print(f"  delta-V against truth, over {len(paired)} geometries")
    print(f"    median                : {np.median(err) * 100:+8.2f} %")
    print(f"    mean                  : {err.mean() * 100:+8.2f} %")
    print(f"    worst                 : {np.abs(err).max() * 100:8.2f} %")
    print(f"    too expensive         : {100 * (err > 1e-9).mean():8.1f} % of them")


def main():
    if not os.path.exists(REFERENCE):
        raise SystemExit(f"{REFERENCE} not found. Run:\n"
                         "    julia --project=. reference_edelbaum.jl")
    rows = list(csv.DictReader(open(REFERENCE)))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
    rows = rows[:limit]
    print(f"reference geometries: {len(rows)}")
    print(f"servicer: {MASS:.0f} kg, Isp {ISP:.0f} s, thrust {THRUST * 1e6:.0f} mN")

    python, julia, truth = [], [], []
    for r in rows:
        a0, i0 = float(r["a0_km"]), float(r["incl0_rad"])
        af, i_f = float(r["af_km"]), float(r["inclf_rad"])
        tof_s = float(r["tof_days"]) * 86400.0
        # The Julia is handed the required *absolute* nodal change, so the
        # target's own drift over the horizon is folded in, as it is there.
        required = float(r["raan_gap_rad"]) + nodal_rate(af, i_f) * tof_s

        s1 = np.array([a0, i0, 0.0, MASS, ISP, THRUST])
        s2 = np.array([af, i_f, required, MASS, ISP, THRUST])
        tofs = np.array([tof_s])

        prod = eb.transfer_cost(s1, s2, tofs)[0]
        ref = eb.transfer_cost(s1, s2, tofs, eb.EMPTY, TRUTH["n_scan"],
                               TRUTH["n_growth"], eb.MAX_GROWTH,
                               TRUTH["growth_rtol"])[0]
        j = float(r["dv_m_s"])

        python.append(None if np.isnan(prod) else prod * 1000.0)
        truth.append(None if np.isnan(ref) else ref * 1000.0)
        julia.append(None if j >= JULIA_SENTINEL else j)

    n = len(rows)
    print(f"\ntruth (over-resolved: {TRUTH['n_scan']} samples, {TRUTH['n_growth']} "
          f"rings, {TRUTH['growth_rtol']:g} refinement)")
    print(f"  reports a transfer      : {sum(v is not None for v in truth)} / {n}")

    summarise("Python, production settings", python, truth, n)
    summarise("Julia, cost/edelbaum_transfer.jl", julia, truth, n)

    print("\nRead this as a statement about the Julia, not only about the port. "
          "\nEvery cost table built with cost/edelbaum_transfer.jl inherits the "
          "\nerrors in its row.")


if __name__ == "__main__":
    main()
