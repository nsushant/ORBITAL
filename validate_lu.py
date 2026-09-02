"""
Validation of the Lu et al. implementation against the published results.

Stage 1 checks the Section 4.2 coasting-orbit estimate against the paper's
Table 2. Stage 2 checks the Section 4.3 continuous-thrust NLP against the
lowest values of the paper's Tables 4, 6 and 8.

This establishes that our implementation of the reference method is correct,
which is the precondition for using it to validate the fast Edelbaum model.

Run:  python3 validate_lu.py
"""

import math

from oos.constants import DAY, R_E
from oos.lu import FMAX_DEFAULT, coasting_orbit_estimate, nlp_transfer_cost, raan_drift_rate

D = math.radians
TOF_DAYS = 100.0

# Test cases as defined in the paper. `a_c`/`incl_c` are the paper's published
# coasting orbits, used as the target for the Stage 2 NLP so that Stage 2 is
# tested independently of any error in Stage 1.
CASES = [
    dict(name="Case 1 (800->800 km, 98->98 deg, RAAN 0->+30)",
         alt0=800.0, incl0=98.0, altf=800.0, inclf=98.0, d_raan=30.0,
         a_c=6853.309, incl_c=99.318,
         paper_j=484.664, paper_jt=242.332, paper_ja=242.332, paper_nlp=492.193,
         seeds=[[D(134.698), D(20.666), 3.563, D(-44.706), D(4.318), 81.758],
                [D(135), D(8), 10, D(-45), D(7), 89]]),
    dict(name="Case 2 (800->900 km, 98->99 deg, RAAN 0->+30)",
         alt0=800.0, incl0=98.0, altf=900.0, inclf=99.0, d_raan=30.0,
         a_c=6873.711, incl_c=100.010,
         paper_j=550.915, paper_jt=304.428, paper_ja=246.487, paper_nlp=561.251,
         seeds=[[D(121.630), D(9.232), 9.999, D(-31.509), D(4.942), 84.881],
                [D(122), D(9), 10, D(-31), D(5), 85]]),
    dict(name="Case 3 (700->1200 km, 49.9->50 deg, RAAN 0->-80)",
         alt0=700.0, incl0=49.9, altf=1200.0, inclf=50.0, d_raan=-80.0,
         a_c=7147.626, incl_c=49.914,
         paper_j=251.9, paper_jt=36.2, paper_ja=215.7, paper_nlp=252.284,
         seeds=[[D(0.9514), D(21.044), 0.517, D(3.869), D(57.122), 98.876],
                [D(1), D(21), 0.5, D(4), D(57), 98.9]]),
]


def rel_err(got, want):
    return 100.0 * (got - want) / want


def stage1():
    print("\nStage 1 — Section 4.2 coasting-orbit estimate (paper Table 2)")
    print("=" * 78)
    rows = []
    for c in CASES:
        orbit = coasting_orbit_estimate(
            R_E + c["alt0"], D(c["incl0"]), 0.0,
            R_E + c["altf"], D(c["inclf"]), D(c["d_raan"]), TOF_DAYS)
        assert orbit is not None, f"{c['name']}: no admissible coasting orbit"

        j = orbit.j_total * 1000.0
        jt = orbit.jt * 1000.0
        ja = orbit.ja * 1000.0
        print(f"\n{c['name']}")
        print(f"  {'quantity':18s} {'computed':>12s} {'paper':>12s} {'error':>10s}")
        print(f"  {'J total [m/s]':18s} {j:12.3f} {c['paper_j']:12.3f} {j - c['paper_j']:10.3f}")
        print(f"  {'Jt [m/s]':18s} {jt:12.3f} {c['paper_jt']:12.3f} {jt - c['paper_jt']:10.3f}")
        print(f"  {'Ja [m/s]':18s} {ja:12.3f} {c['paper_ja']:12.3f} {ja - c['paper_ja']:10.3f}")
        print(f"  {'coast a [km]':18s} {orbit.a_c:12.3f} {c['a_c']:12.3f} {orbit.a_c - c['a_c']:10.3f}")
        print(f"  {'coast I [deg]':18s} {math.degrees(orbit.incl_c):12.3f} {c['incl_c']:12.3f} "
              f"{math.degrees(orbit.incl_c) - c['incl_c']:10.3f}")
        rows.append((c["name"], j, c["paper_j"], rel_err(j, c["paper_j"])))
    return rows


def stage2():
    print("\n\nStage 2 — Section 4.3 continuous-thrust NLP (paper Tables 4/6/8)")
    print("=" * 78)
    print(f"{'case':10s} {'computed [m/s]':>16s} {'paper [m/s]':>14s} {'error':>10s}")
    rows = []
    for n, c in enumerate(CASES, start=1):
        a0, af = R_E + c["alt0"], R_E + c["altf"]
        incl0, inclf = D(c["incl0"]), D(c["inclf"])
        # The target RAAN is propagated to the end of the horizon, matching the
        # convention used by the Section 4.2 estimate.
        raan_target = D(c["d_raan"]) + raan_drift_rate(af, inclf) * TOF_DAYS * DAY

        dv = nlp_transfer_cost(a0, incl0, af, inclf, c["a_c"], D(c["incl_c"]),
                               0.0, raan_target, TOF_DAYS, c["seeds"],
                               fmax=FMAX_DEFAULT)
        if dv is None:
            print(f"{n:<10d} {'NO CONVERGENCE':>16s} {c['paper_nlp']:14.3f} {'-':>10s}")
            rows.append((c["name"], None, c["paper_nlp"], None))
            continue
        e = rel_err(dv, c["paper_nlp"])
        print(f"{n:<10d} {dv:16.3f} {c['paper_nlp']:14.3f} {e:9.2f}%")
        rows.append((c["name"], dv, c["paper_nlp"], e))
    return rows


if __name__ == "__main__":
    s1 = stage1()
    s2 = stage2()
    worst1 = max(abs(r[3]) for r in s1)
    ok2 = [r for r in s2 if r[3] is not None]
    worst2 = max(abs(r[3]) for r in ok2) if ok2 else float("nan")
    print(f"\nWorst relative error — Stage 1: {worst1:.2f} %   Stage 2: {worst2:.2f} % "
          f"({len(ok2)}/{len(s2)} converged)")
