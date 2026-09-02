"""
Numerical gate on the Python Edelbaum port (docs/edelbaum_port_audit.md).

Diffs oos/edelbaum.py against the delta-V dumped by reference_edelbaum.jl on the
same geometries. Run the Julia script first, on a machine that has Julia.

The Julia model uses a different constant set - notably Earth's mean radius
rather than its equatorial radius in the J2 term - so the comparison is reported
twice: against the Python model on the standard constants, which is the
difference that matters for the paper, and against the Python model forced onto
the Julia constants, which isolates everything that is not the constant set.
"""

import csv
import importlib
import os
import shutil
import sys
import tempfile

import numpy as np

REFERENCE = "outputs/edelbaum_reference_julia.csv"
JULIA_CONSTANTS = '''J2 = 1.0826e-3
R_E = 6371.0
MU = 398600.44
G0 = 0.00981
DAY = 86400.0
K_J2 = 3.0 * J2 * R_E**2 / (2.0 * MU**3)
'''


def load_model(package_root):
    sys.path.insert(0, package_root)
    for name in [m for m in sys.modules if m.startswith("oos")]:
        del sys.modules[name]
    module = importlib.import_module("oos.edelbaum")
    sys.path.pop(0)
    return module


def compare(model, rows, label):
    ours, theirs = [], []
    only_julia = only_python = 0
    for r in rows:
        dv = model.transfer_dv(r["a0_km"], r["incl0_rad"], 0.0, r["af_km"],
                               r["inclf_rad"],
                               r["raan_gap_rad"] + _drift(model, r) * r["tof_days"] * 86400.0,
                               r["tof_days"], mass=335.0, isp=2800.0, thrust=1e-4)
        julia = r["dv_m_s"]
        julia_feasible = julia < 1e6          # the Julia sentinel is 2e7
        if dv is None and julia_feasible:
            only_julia += 1
        elif dv is not None and not julia_feasible:
            only_python += 1
        elif dv is not None and julia_feasible:
            ours.append(dv)
            theirs.append(julia)

    print(f"\n--- {label}")
    print(f"  both feasible          : {len(ours)} / {len(rows)}")
    print(f"  Julia only             : {only_julia}")
    print(f"  Python only            : {only_python}")
    if not ours:
        print("  nothing to compare")
        return
    a, b = np.array(ours), np.array(theirs)
    rel = 100.0 * (a - b) / b
    print(f"  mean |difference|      : {np.abs(rel).mean():.3f} %")
    print(f"  median |difference|    : {np.median(np.abs(rel)):.3f} %")
    print(f"  max |difference|       : {np.abs(rel).max():.3f} %")
    print(f"  signed mean            : {rel.mean():+.3f} %")


def _drift(model, r):
    return model.j2_raan_rate(r["af_km"], r["inclf_rad"])


def main():
    if not os.path.exists(REFERENCE):
        raise SystemExit(f"{REFERENCE} not found - run: julia --project=. reference_edelbaum.jl")
    rows = []
    with open(REFERENCE) as fh:
        for row in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in row.items()})
    print(f"reference geometries: {len(rows)}")

    here = os.path.dirname(os.path.abspath(__file__))
    compare(load_model(here), rows, "Python on the standard constants (what the paper uses)")

    tmp = tempfile.mkdtemp()
    shutil.copytree(os.path.join(here, "oos"), os.path.join(tmp, "oos"))
    with open(os.path.join(tmp, "oos", "constants.py"), "w") as fh:
        fh.write(JULIA_CONSTANTS)
    compare(load_model(tmp), rows, "Python forced onto the Julia constants (isolates the algorithm)")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
