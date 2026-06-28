"""
starlink/summarise_constellation.py — Sampled constellation summary table.

Reads outputs/simulation.h5, extracts initial orbital elements for all sat_*
nodes (excluding depot), bins satellites into shells by inclination, and prints
an Acta Astronautica–compliant LaTeX table.

Columns per shell:
  Inc (°) | Alt (km) | SMA (km) | N selected | RAAN range (°) | N planes covered

Run: python starlink/summarise_constellation.py
"""

import math, os
import h5py
import numpy as np

SIM_H5 = os.path.join("outputs", "simulation.h5")
RE     = 6371.0    # km — Earth mean radius

# Shell definitions matching fetch_and_sample.jl
SHELLS = [
    dict(name="Shell 1", inc=53.0,  alt=550.0),
    dict(name="Shell 2", inc=53.2,  alt=540.0),
    dict(name="Shell 3", inc=70.0,  alt=570.0),
    dict(name="Shell 4", inc=97.6,  alt=560.0),
    dict(name="Shell 5", inc=43.0,  alt=530.0),
]
INC_TOL = 1.5    # degrees — same tolerance as fetch_and_sample.jl

def assign_shell(inc_deg):
    for k, sh in enumerate(SHELLS):
        if abs(inc_deg - sh["inc"]) <= INC_TOL:
            return k
    return -1   # unrecognised

# ── Read initial orbital elements ─────────────────────────────────────────────

records = []   # list of (name, sma_km, inc_deg, raan_deg)

with h5py.File(SIM_H5, "r") as f:
    names = [n.decode() if isinstance(n, bytes) else n
             for n in f["metadata"]["names"][()]]
    for name in names:
        if name.startswith("depot"):
            continue
        oe = f[name]["orbital_elements"][0, :]   # t=0 row: [a, i, RAAN, nu]
        sma      = float(oe[0])                  # km
        inc_deg  = math.degrees(float(oe[1]))
        raan_deg = math.degrees(float(oe[2])) % 360.0
        records.append((name, sma, inc_deg, raan_deg))

print(f"Total satellites read: {len(records)}")

# ── Bin into shells ───────────────────────────────────────────────────────────

from collections import defaultdict
bins = defaultdict(list)   # shell_idx → list of (sma, inc_deg, raan_deg)

unassigned = 0
for name, sma, inc_deg, raan_deg in records:
    k = assign_shell(inc_deg)
    if k < 0:
        unassigned += 1
    else:
        bins[k].append((sma, inc_deg, raan_deg))

if unassigned:
    print(f"Warning: {unassigned} satellites not assigned to any shell")

# ── Compute per-shell statistics ──────────────────────────────────────────────

rows = []
for k, sh in enumerate(SHELLS):
    entries = bins[k]
    if not entries:
        rows.append(dict(shell=sh["name"], inc=sh["inc"], alt=sh["alt"],
                         sma=RE+sh["alt"], n=0,
                         raan_min=float("nan"), raan_max=float("nan"),
                         n_planes=0))
        continue
    smas      = [e[0] for e in entries]
    incs      = [e[1] for e in entries]
    raans     = sorted(e[2] for e in entries)
    mean_sma  = np.mean(smas)
    mean_inc  = np.mean(incs)
    mean_alt  = mean_sma - RE

    # Count distinct 5°-RAAN bins (mirrors fetch_and_sample.jl plane binning)
    plane_bins = set(int(r // 5) for r in raans)

    rows.append(dict(
        shell    = sh["name"],
        inc      = mean_inc,
        alt      = mean_alt,
        sma      = mean_sma,
        n        = len(entries),
        raan_min = raans[0],
        raan_max = raans[-1],
        n_planes = len(plane_bins),
    ))

# ── Print LaTeX table ─────────────────────────────────────────────────────────

print()
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Sampled Starlink constellation: orbital shell statistics. "
      r"Inclination and altitude are mean values of the sampled subset. "
      r"RAAN range and plane count use 5\textdegree{} bins.}")
print(r"\label{tab:starlink_shells}")
print(r"\begin{tabular}{lccccccc}")
print(r"\toprule")
print(r"Shell & Inc (\textdegree) & Alt (km) & SMA (km) & "
      r"$N$ & RAAN$_\mathrm{min}$ (\textdegree) & "
      r"RAAN$_\mathrm{max}$ (\textdegree) & Planes \\")
print(r"\midrule")

total_n = 0
for r in rows:
    if r["n"] == 0:
        print(f"{r['shell']} & {r['inc']:.1f} & {r['alt']:.0f} & "
              f"{r['sma']:.0f} & 0 & -- & -- & -- \\\\")
    else:
        print(f"{r['shell']} & {r['inc']:.1f} & {r['alt']:.0f} & "
              f"{r['sma']:.0f} & {r['n']} & "
              f"{r['raan_min']:.1f} & {r['raan_max']:.1f} & "
              f"{r['n_planes']} \\\\")
    total_n += r["n"]

print(r"\midrule")
print(f"Total & & & & {total_n} & & & \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
