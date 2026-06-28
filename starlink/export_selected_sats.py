"""
starlink/export_selected_sats.py — Match sampled satellites back to Celestrak catalog.

For each sat_1..sat_N in outputs/simulation.h5, finds the closest matching entry
in starlink/starlink_catalog.json by (inclination, RAAN) and outputs a LaTeX
table of OBJECT_NAME, NORAD_CAT_ID, COSPAR ID, shell, inclination, altitude, RAAN.

Run: python starlink/export_selected_sats.py
"""

import json, math, os
import h5py
import numpy as np

SIM_H5      = os.path.join("outputs", "simulation.h5")
CATALOG_JSON = os.path.join("starlink", "starlink_catalog.json")
RE          = 6371.0   # km

MU = 398600.4418       # km³/s²

def mm_to_sma(mm_rev_per_day):
    n = mm_rev_per_day * 2 * math.pi / 86400.0
    return (MU / n**2) ** (1/3)

# ── Load catalog ──────────────────────────────────────────────────────────────

with open(CATALOG_JSON) as f:
    catalog = json.load(f)

cat_inc  = np.array([e["INCLINATION"]    for e in catalog])
cat_raan = np.array([e["RA_OF_ASC_NODE"] for e in catalog])
cat_sma  = np.array([mm_to_sma(e["MEAN_MOTION"]) for e in catalog])

# ── Load simulation orbital elements at t=0 ───────────────────────────────────

records = []
with h5py.File(SIM_H5, "r") as f:
    names = [n.decode() if isinstance(n, bytes) else n
             for n in f["metadata"]["names"][()]]
    for name in names:
        if name.startswith("depot"):
            continue
        oe       = f[name]["orbital_elements"][0, :]
        sma      = float(oe[0])
        inc_deg  = math.degrees(float(oe[1]))
        raan_deg = math.degrees(float(oe[2])) % 360.0
        records.append(dict(name=name, sma=sma, inc=inc_deg, raan=raan_deg))

print(f"Satellites to match: {len(records)}")

# ── Match each sat to catalog entry ───────────────────────────────────────────
# Distance metric: angular separation in (inc, RAAN) space [degrees]

SHELL_LABELS = {
    1: "53.0°/550 km",
    2: "53.2°/540 km",
    3: "70.0°/570 km",
    4: "97.6°/560 km",
    5: "43.0°/530 km",
}
SHELLS = [
    (53.0, 550.0), (53.2, 540.0), (70.0, 570.0),
    (97.6, 560.0), (43.0, 530.0),
]
INC_TOL = 1.5

def assign_shell(inc_deg):
    for k, (si, _) in enumerate(SHELLS):
        if abs(inc_deg - si) <= INC_TOL:
            return k + 1
    return 0

matched = []
for rec in records:
    # Angular distance in (inc, RAAN) normalised by typical spread
    d_inc  = cat_inc  - rec["inc"]
    d_raan = cat_raan - rec["raan"]
    # Wrap RAAN difference to [-180, 180]
    d_raan = (d_raan + 180) % 360 - 180
    dist   = np.sqrt(d_inc**2 + d_raan**2)
    idx    = int(np.argmin(dist))
    e      = catalog[idx]
    alt    = cat_sma[idx] - RE
    shell  = assign_shell(e["INCLINATION"])
    matched.append(dict(
        sim_name   = rec["name"],
        obj_name   = e["OBJECT_NAME"],
        norad_id   = e["NORAD_CAT_ID"],
        cospar_id  = e.get("OBJECT_ID", "—"),
        inc        = e["INCLINATION"],
        raan       = e["RA_OF_ASC_NODE"],
        alt        = alt,
        shell      = shell,
        match_dist = float(dist[idx]),
    ))

# Warn on poor matches
poor = [m for m in matched if m["match_dist"] > 1.0]
if poor:
    print(f"Warning: {len(poor)} poor matches (angular dist > 1°):")
    for m in poor[:5]:
        print(f"  {m['sim_name']} → {m['obj_name']} dist={m['match_dist']:.3f}°")

# Sort by shell then NORAD ID
matched.sort(key=lambda m: (m["shell"], m["norad_id"]))

# ── Print LaTeX table ─────────────────────────────────────────────────────────

print()
print(r"\begin{longtable}{lllccc}")
print(r"\caption{Celestrak catalog references for the 200 Starlink satellites")
print(r"sampled in the case study. Data retrieved from the Celestrak GP catalog")
print(r"\citep{celestrak2024}.}")
print(r"\label{tab:starlink_catalog} \\")
print(r"\toprule")
print(r"Object name & NORAD ID & COSPAR ID & Shell & Inc (\textdegree) & Alt (km) \\")
print(r"\midrule")
print(r"\endfirsthead")
print(r"\multicolumn{6}{c}{\tablename\ \thetable{} -- continued} \\")
print(r"\toprule")
print(r"Object name & NORAD ID & COSPAR ID & Shell & Inc (\textdegree) & Alt (km) \\")
print(r"\midrule")
print(r"\endhead")
print(r"\midrule \multicolumn{6}{r}{Continued on next page} \\")
print(r"\endfoot")
print(r"\bottomrule")
print(r"\endlastfoot")

prev_shell = None
for m in matched:
    if m["shell"] != prev_shell:
        if prev_shell is not None:
            print(r"\midrule")
        print(r"\multicolumn{6}{l}{\textit{Shell " +
              SHELL_LABELS.get(m["shell"], str(m["shell"])) + r"}} \\")
        prev_shell = m["shell"]
    cospar = m["cospar_id"].replace("_", r"\_") if m["cospar_id"] else "—"
    print(f"{m['obj_name']} & {m['norad_id']} & {cospar} & "
          f"{m['shell']} & {m['inc']:.2f} & {m['alt']:.0f} \\\\")

print(r"\end{longtable}")
print(f"\n% Total matched: {len(matched)}")
