"""
plot_propulsion_trade.py — BCR vs propellant mass for Xe Hall thruster (20 kW, Isp=2800 s).

Assumptions:
  - Propellant : Xenon, Isp = 2800 s, cost = $340/kg (Tirila et al. 2023)
  - Launch cost : m_wet × $3700/kg (rideshare rate, no spare-capacity revenue)
  - Vehicle cost: $50M per servicer (fixed)
  - Dry mass    : 300 kg
  - DV_BUDGET   : 5000 m/s per sortie

BCR = value_recovered / (c_launch + c_prop + c_vehicle)

Output: outputs/architecture_bcr_vs_propellant.pdf
  2×2 grid, one panel per algorithm.
  x-axis: propellant per vehicle per sortie [kg]
  y-axis: BCR

Run: python plot_propulsion_trade.py
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import binned_statistic

OUT       = "outputs"
FRONT_CSV = os.path.join(OUT, "starlink_pareto_fronts.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
G0            = 9.80665
ISP_XE        = 2800.0        # s  — 20 kW Hall thruster
VE_XE         = ISP_XE * G0   # m/s exhaust velocity
XE_COST_KG    = 340.0         # $/kg  (Tirila et al. 2023 Table 2)
RIDESHARE_RATE = 3700.0        # $/kg  launch cost
C_VEHICLE     = 50_000_000.0  # $ per servicer
M_DRY         = 300.0         # kg
DV_BUDGET     = 5000.0        # m/s per sortie
N_DEMANDS     = 200

ALG_NAMES = ["MDLS", "NSGA-III", "MOPSO-CD"]
COLOURS   = {"MDLS": "#1f77b4", "NSGA-III": "#ff7f0e",
             "MOPSO-CD": "#d62728"}
FS        = 13
FS_TICK   = 11
FS_TITLE  = 14

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(FRONT_CSV):
    print(f"[error] {FRONT_CSV} not found — run starlink/run_starlink.jl first")
    exit(1)

pf = pd.read_csv(FRONT_CSV)

# backwards-compat column name
if "f2_unrecovered_value" not in pf.columns and "f2_unassigned_time" in pf.columns:
    pf = pf.rename(columns={"f2_unassigned_time": "f2_unrecovered_value"})

if "value_recovered" not in pf.columns:
    print("[error] 'value_recovered' column missing — re-run starlink/run_starlink.jl")
    exit(1)

# ── BCR computation ───────────────────────────────────────────────────────────

def compute_bcr(row):
    f1   = float(row["f1_dv"])
    f3   = max(float(row["f3_vehicles"]), 1.0)
    vr   = float(row["value_recovered"])

    dv_per_veh   = f1 / f3
    n_sorties    = max(1, math.ceil(dv_per_veh / DV_BUDGET))
    dv_per_sort  = dv_per_veh / n_sorties
    m_prop       = M_DRY * (math.exp(dv_per_sort / VE_XE) - 1.0)
    m_wet        = M_DRY + m_prop

    c_launch  = f3 * m_wet * RIDESHARE_RATE
    c_prop    = f3 * n_sorties * m_prop * XE_COST_KG
    c_vehicle = f3 * C_VEHICLE
    cost      = c_launch + c_prop + c_vehicle

    bcr      = vr / cost if cost > 0 else np.nan
    return bcr, m_prop

pf["bcr"], pf["m_prop"] = zip(*pf.apply(compute_bcr, axis=1))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False, sharey=False)
fig.suptitle(
    f"BCR vs Propellant per Vehicle per Sortie\n"
    f"Xe Hall thruster  |  Isp = {int(ISP_XE)} s  |  "
    f"dry mass = {int(M_DRY)} kg  |  vehicle cost = \${int(C_VEHICLE/1e6)}M  |  "
    f"launch = \${int(RIDESHARE_RATE)}/kg",
    fontsize=FS)

N_BINS = 20

for ax, alg in zip(axes.flat, ALG_NAMES):
    sub = pf[pf["algorithm"] == alg].dropna(subset=["bcr", "m_prop"])
    col = COLOURS[alg]

    # scatter per trial
    for _, tgrp in sub.groupby("trial"):
        ax.scatter(tgrp["m_prop"], tgrp["bcr"],
                   color=col, alpha=0.25, s=12, linewidths=0)

    # median trend
    if len(sub) >= N_BINS:
        med, edges, _ = binned_statistic(
            sub["m_prop"], sub["bcr"], statistic="median", bins=N_BINS)
        cen  = 0.5 * (edges[:-1] + edges[1:])
        mask = ~np.isnan(med)
        ax.plot(cen[mask], med[mask], color=col, lw=2)

    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="BCR = 1")
    ax.set_title(alg, fontsize=FS_TITLE)
    ax.set_xlabel("Propellant per vehicle per sortie [kg]", fontsize=FS)
    ax.set_ylabel("BCR", fontsize=FS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_ylim(bottom=0)

fig.tight_layout()
out_path = os.path.join(OUT, "architecture_bcr_vs_propellant.pdf")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
