"""
plot_bcr.py — BCR vs propellant mass capacity (Xe Hall thruster, Isp = 2800 s).

Reads SE4 sensitivity results (dv_budget varied: 1500, 3000, 5000, 8000 m/s).
For each (algorithm, budget, trial), extracts the knee-point solution from the
Pareto front and computes BCR using the corrected asset-value assumptions.

Assumptions
-----------
Propellant   : Xenon, Isp = 2800 s, cost = $340/kg (Tirila et al. 2023)
F9 launch    : 22,000 kg capacity, $74 M  → market rate = $74M/22000 = $3,363.64/kg
Internal rate: market rate / 2 = $1,681.82/kg  (SpaceX internal Starlink launch)
Vehicle cost : $50 M per servicer (fixed)
Dry mass     : 300 kg
V1 satellite : $700 K manufacturing + 260 kg × internal rate = $1.137 M
V2 satellite : $1.77 M manufacturing + 800 kg × internal rate = $3.116 M
V2 fraction  : 30% of fleet
Average asset value: 0.70 × 1.137 M + 0.30 × 3.116 M = $1.731 M
Total asset value (200 demands): $346.2 M  (approximate; actual varies per trial)

BCR formula
-----------
recovered_value   = TOTAL_ASSET_V − f2
rideshare_revenue = max(0, F9_CAPACITY − f3 × m_wet) × MARKET_RATE
benefits          = recovered_value + rideshare_revenue

n_sorties_total   = f1 / dv_budget          (total ΔV / per-sortie budget)
c_launch          = f3 × m_wet × MARKET_RATE
c_propellant      = n_sorties_total × m_prop × XE_COST
c_vehicle         = f3 × C_VEHICLE
costs             = c_launch + c_propellant + c_vehicle

BCR = benefits / costs

Output: outputs/bcr_vs_propellant.pdf  — 3 panels (one per algorithm)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

G0            = 9.80665                  # m/s²
ISP_XE        = 2800.0                   # s
VE_XE         = ISP_XE * G0             # m/s
XE_COST       = 340.0                    # $/kg
F9_COST       = 74_000_000.0            # $
F9_CAPACITY   = 22_000.0                # kg
MARKET_RATE   = F9_COST / F9_CAPACITY   # $/kg = 3363.64
INTERNAL_RATE = MARKET_RATE / 2.0       # $/kg = 1681.82
C_VEHICLE     = 50_000_000.0            # $ per servicer
M_DRY         = 300.0                   # kg

V1_MFG        = 700_000.0               # $
V1_MASS       = 260.0                   # kg
V2_MFG        = 1_770_000.0             # $
V2_MASS       = 800.0                   # kg
V2_FRACTION   = 0.30

V1_VALUE      = V1_MFG + INTERNAL_RATE * V1_MASS   # ~$1.137 M
V2_VALUE      = V2_MFG + INTERNAL_RATE * V2_MASS   # ~$3.116 M
AVG_V_SAT     = (1 - V2_FRACTION) * V1_VALUE + V2_FRACTION * V2_VALUE
N_DEMANDS     = 200
TOTAL_ASSET_V = N_DEMANDS * AVG_V_SAT

PENALTY       = 1e6

DV_BUDGETS    = [1500, 3000, 5000, 8000]   # m/s — SE4 levels
ALG_KEYS      = ["mdls", "nsga3", "pso"]
ALG_LABELS    = {"mdls": "MDLS", "nsga3": "NSGA-III", "pso": "MOPSO-CD"}
COLOURS       = {"mdls": "#1f77b4", "nsga3": "#ff7f0e", "pso": "#d62728"}
N_TRIALS      = 5

RES_DIR = os.path.join("outputs", "sensitivity_results")
OUT_DIR = "outputs"

# ── Solution selectors ────────────────────────────────────────────────────────

def knee_point(df):
    """Row closest to ideal [0,0,0] in within-front normalised space."""
    cols = ["f1_dv", "f2_unrecovered_value", "f3_vehicles"]
    pts  = df[cols].values.astype(float)
    lo   = pts.min(axis=0)
    hi   = pts.max(axis=0)
    rng  = np.maximum(hi - lo, 1e-10)
    norm = (pts - lo) / rng
    return df.iloc[np.argmin((norm ** 2).sum(axis=1))]

def max_coverage(df):
    """Row with minimum unrecovered asset value (maximum satellites serviced)."""
    return df.loc[df["f2_unrecovered_value"].idxmin()]

# ── Collect BCR per (algorithm, budget, trial, selector) ─────────────────────

def collect_records(selector):
    records = []
    for alg in ALG_KEYS:
        for dv_budget in DV_BUDGETS:
            m_prop = M_DRY * (math.exp(dv_budget / VE_XE) - 1.0)
            m_wet  = M_DRY + m_prop

            for trial in range(1, N_TRIALS + 1):
                path = os.path.join(RES_DIR,
                                    f"{alg}_dvbudget_{dv_budget}_{trial:02d}.csv")
                if not os.path.exists(path):
                    print(f"  [skip] missing: {path}")
                    continue

                df = pd.read_csv(path)
                if df.empty:
                    continue

                if "f2_unrecovered_value" not in df.columns and "f2_unassigned_time" in df.columns:
                    df = df.rename(columns={"f2_unassigned_time": "f2_unrecovered_value"})

                df = df[df["f1_dv"] < PENALTY].copy()
                if df.empty:
                    continue

                kp = selector(df)
                f1 = float(kp["f1_dv"])
                f2 = float(kp["f2_unrecovered_value"])
                f3 = max(float(kp["f3_vehicles"]), 1.0)

                recovered_value = max(0.0, TOTAL_ASSET_V - f2)
                benefits        = recovered_value

                n_sorties_total = f1 / dv_budget if dv_budget > 0 else 0.0
                c_launch        = f3 * m_wet * MARKET_RATE
                c_propellant    = n_sorties_total * m_prop * XE_COST
                c_vehicle       = f3 * C_VEHICLE
                costs           = c_launch + c_propellant + c_vehicle

                bcr = benefits / costs if costs > 0 else np.nan

                records.append({
                    "algorithm":       alg,
                    "dv_budget":       dv_budget,
                    "m_prop":          m_prop,
                    "trial":           trial,
                    "bcr":             bcr,
                    "recovered_value": recovered_value,
                })
    return pd.DataFrame(records)

data_knee   = collect_records(knee_point)
data_maxcov = collect_records(max_coverage)

for label, df in [("knee", data_knee), ("maxcov", data_maxcov)]:
    if df.empty:
        print(f"[error] No data for {label} — check outputs/sensitivity_results/ for dvbudget CSVs")
        exit(1)

# ── Plot helpers ──────────────────────────────────────────────────────────────

FS       = 13
FS_TICK  = 11
FS_TITLE = 14

def collect_stats(sub, column):
    x_vals, med_vals, lo_vals, hi_vals = [], [], [], []
    for dv_budget in DV_BUDGETS:
        grp = sub[sub["dv_budget"] == dv_budget][column].dropna()
        if len(grp) == 0:
            continue
        m_prop = float(sub[sub["dv_budget"] == dv_budget]["m_prop"].iloc[0])
        x_vals.append(m_prop)
        med_vals.append(float(np.median(grp)))
        lo_vals.append(float(np.percentile(grp, 25)))
        hi_vals.append(float(np.percentile(grp, 75)))
    return (np.array(x_vals), np.array(med_vals),
            np.array(lo_vals), np.array(hi_vals))

def plot_panels(data, out_path, title_suffix):
    col = COLOURS["mdls"]
    sub = data[data["algorithm"] == "mdls"]

    fig, (ax_bcr, ax_rec) = plt.subplots(2, 1, figsize=(6, 9))

    # BCR panel
    x, med, lo, hi = collect_stats(sub.dropna(subset=["bcr"]), "bcr")
    ax_bcr.fill_between(x, lo, hi, color=col, alpha=0.20)
    ax_bcr.plot(x, med, color=col, lw=2, marker="o", ms=5)
    ax_bcr.axhline(1.0, color="black", lw=1.2, ls="--", label="BCR = 1")
    ax_bcr.set_title(f"Propellant Budget vs BCR ({title_suffix})", fontsize=FS_TITLE)
    ax_bcr.set_xlabel("Propellant Capacity [kg]", fontsize=FS)
    ax_bcr.set_ylabel("BCR", fontsize=FS)
    ax_bcr.tick_params(labelsize=FS_TICK)
    ax_bcr.grid(True, linewidth=0.4, alpha=0.5)
    ax_bcr.set_ylim(bottom=0)
    ax_bcr.legend(fontsize=FS_TICK)

    # Recovered value panel
    x, med, lo, hi = collect_stats(sub.dropna(subset=["recovered_value"]), "recovered_value")
    ax_rec.fill_between(x, lo / 1e6, hi / 1e6, color=col, alpha=0.20)
    ax_rec.plot(x, med / 1e6, color=col, lw=2, marker="o", ms=5)
    ax_rec.set_title(f"Propellant Budget vs Recovered Value ({title_suffix})", fontsize=FS_TITLE)
    ax_rec.set_xlabel("Propellant Capacity [kg]", fontsize=FS)
    ax_rec.set_ylabel("Recovered Asset Value [$M]", fontsize=FS)
    ax_rec.tick_params(labelsize=FS_TICK)
    ax_rec.grid(True, linewidth=0.4, alpha=0.5)
    ax_rec.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

# ── Generate both PDFs ────────────────────────────────────────────────────────

plot_panels(data_knee,   os.path.join(OUT_DIR, "bcr_vs_propellant_knee.pdf"),        "Knee Point")
plot_panels(data_maxcov, os.path.join(OUT_DIR, "bcr_vs_propellant_maxcoverage.pdf"), "Max Coverage")

print(f"\nConstants used:")
print(f"  V1 value = ${V1_VALUE/1e6:.3f}M,  V2 value = ${V2_VALUE/1e6:.3f}M")
print(f"  avg_V_sat = ${AVG_V_SAT/1e6:.3f}M,  TOTAL_ASSET_V = ${TOTAL_ASSET_V/1e6:.1f}M")
print(f"  MARKET_RATE = ${MARKET_RATE:.2f}/kg,  INTERNAL_RATE = ${INTERNAL_RATE:.2f}/kg")
