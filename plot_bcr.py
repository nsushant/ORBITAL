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
V1.5 satellite : $700 K manufacturing + 303 kg × internal rate = $1.210 M
V2 Mini satellite : $1.77 M manufacturing + 800 kg × internal rate = $3.116 M
V2 fraction  : 30% of fleet
Average asset value: 0.70 × 1.210 M + 0.30 × 3.116 M = $1.782 M
Total asset value (200 demands): $356.4 M  (approximate; actual varies per trial)

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
import h5py
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

V1_MASS       = 303.0   # kg (V1.5 DAS-filed mass)
V2_MASS       = 800.0   # kg (V2 Mini DAS-filed mass)
V2_FRACTION   = 0.30

# ($1K/kg mfg + $1,850/kg launch) × mass
V1_VALUE      = (1_000.0 + 1_850.0) * V1_MASS   # $864,150 base, pre-depreciation
V2_VALUE      = (1_000.0 + 1_850.0) * V2_MASS   # $2,280,000 base, pre-depreciation

# Apply Weibull depreciation at avg fleet age of 2.0 years (k=1.5, λ=5 yr)
import math as _math
_LAMBDA, _K, _AVG_AGE = 5.0, 1.5, 2.0
_DEPREC       = _math.exp(-(_AVG_AGE / _LAMBDA) ** _K)   # ~0.857
V1_VALUE_DEP  = V1_VALUE * _DEPREC     # ~$740K
V2_VALUE_DEP  = V2_VALUE * _DEPREC     # ~$1.95M
AVG_V_SAT     = (1 - V2_FRACTION) * V1_VALUE_DEP + V2_FRACTION * V2_VALUE_DEP
N_DEMANDS     = 200
TOTAL_ASSET_V = N_DEMANDS * AVG_V_SAT

PENALTY       = 1e6

DV_BUDGETS    = [1500, 3000, 5000, 8000]   # m/s — SE4 levels (10000 has no demand file)
ALG_KEYS      = ["mdls", "nsga3", "pso"]
ALG_LABELS    = {"mdls": "MDLS", "nsga3": "NSGA-III", "pso": "MOPSO-CD"}
COLOURS       = {"mdls": "#1f77b4", "nsga3": "#ff7f0e", "pso": "#d62728"}
N_TRIALS      = 5

RES_DIR = os.path.join("outputs", "sensitivity_results")
DEM_DIR = os.path.join("outputs", "sensitivity_demands")
OUT_DIR = "outputs"

# ── Demand loader (asset values per trial) ────────────────────────────────────

def load_asset_values(dv_budget, trial):
    """Return (total_asset_v, avg_asset_v, n_demands) from the JLD2 demand file."""
    path = os.path.join(DEM_DIR, f"bcr_dvbudget_{dv_budget}_{trial:02d}.jld2")
    if not os.path.exists(path):
        return TOTAL_ASSET_V, AVG_V_SAT, N_DEMANDS   # fallback to constants
    try:
        with h5py.File(path, "r") as f:
            top_ref = f["demands"][()][0]
            kvvec   = f[top_ref][()]
            for ref in kvvec:
                pair = f[ref][()]
                key  = pair[0].decode()
                if key == "asset_values":
                    vals = f[pair[1]][()].astype(float)
                    return float(vals.sum()), float(vals.mean()), int(len(vals))
    except Exception:
        pass
    return TOTAL_ASSET_V, AVG_V_SAT, N_DEMANDS

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

                total_v, avg_v, n_dem = load_asset_values(dv_budget, trial)

                kp = selector(df)
                f1 = float(kp["f1_dv"])
                f2 = float(kp["f2_unrecovered_value"])
                f3 = max(float(kp["f3_vehicles"]), 1.0)

                recovered_value = max(0.0, total_v - f2)
                n_lost          = min(n_dem, round(f2 / avg_v)) if avg_v > 0 else 0
                n_served        = n_dem - n_lost
                benefits        = recovered_value

                n_sorties_total = f1 / dv_budget if dv_budget > 0 else 0.0
                c_launch        = f3 * m_wet * MARKET_RATE
                c_propellant    = n_sorties_total * m_prop * XE_COST
                c_vehicle       = f3 * C_VEHICLE
                costs           = c_launch + c_propellant + c_vehicle

                bcr = benefits / costs if costs > 0 else np.nan

                # Max affordable manufacturing cost per vehicle for BCR = 1
                affordable_mfg = (benefits - c_launch - c_propellant) / f3

                records.append({
                    "algorithm":        alg,
                    "dv_budget":        dv_budget,
                    "m_prop":           m_prop,
                    "trial":            trial,
                    "bcr":              bcr,
                    "recovered_value":  recovered_value,
                    "n_served":         n_served,
                    "n_lost":           n_lost,
                    "total_asset_v_M":  total_v / 1e6,
                    "affordable_mfg_M": affordable_mfg / 1e6,
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

    fig, (ax_bcr, ax_rec, ax_sat) = plt.subplots(3, 1, figsize=(6, 13))

    # BCR panel
    x, med, lo, hi = collect_stats(sub.dropna(subset=["bcr"]), "bcr")
    ax_bcr.fill_between(x, lo, hi, color=col, alpha=0.20)
    ax_bcr.plot(x, med, color=col, lw=2, marker="o", ms=5)
    ax_bcr.axhline(1.0, color="black", lw=1.2, ls="--", label="BCR = 1")
    ax_bcr.set_title(f"Propellant Budget vs BCR ({title_suffix})", fontsize=FS_TITLE)
    ax_bcr.set_xlabel("Propellant Capacity [kg]", fontsize=FS)
    ax_bcr.set_ylabel("BCR", fontsize=FS)
    ax_bcr.tick_params(labelsize=FS_TICK)
    ax_bcr.grid(False)
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
    ax_rec.grid(False)
    ax_rec.set_ylim(bottom=0)

    # Satellite counts panel
    x_s, med_s, lo_s, hi_s = collect_stats(sub.dropna(subset=["n_served"]), "n_served")
    x_l, med_l, lo_l, hi_l = collect_stats(sub.dropna(subset=["n_lost"]),   "n_lost")
    ax_sat.fill_between(x_s, lo_s, hi_s, color="#2ca02c", alpha=0.20)
    ax_sat.plot(x_s, med_s, color="#2ca02c", lw=2, marker="o", ms=5, label="Serviced")
    ax_sat.fill_between(x_l, lo_l, hi_l, color="#d62728", alpha=0.20)
    ax_sat.plot(x_l, med_l, color="#d62728", lw=2, marker="s", ms=5, label="Lost")
    ax_sat.set_title(f"Propellant Budget vs Satellites ({title_suffix})", fontsize=FS_TITLE)
    ax_sat.set_xlabel("Propellant Capacity [kg]", fontsize=FS)
    ax_sat.set_ylabel("Number of Satellites", fontsize=FS)
    ax_sat.tick_params(labelsize=FS_TICK)
    ax_sat.grid(False)
    ax_sat.set_ylim(bottom=0)
    ax_sat.legend(fontsize=FS_TICK)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

# ── Generate both PDFs ────────────────────────────────────────────────────────

plot_panels(data_knee,   os.path.join(OUT_DIR, "bcr_vs_propellant_knee.pdf"),        "Knee Point")
plot_panels(data_maxcov, os.path.join(OUT_DIR, "bcr_vs_propellant_maxcoverage.pdf"), "Max Coverage")

# ── Manufacturing budget plot (knee point, MDLS only) ─────────────────────────

col = COLOURS["mdls"]
sub = data_knee[data_knee["algorithm"] == "mdls"]

fig, ax = plt.subplots(figsize=(6, 4))
x, med, lo, hi = collect_stats(sub.dropna(subset=["affordable_mfg_M"]), "affordable_mfg_M")
ax.fill_between(x, lo, hi, color=col, alpha=0.20)
ax.plot(x, med, color=col, lw=2, marker="o", ms=5)
ax.set_xlabel("Propellant Capacity [kg]", fontsize=FS)
ax.set_ylabel("Manufacturing Cost [\\$M]", fontsize=FS)
ax.set_title("Propellant Budget vs Manufacturing Cost at BCR = 1", fontsize=FS_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.grid(False)
fig.tight_layout()
out_mfg = os.path.join(OUT_DIR, "manufacturing_budget_vs_propellant.pdf")
fig.savefig(out_mfg, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_mfg}")

print(f"\nConstants used:")
print(f"  V1 base = ${V1_VALUE/1e6:.3f}M → depreciated = ${V1_VALUE_DEP/1e6:.3f}M")
print(f"  V2 base = ${V2_VALUE/1e6:.3f}M → depreciated = ${V2_VALUE_DEP/1e6:.3f}M")
print(f"  avg_V_sat (depreciated) = ${AVG_V_SAT/1e6:.3f}M,  TOTAL_ASSET_V = ${TOTAL_ASSET_V/1e6:.1f}M")
print(f"  MARKET_RATE = ${MARKET_RATE:.2f}/kg,  INTERNAL_RATE = ${INTERNAL_RATE:.2f}/kg")
