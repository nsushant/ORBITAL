"""
plot_bcr_architecture.py — OOS architecture feasibility from MDLS SE4/SE5 sweep.

For each propellant budget level:
  - Plot ALL Pareto-optimal solutions (all trials) as scatter
  - x = fleet size (f3), y = unrecovered value (f2) [$M]
  - colour = BCR (RdYlGn diverging, centred at BCR=1)
  - One subplot per ΔV budget level

Run: python plot_bcr_architecture.py
"""

import os, glob, math, argparse, json
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--h5",      default="outputs/pure_sensitivity_results.h5",
                    help="HDF5 results file")
parser.add_argument("--res-dir", default="outputs/pure_results",
                    help="Directory with normalisation CSVs")
parser.add_argument("--out-dir", default="outputs",
                    help="Output directory for PDFs")
args = parser.parse_args()

H5_FILE = args.h5
RES_DIR = args.res_dir
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

DV_BUDGET_SOURCES = [
    ("bcr_mixed", [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000]),
]

FS      = 14
FS_TICK = 12

# ---------------------------------------------------------------------------
# BCR constants
# ---------------------------------------------------------------------------

G0          = 9.80665
ISP_XE      = 2800.0
VE_XE       = ISP_XE * G0
XE_COST     = 340.0
F9_COST     = 74_000_000.0
F9_CAPACITY = 22_000.0
MARKET_RATE = F9_COST / F9_CAPACITY
M_DRY       = 300.0

# ---------------------------------------------------------------------------
# Manufacturing cost model
# SSCM CER (Foreman et al. 2016, Eq. 1): C_bus [FY00$K] = 781 + 26.1*m^1.261
# C_total = C_bus * (1 + alpha);  inflation FY2000->2026: x2.005 (NASA NNSI 2026)
# Wright's law (b=0.67): RC_avg(N) = RC_1 * N^(log2(0.67)) = RC_1 * N^(-0.578)
# C_unit(N) = NRE/N + RC_avg(N),  NRE=0.6*C_total, RC_1=0.4*C_total
# ---------------------------------------------------------------------------

INFLATION = 2.005

def sscm_unit_cost(m_dry, alpha, N):
    """Manufacturing cost per unit [$] using SSCM CER + learning curve."""
    c_bus_fy00k = 781.0 + 26.1 * (m_dry ** 1.261)
    c_total     = c_bus_fy00k * 1000.0 * INFLATION * (1.0 + alpha)
    nre         = 0.6 * c_total
    rc1         = 0.4 * c_total
    rc_avg      = rc1 * (N ** (-0.578))
    return nre / N + rc_avg

OPERATOR_FEE = 0.10   # operator receives 10% of recovered satellite value as revenue
PENALTY      = 1e6

def compute_bcr(f1, f2, f3, dv_budget, total_demand_value):
    """Operator BCR = revenue / costs.
    revenue   = OPERATOR_FEE × recovered value (10% of satellite replacement value saved)
    recovered = total_demand_value - f2
    N in learning curve = f3 (actual fleet size produced).
    """
    n         = max(1, int(round(f3)))
    c_vehicle = sscm_unit_cost(M_DRY, 1.5, n)
    m_prop    = M_DRY * (math.exp(dv_budget / VE_XE) - 1.0)
    m_wet     = M_DRY + m_prop
    recovered = max(0.0, total_demand_value - f2)
    revenue   = OPERATOR_FEE * recovered
    n_sorties = f1 / dv_budget
    costs     = f3 * m_wet * MARKET_RATE + n_sorties * m_prop * XE_COST + f3 * c_vehicle
    return revenue / costs if costs > 0 else np.nan

# ---------------------------------------------------------------------------
# Load ALL Pareto solutions for each (budget, trial)
# ---------------------------------------------------------------------------

if not os.path.exists(H5_FILE):
    raise FileNotFoundError(f"{H5_FILE} not found")

# Collect unique budget levels (sorted)
all_budget_levels = []
for _, budgets in DV_BUDGET_SOURCES:
    for b in budgets:
        if b not in all_budget_levels:
            all_budget_levels.append(b)
all_budget_levels = sorted(set(all_budget_levels))

# Map budget -> list of solution rows (f1, f2, f3, bcr)
budget_data = {b: [] for b in all_budget_levels}

with h5py.File(H5_FILE, "r") as fid:
    for prefix, budgets in DV_BUDGET_SOURCES:
        for dv_budget in budgets:
            key  = f"{prefix}_{dv_budget}"
            path = f"mdls/{key}"
            if path not in fid:
                print(f"  [skip] {path} not in HDF5")
                continue
            grp = fid[path]
            for trial_key in sorted(grp.keys()):
                ds = grp[trial_key]
                data = ds[:]
                if data.ndim == 2 and data.shape[0] == 3 and data.shape[1] != 3:
                    data = data.T
                if len(data) == 0:
                    continue
                data = data[data[:, 0] < PENALTY]
                if len(data) == 0:
                    continue
                total_demand_value = ds.attrs.get("total_demand_value", None)
                if total_demand_value is None or total_demand_value == 0:
                    # fallback: estimate from max f2 in this trial
                    total_demand_value = float(data[:, 1].max())
                for row in data:
                    f1, f2, f3 = row[0], row[1], row[2]
                    bcr = compute_bcr(f1, f2, f3, dv_budget, total_demand_value)
                    budget_data[dv_budget].append({
                        "f1": f1, "f2": f2 / 1e6, "f3": f3, "bcr": bcr
                    })

# ---------------------------------------------------------------------------
# Plot — one panel per budget level
# ---------------------------------------------------------------------------

n_budgets = len(all_budget_levels)
ncols = 4
nrows = math.ceil(n_budgets / ncols)

# Compute global BCR range for shared colormap
all_bcrs = []
for b in all_budget_levels:
    for row in budget_data[b]:
        if row["bcr"] is not None and not np.isnan(row["bcr"]):
            all_bcrs.append(row["bcr"])

bcr_max = min(np.nanpercentile(all_bcrs, 95), 5.0) if all_bcrs else 3.0
bcr_max = max(bcr_max, 1.01)   # ensure vmax > vcenter for TwoSlopeNorm
norm    = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=bcr_max)
cmap    = cm.RdYlGn

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(4.5 * ncols, 4.0 * nrows),
                          constrained_layout=True)
axes = np.array(axes).flatten()

for i, dv_budget in enumerate(all_budget_levels):
    ax   = axes[i]
    rows = budget_data[dv_budget]
    if not rows:
        ax.set_visible(False)
        continue

    df_b = pd.DataFrame(rows)
    sc = ax.scatter(
        df_b["f3"], df_b["f2"],
        c=df_b["bcr"], cmap=cmap, norm=norm,
        s=30, alpha=0.7, edgecolors="none"
    )

    ax.set_title(f"ΔV = {dv_budget/1000:.1f} km/s", fontsize=FS)
    ax.set_xlabel("Fleet size", fontsize=FS)
    ax.set_ylabel("Unrecovered value [$M]", fontsize=FS)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Hide unused panels
for j in range(n_budgets, len(axes)):
    axes[j].set_visible(False)

# Shared colourbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[:n_budgets], shrink=0.6, pad=0.02)
cbar.set_label("BCR", fontsize=FS)
cbar.ax.tick_params(labelsize=FS_TICK)
cbar.ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")

out = os.path.join(OUT_DIR, "bcr_architecture.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
