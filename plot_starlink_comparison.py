"""
plot_starlink_comparison.py

Compare MDLS (Julia) vs NSGA-III (Python) Pareto fronts on the Starlink
satellite servicing problem.

Shadow plots use KDE contours (matching plot_shadow.py style) with a scatter
fallback when there are too few points. Hypervolume is computed with moocore
using a fixed reference point derived from MDLS only, so penalty-inflated
NSGA-III solutions don't distort the comparison.

Run: python plot_starlink_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import gaussian_kde
import moocore
import os

MDLS_FILE  = "/Users/sushantnigudkar/basic_project/outputs/mdls_pareto_comparative.csv"
NSGA3_FILE = "outputs/nsga3_pareto_comparative.csv"
MOEAD_FILE = "outputs/moead_pareto_comparative.csv"
PSO_FILE   = "outputs/pso_pareto_comparative.csv"

OBJ_COLS = ["f1_dv", "f2_unassigned_time", "f3_vehicles"]
PENALTY  = 1e6

def load_front(path, label):
    if not os.path.exists(path):
        print(f"{label}: file not found ({path}), skipping.")
        return pd.DataFrame(columns=OBJ_COLS)
    raw = pd.read_csv(path)[OBJ_COLS].copy()
    df  = raw[np.isfinite(raw["f1_dv"]) & (raw["f1_dv"] < PENALTY)].copy()
    print(f"{label}: {len(df)} points ({len(raw) - len(df)} filtered out)")
    return df

# ── Load data ─────────────────────────────────────────────────────────────────
mdls  = load_front(MDLS_FILE,  "MDLS    ")
nsga3 = load_front(NSGA3_FILE, "NSGA-III")
moead = load_front(MOEAD_FILE, "MOEA-D  ")
pso   = load_front(PSO_FILE,   "PSO     ")

print()
for name, df in [("MDLS    ", mdls), ("NSGA-III", nsga3), ("MOEA-D  ", moead), ("PSO     ", pso)]:
    if len(df) == 0:
        print(f"{name} — (no valid points)"); continue
    print(f"{name} — ΔV: [{df.f1_dv.min():.0f}, {df.f1_dv.max():.0f}]  "
          f"unserved: [{df.f2_unassigned_time.min():.1f}, {df.f2_unassigned_time.max():.1f}]  "
          f"vehicles: [{df.f3_vehicles.min():.0f}, {df.f3_vehicles.max():.0f}]")

# ── Hypervolume ───────────────────────────────────────────────────────────────
all_fronts = [df for df in [mdls, nsga3, moead, pso] if len(df) > 0]
combined   = np.vstack([df[OBJ_COLS].values for df in all_fronts])
ref        = combined.max(axis=0) * 1.1
print(f"\nReference point (1.1 × combined worst): {ref}")

def hv(df):
    return moocore.hypervolume(df[OBJ_COLS].values, ref=ref) if len(df) > 0 else 0.0

hv_mdls  = hv(mdls)
hv_nsga3 = hv(nsga3)
hv_moead = hv(moead)
hv_pso   = hv(pso)
print(f"HV MDLS    : {hv_mdls:.4e}")
print(f"HV NSGA-III: {hv_nsga3:.4e}")
print(f"HV MOEA-D  : {hv_moead:.4e}")
print(f"HV PSO     : {hv_pso:.4e}")
if hv_mdls > 0:
    print(f"HV ratio NSGA/MDLS : {hv_nsga3/hv_mdls:.4f}")
    print(f"HV ratio MOEAD/MDLS: {hv_moead/hv_mdls:.4f}")
    print(f"HV ratio PSO/MDLS  : {hv_pso/hv_mdls:.4f}")

# ── KDE shadow helper ─────────────────────────────────────────────────────────
MIN_KDE = 15

def plot_kde_shadow(ax, x, y, colour, label):
    """KDE contours if enough points, scatter fallback otherwise."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return
    if len(x) < MIN_KDE:
        ax.scatter(x, y, c=colour, s=35, alpha=0.75, linewidths=0,
                   label=label, zorder=3)
        return
    try:
        kde    = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xg     = np.linspace(x.min(), x.max(), 120)
        yg     = np.linspace(y.min(), y.max(), 120)
        X, Y   = np.meshgrid(xg, yg)
        Z      = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        z_flat = Z[Z > Z.max() * 0.05]
        levels = np.unique(np.percentile(z_flat, [40, 65, 82, 93, 98]))
        if len(levels) < 2:
            raise ValueError("too few levels")
        ax.contour(X, Y, Z, levels=levels, colors=[colour],
                   alpha=0.85, linewidths=1.5)
        # invisible scatter for legend entry
        ax.scatter([], [], c=colour, s=35, label=label)
    except Exception:
        ax.scatter(x, y, c=colour, s=35, alpha=0.75, linewidths=0,
                   label=label, zorder=3)

# ── Plot ──────────────────────────────────────────────────────────────────────
MDLS_COL  = "#1f77b4"
NSGA3_COL = "#ff7f0e"
MOEAD_COL = "#2ca02c"
PSO_COL   = "#9467bd"

pairs = [
    ("f2_unassigned_time", "f1_dv", "Unserved time [days]", "ΔV [m/s]"),
    ("f3_vehicles",        "f1_dv", "Vehicles used",        "ΔV [m/s]"),
    ("f2_unassigned_time", "f3_vehicles", "Unserved time [days]", "Vehicles used"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
hv_str = (f"HV: MDLS={hv_mdls:.3e}  NSGA-III={hv_nsga3:.3e}  "
          f"MOEA-D={hv_moead:.3e}  PSO={hv_pso:.3e}")
fig.suptitle(f"MDLS vs NSGA-III vs MOEA-D vs PSO — Starlink  |  {hv_str}", fontsize=9)

for ax, (xc, yc, xl, yl) in zip(axes, pairs):
    plot_kde_shadow(ax, mdls[xc],  mdls[yc],  MDLS_COL,
                    f"MDLS ({len(mdls)} pts)")
    plot_kde_shadow(ax, nsga3[xc], nsga3[yc], NSGA3_COL,
                    f"NSGA-III ({len(nsga3)} pts)")
    plot_kde_shadow(ax, moead[xc], moead[yc], MOEAD_COL,
                    f"MOEA-D ({len(moead)} pts)")
    plot_kde_shadow(ax, pso[xc],   pso[yc],   PSO_COL,
                    f"PSO ({len(pso)} pts)")
    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

plt.tight_layout()
out = "outputs/starlink_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out}")
