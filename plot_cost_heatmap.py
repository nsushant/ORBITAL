"""
plot_cost_heatmap.py

Plot the ΔV cost surface for a satellite pair as a scatter/heatmap over
departure × arrival time. Visually confirms whether the AMR grid is
non-uniform (dense where gradients are steep, sparse where flat).

Usage:
    python plot_cost_heatmap.py               # depot(301) → sat_1(1)
    python plot_cost_heatmap.py 301 50        # depot → sat_50
    python plot_cost_heatmap.py 1 2           # sat_1 → sat_2
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

COST_FILE = "outputs/cost_table.h5"

# Parse CLI args
from_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 301
to_idx   = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print(f"Loading cost table from {COST_FILE} ...")
with h5py.File(COST_FILE, "r") as f:
    fr  = f["from_idx"][()]
    to  = f["to_idx"][()]
    dep = f["dep"][()]
    arr = f["arr"][()]
    dv  = f["cost"][()]

mask = (fr == from_idx) & (to == to_idx)
n = mask.sum()
print(f"Pair ({from_idx} → {to_idx}): {n} grid points")

if n == 0:
    print("No entries found for this pair. Try a different pair.")
    sys.exit(1)

dep_p = dep[mask]
arr_p = arr[mask]
dv_p  = dv[mask]

# Unique grid axis counts
n_dep = len(np.unique(dep_p))
n_arr = len(np.unique(arr_p))
print(f"  Unique dep times: {n_dep},  unique arr times: {n_arr}")
print(f"  ΔV range: {dv_p.min():.0f} – {dv_p.max():.0f} m/s")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Cost table pair: {from_idx} → {to_idx}  ({n} points)", fontsize=12)

# ── Left: grid structure ──────────────────────────────────────────────────────
axes[0].scatter(dep_p, arr_p, s=6, c="steelblue", alpha=0.6, linewidths=0)
axes[0].set(title="Grid point distribution",
            xlabel="Departure time [days]",
            ylabel="Arrival time [days]")
axes[0].text(0.02, 0.97,
             f"{n_dep} dep × {n_arr} arr = {n} pts",
             transform=axes[0].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# ── Right: ΔV heatmap ─────────────────────────────────────────────────────────
vmin = np.nanpercentile(dv_p, 2)
vmax = np.nanpercentile(dv_p, 98)
sc = axes[1].scatter(dep_p, arr_p, s=6, c=dv_p,
                     cmap="plasma", vmin=vmin, vmax=vmax,
                     linewidths=0)
plt.colorbar(sc, ax=axes[1], label="ΔV [m/s]")
axes[1].set(title="ΔV cost surface",
            xlabel="Departure time [days]",
            ylabel="Arrival time [days]")

plt.tight_layout()
out = f"cost_heatmap_{from_idx}_{to_idx}.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
