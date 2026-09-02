"""
plot_cost_table.py — pixel map of ΔV cost for a single (origin, dest) pair.

Usage:
  python plot_cost_table.py depot_1 sat_1
  python plot_cost_table.py sat_5 depot_1
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from loaders import load_cost_table_jld2, load_sim_name_map

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <origin> <dest>")
    sys.exit(1)

origin_name = sys.argv[1]
dest_name   = sys.argv[2]

COST_FILE = "outputs/cost_table.jld2"

ct, meta = load_cost_table_jld2(COST_FILE)
nm = load_sim_name_map("outputs/simulation.h5", depot_idx=meta["depot_idx"])

if origin_name not in nm:
    sys.exit(f"Unknown origin: {origin_name}")
if dest_name not in nm:
    sys.exit(f"Unknown dest: {dest_name}")

origin_idx = nm[origin_name]
dest_idx   = nm[dest_name]

dep_grid = meta["dep_grid"]
arr_grid = meta["arr_grid"]

# Build 2D cost matrix
cost_matrix = np.full((len(dep_grid), len(arr_grid)), np.nan)
for di, dep in enumerate(dep_grid):
    for ai, arr in enumerate(arr_grid):
        c = ct.get((origin_idx, dest_idx, dep, arr), np.inf)
        if not np.isinf(c):
            cost_matrix[di, ai] = c

fig, ax = plt.subplots(figsize=(10, 8))
vmin = np.nanmin(cost_matrix)
vmax = 10000.0  # cap at 10 km/s to show structure in feasible region
norm = mcolors.LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
im = ax.pcolormesh(arr_grid, dep_grid, cost_matrix, shading="nearest", cmap="viridis", norm=norm)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("ΔV [m/s]", fontsize=14)

ax.set_xlabel("Arrival [days]", fontsize=14)
ax.set_ylabel("Departure [days]", fontsize=14)
ax.set_title(f"Cost table: {origin_name} → {dest_name}", fontsize=16)

out_path = os.path.join("outputs", f"cost_table_{origin_name}_{dest_name}.pdf")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_path}")
