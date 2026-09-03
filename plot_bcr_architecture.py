"""
plot_bcr_architecture.py — Pareto scatter coloured by implied-contract BCR*.

One panel per client. x = fleet size, y = that client's unrecovered value [$M],
colour = BCR* (RdYlGn, centred at 1).

Run: python3 plot_bcr_architecture.py --h5 outputs/long_horizon_results_edelbaum.h5
"""

import os, argparse
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from bcr_model import (
    unpack_front, client_slice, operator_cost, implied_contract,
    client_label, DV_BUDGET, M_DRY,
)

parser = argparse.ArgumentParser()
parser.add_argument("--h5",      default="outputs/long_horizon_results_edelbaum.h5")
parser.add_argument("--res-dir", default="outputs/long_horizon_results_edelbaum")
parser.add_argument("--out-dir", default="outputs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

DV_BUDGET_SOURCES = [
    ("bcr_mixed", [int(DV_BUDGET)]),
]
FS      = 14
FS_TICK = 12
PENALTY = 1e6

if not os.path.exists(args.h5):
    raise FileNotFoundError(f"{args.h5} not found")

client_rows = {}
with h5py.File(args.h5, "r") as fid:
    for prefix, budgets in DV_BUDGET_SOURCES:
        for dv_budget in budgets:
            path = f"mdls/{prefix}_{dv_budget}"
            if path not in fid:
                print(f"  [skip] {path} not in HDF5")
                continue
            grp = fid[path]
            for trial_key in sorted(grp.keys()):
                front = unpack_front(grp[trial_key])
                data = front["data"]
                if len(data) == 0:
                    continue
                data = data[data[:, 0] < PENALTY]
                if len(data) == 0 or not front["clients"]:
                    continue
                for row in data:
                    f1, f2, f3 = float(row[0]), float(row[1]), float(row[2])
                    C = operator_cost(f1, f3, dv_budget, M_DRY)
                    for c in front["clients"]:
                        rec, uns, _ = client_slice(front, row, c)
                        _, bcr = implied_contract(rec, uns, C)
                        client_rows.setdefault(c, []).append({
                            "f1": f1, "f2": uns / 1e6, "f3": f3, "bcr": bcr,
                        })

clients = [c for c in ("starlink", "planet") if c in client_rows] or list(client_rows)
if not clients:
    raise SystemExit("No per-client columns — re-run MDLS after sat_clients.json")

all_bcrs = [r["bcr"] for rows in client_rows.values() for r in rows
            if r["bcr"] is not None and np.isfinite(r["bcr"])]
bcr_max = min(np.nanpercentile(all_bcrs, 95), 5.0) if all_bcrs else 3.0
bcr_max = max(bcr_max, 1.01)
norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=bcr_max)
cmap = cm.RdYlGn

n = len(clients)
fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), constrained_layout=True)
if n == 1:
    axes = [axes]

for ax, c in zip(axes, clients):
    df_b = pd.DataFrame(client_rows[c])
    ax.scatter(df_b["f3"], df_b["f2"], c=df_b["bcr"], cmap=cmap, norm=norm,
               s=30, alpha=0.7, edgecolors="none")
    ax.set_title(client_label(c), fontsize=FS)
    ax.set_xlabel("Fleet size", fontsize=FS)
    ax.set_ylabel("Unrecovered value [$M]", fontsize=FS)
    ax.tick_params(labelsize=FS_TICK)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
cbar.set_label("BCR*", fontsize=FS)
cbar.ax.tick_params(labelsize=FS_TICK)
cbar.ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")

out = os.path.join(args.out_dir, "bcr_architecture.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
