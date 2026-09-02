"""
plot_bcr_mass_trade.py — Implied-contract BCR* vs servicer dry mass, per client.

Uses the max-coverage (min mix-f2) solution. Manufacturing is independent of
dry mass; launch and xenon still scale with wet mass, so C(m) and thus BCR*
move with m.

Run: python plot_bcr_mass_trade.py --h5 outputs/long_horizon_results_edelbaum.h5
"""

import os, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bcr_model import (
    unpack_front, client_slice, operator_cost, implied_contract,
    client_label, DV_BUDGET,
)

parser = argparse.ArgumentParser()
parser.add_argument("--h5",      default="outputs/long_horizon_results_edelbaum.h5")
parser.add_argument("--out-dir", default="outputs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

BUDGETS = [int(DV_BUDGET)]
COLORS  = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]


def select_best_f2(data):
    return data[np.argmin(data[:, 1])]


best = {}   # budget -> (row, front)
with h5py.File(args.h5, "r") as f:
    for b in BUDGETS:
        path = f"mdls/bcr_mixed_{b}"
        if path not in f:
            continue
        grp = f[path]
        for tk in sorted(grp.keys()):
            front = unpack_front(grp[tk])
            data = front["data"]
            data = data[data[:, 0] < 1e6]
            if len(data) == 0 or not front["clients"]:
                continue
            best[b] = (select_best_f2(data), front)

if not best:
    raise SystemExit("No per-client fronts — re-run MDLS after sat_clients.json")

clients = None
print(f"Loaded {len(best)} budget levels")
for b in sorted(best):
    row, front = best[b]
    if clients is None:
        clients = list(front["clients"])
    print(f"  dv={b:5d}  max-cov f3={row[2]:.0f}  mix f2=${row[1]/1e6:.0f}M")

mass_range = np.concatenate([
    np.arange(50,  500,  25),
    np.arange(500, 2001, 50),
]).astype(int)

bcr_by_client = {c: [] for c in clients}
for m in mass_range:
    for c in clients:
        vals = []
        for b, (row, front) in best.items():
            f1, f3 = float(row[0]), float(row[2])
            rec, uns, _ = client_slice(front, row, c)
            C = operator_cost(f1, f3, b, float(m))
            _, bcr = implied_contract(rec, uns, C)
            vals.append(bcr)
        bcr_by_client[c].append(np.mean(vals))

FS = 20
masses = np.array(mass_range, dtype=float)
fig, ax = plt.subplots(figsize=(10, 6))
for i, c in enumerate(clients):
    ax.plot(masses, bcr_by_client[c], color=COLORS[i % len(COLORS)],
            linewidth=2.5, label=client_label(c))
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Servicer dry mass [kg]", fontsize=FS)
ax.set_ylabel("BCR*", fontsize=FS)
ax.tick_params(labelsize=FS - 2)
ax.set_xlim(masses[0], masses[-1])
ax.set_ylim(bottom=0)
ax.legend(fontsize=FS - 4)

fig.tight_layout()
outpath = os.path.join(args.out_dir, "bcr_mass_trade.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
