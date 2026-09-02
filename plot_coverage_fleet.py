"""
plot_coverage_fleet.py — Fleet size vs implied-contract BCR* and F* per client.

For each fleet size, pick the cheapest mutually viable equal-BCR contract:
argmin F* among points with BCR* >= 1. F* equalizes client and operator BCR;
BCR* is that common value. Slices with no viable point are omitted (nan).

Run: python plot_coverage_fleet.py --h5 outputs/long_horizon_results_edelbaum.h5
"""

import os, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from bcr_model import (
    unpack_front, min_viable_contract, client_label, DV_BUDGET, M_DRY,
)

parser = argparse.ArgumentParser()
parser.add_argument("--h5",       default="outputs/long_horizon_results_edelbaum.h5")
parser.add_argument("--out-dir",  default="outputs")
parser.add_argument("--scenario", default="bcr_mixed")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

BUDGETS = [int(DV_BUDGET)]
MAX_F3  = 50
COLORS  = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

all_by_f3 = defaultdict(list)
clients = []
with h5py.File(args.h5, "r") as f:
    for b in BUDGETS:
        path = f"mdls/{args.scenario}_{b}"
        if path not in f:
            continue
        grp = f[path]
        for tk in sorted(grp.keys()):
            front = unpack_front(grp[tk])
            data, tdv = front["data"], front["tdv"]
            if tdv <= 0:
                continue
            if not clients:
                clients = list(front["clients"])
            data = data[data[:, 0] < 1e6]
            for row in data:
                key = int(round(row[2]))
                if key < 1 or key > MAX_F3:
                    continue
                all_by_f3[key].append((row, tdv, float(b), front))

if not all_by_f3:
    raise SystemExit(f"No solutions in {args.h5} for {args.scenario}_*")
if not clients:
    raise SystemExit("No per-client columns — re-run MDLS after sat_clients.json")

bcr_by = {c: defaultdict(list) for c in clients}
F_by   = {c: defaultdict(list) for c in clients}

for key, recs in all_by_f3.items():
    by_trial = defaultdict(list)
    for rec in recs:
        by_trial[id(rec[3])].append(rec)
    for trial_recs in by_trial.values():
        rows = np.array([r[0] for r in trial_recs], dtype=float)
        front = trial_recs[0][3]
        b = trial_recs[0][2]
        picked = min_viable_contract(front, b, M_DRY, data=rows)
        for c in clients:
            s = picked.get(c)
            if s is None or not s["viable"]:
                continue
            bcr_by[c][key].append(s["bcr"])
            F_by[c][key].append(s["F"] / 1e6)

f3_vals = sorted(set().union(*(bcr_by[c].keys() for c in clients)))
if not f3_vals:
    raise SystemExit("No fleet size has BCR* >= 1")
print(f"Viable fleet sizes: {min(f3_vals)} – {max(f3_vals)}  clients={clients}")
for c in clients:
    n_ok = sum(len(bcr_by[c][k]) for k in f3_vals)
    print(f"  {c}: {n_ok} viable (f3, trial) picks")


def band(by_f3, keys):
    mean, lo, hi = [], [], []
    for k in keys:
        vals = [v for v in by_f3.get(k, []) if np.isfinite(v)]
        if not vals:
            mean.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            mean.append(np.mean(vals)); lo.append(np.min(vals)); hi.append(np.max(vals))
    return np.array(mean), np.array(lo), np.array(hi)


FS = 18
f3_arr = np.array(f3_vals)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i, c in enumerate(clients):
    col = COLORS[i % len(COLORS)]
    mean, lo, hi = band(bcr_by[c], f3_vals)
    ax1.fill_between(f3_arr, lo, hi, alpha=0.15, color=col)
    ax1.plot(f3_arr, mean, color=col, linewidth=2.5, label=client_label(c))
    Fm, Flo, Fhi = band(F_by[c], f3_vals)
    ax2.fill_between(f3_arr, Flo, Fhi, alpha=0.15, color=col)
    ax2.plot(f3_arr, Fm, color=col, linewidth=2.5, label=client_label(c))

ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
ax1.set_xlabel("Fleet size (number of servicers)", fontsize=FS)
ax1.set_ylabel("BCR*", fontsize=FS)
ax1.tick_params(labelsize=FS - 2)
ax1.set_xlim(1, MAX_F3)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=FS - 4)

ax2.set_xlabel("Fleet size (number of servicers)", fontsize=FS)
ax2.set_ylabel("Implied contract F* [$M]", fontsize=FS)
ax2.tick_params(labelsize=FS - 2)
ax2.set_xlim(1, MAX_F3)
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=FS - 4)

fig.tight_layout()
outpath = os.path.join(args.out_dir, "coverage_vs_fleet.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
