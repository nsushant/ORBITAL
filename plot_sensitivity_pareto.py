"""
plot_sensitivity_pareto.py — Pareto front scatter plots for demand-based sensitivity.

For each sub-experiment: one figure with rows = algorithms, cols = factor levels.
Each panel shows the normalised Pareto front (all trials overlaid as scatter).
Mirrors the layout of plot_pure_pareto.py.

Run: python plot_sensitivity_pareto.py
"""

import os, glob
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

H5_FILE  = "outputs/pure_sensitivity_results.h5"
RES_DIR  = "outputs/pure_results"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = {
    "mdls":       "MDLS",
    "nsga3rk":    "NSGArk",
    "nsga3rk_ot": "NSGArk-OT",
}

COLOURS = {
    "mdls":       "#1f77b4",
    "nsga3rk":    "#d62728",
    "nsga3rk_ot": "#2ca02c",
}

FS      = 14
FS_TICK = 11

SUBEXPS = [
    dict(
        key    = "size",
        title  = "Instance size",
        levels = ["size_10", "size_50", "size_100", "size_150", "size_200"],
        labels = ["N=10", "N=50", "N=100", "N=150", "N=200"],
    ),
    dict(
        key    = "disttype",
        title  = "Distribution type",
        levels = ["disttype_normal", "disttype_uniform"],
        labels = ["Normal", "Uniform"],
    ),
    dict(
        key    = "dv",
        title  = "ΔV threshold",
        levels = ["dv_3000", "dv_5000", "dv_8000", "dv_12000"],
        labels = ["3000 m/s", "5000 m/s", "8000 m/s", "12000 m/s"],
    ),
    dict(
        key    = "dvbudget",
        title  = "Vehicle ΔV budget",
        levels = ["dvbudget_1500", "dvbudget_3000", "dvbudget_5000",
                  "dvbudget_8000", "dvbudget_10000"],
        labels = ["1500", "3000", "5000", "8000", "10000"],
    ),
]

OBJ_PAIRS = [
    ("f1", "f2", "ΔV (norm.)", "Lost value (norm.)"),
    ("f1", "f3", "ΔV (norm.)", "Vehicles (norm.)"),
    ("f2", "f3", "Lost value (norm.)", "Vehicles (norm.)"),
]

# ---------------------------------------------------------------------------
# Normalisation factors (99th pct from pure_results, same as other scripts)
# ---------------------------------------------------------------------------

records = []
for algo in ALGOS:
    for path in glob.glob(os.path.join(RES_DIR, f"{algo}_*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            records.append(df)

if not records:
    raise FileNotFoundError(f"No CSVs found in {RES_DIR}")

all_data = pd.concat(records, ignore_index=True)
REF_DV = all_data["f1_dv"].quantile(0.99)
REF_F2 = all_data["f2_unrecovered_value"].quantile(0.99)
REF_F3 = all_data["f3_vehicles"].quantile(0.99)

print(f"Normalisation: dv={REF_DV:.0f}  f2={REF_F2:.3e}  f3={REF_F3:.0f}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fronts(fid, algo, scenario_key):
    path = f"{algo}/{scenario_key}"
    if path not in fid:
        return []
    grp = fid[path]
    fronts = []
    for trial_key in sorted(grp.keys()):
        data = grp[trial_key][:]
        if data.ndim == 2 and data.shape[0] == 3 and data.shape[1] != 3:
            data = data.T
        fronts.append(data)
    return fronts


def normalise(F):
    Fn = F.copy().astype(float)
    Fn[:, 0] /= REF_DV
    Fn[:, 1] /= REF_F2
    Fn[:, 2] /= REF_F3
    return Fn


OBJ_IDX = {"f1": 0, "f2": 1, "f3": 2}

# ---------------------------------------------------------------------------
# Plot — one PDF per sub-experiment, rows=algos, cols=levels
# ---------------------------------------------------------------------------

if not os.path.exists(H5_FILE):
    raise FileNotFoundError(f"{H5_FILE} not found — run run_pure_sensitivity.sh first")

algo_keys = list(ALGOS.keys())

with h5py.File(H5_FILE, "r") as fid:
    for se in SUBEXPS:
        levels   = se["levels"]
        labels   = se["labels"]
        n_levels = len(levels)
        n_algos  = len(algo_keys)

        # One figure per objective pair
        for (xkey, ykey, xlabel, ylabel) in OBJ_PAIRS:
            xi, yi = OBJ_IDX[xkey], OBJ_IDX[ykey]

            fig, axes = plt.subplots(n_algos, n_levels,
                                     figsize=(3.5 * n_levels, 3.5 * n_algos),
                                     sharex=True, sharey=True)
            if n_algos == 1:
                axes = [axes]
            if n_levels == 1:
                axes = [[ax] for ax in axes]

            fig.suptitle(f"{se['title']} — {xlabel} vs {ylabel}", fontsize=FS + 2)

            for row, algo in enumerate(algo_keys):
                for col, (lv, lv_label) in enumerate(zip(levels, labels)):
                    ax = axes[row][col]
                    fronts = load_fronts(fid, algo, lv)
                    for F in fronts:
                        if len(F) == 0:
                            continue
                        Fn = normalise(F)
                        ax.scatter(Fn[:, xi], Fn[:, yi],
                                   c=COLOURS[algo], s=8, alpha=0.5, linewidths=0)
                    if row == 0:
                        ax.set_title(lv_label, fontsize=FS)
                    if col == 0:
                        ax.set_ylabel(f"{ALGOS[algo]}\n{ylabel}", fontsize=FS - 2)
                    if row == n_algos - 1:
                        ax.set_xlabel(xlabel, fontsize=FS - 2)
                    ax.tick_params(labelsize=FS_TICK)
                    ax.grid(False)

            plt.tight_layout()
            pair_tag = f"{xkey}_{ykey}"
            out = os.path.join(OUT_DIR, f"sensitivity_pareto_{se['key']}_{pair_tag}.pdf")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")

print("Done.")
