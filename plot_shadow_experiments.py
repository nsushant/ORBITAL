"""
plot_shadow_experiments.py — KDE shadow plots for numerical experiment Pareto fronts.

Three figures (one per pairwise 2D projection), each with 4 subplots (one per scenario).
KDE contours per algorithm overlaid to show explored objective space.

Run: python plot_shadow_experiments.py
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT     = "outputs"
RES_DIR = os.path.join(OUT, "exp_results")

SCENARIOS = ["tight_normal", "loose_uniform", "tight_low_dv", "loose_high_dv"]
SCENARIO_LABELS = {
    "tight_normal":  "Scenario 1",
    "loose_uniform": "Scenario 2",
    "tight_low_dv":  "Scenario 3",
    "loose_high_dv": "Scenario 4",
}

ALGOS       = ["mdls", "nsga3", "pso"]
ALG_LABELS  = {"mdls": "MDLS", "nsga3": "NSGA-III", "pso": "MOPSO-CD"}
ALG_COLOURS = {"mdls": "#1f77b4", "nsga3": "#ff7f0e", "pso": "#d62728"}

FS       = 24
FS_TICK  = 20
FS_TITLE = 24
MIN_KDE_POINTS = 60
PENALTY  = 1e6
AXIS_PAD = 0.05   # fractional padding beyond data range

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all():
    rows = []
    for sc in SCENARIOS:
        for algo in ALGOS:
            files = sorted(glob.glob(os.path.join(RES_DIR, f"{algo}_{sc}_*.csv")))
            for f in files:
                df = pd.read_csv(f)
                df = df[df["f1_dv"] < PENALTY]
                df["scenario"]  = sc
                df["algorithm"] = ALG_LABELS[algo]
                rows.append(df)
    return pd.concat(rows, ignore_index=True)

print("Loading fronts ...")
df = load_all()
print(f"  {len(df)} total points")

# ---------------------------------------------------------------------------
# Normalisation (matches aggregate_results.py)
# ---------------------------------------------------------------------------

N_DEMANDS  = 200
V1_VAL     = 864_150.0
V2_VAL     = 2_280_000.0
V2_FRAC    = 0.30
DV_BUDGET  = 5000.0
N_VEHICLES = 20
AVG_ASSET  = V2_FRAC * V2_VAL + (1 - V2_FRAC) * V1_VAL
FIXED_NADIR = np.array([N_DEMANDS * DV_BUDGET,
                         N_DEMANDS * AVG_ASSET * 1.10,
                         N_VEHICLES + 5])

# Compute per-scenario ideal from data, normalise
for sc in SCENARIOS:
    sub = df[df["scenario"] == sc]
    if sub.empty:
        df.loc[df["scenario"] == sc, ["f1_norm", "f2_norm", "f3_norm"]] = np.nan
        continue
    ideal = sub[["f1_dv", "f2_unrecovered_value", "f3_vehicles"]].values.min(axis=0)
    rng   = np.maximum(FIXED_NADIR - ideal, 1e-10)
    df.loc[df["scenario"] == sc, "f1_norm"] = (sub["f1_dv"]               - ideal[0]) / rng[0]
    df.loc[df["scenario"] == sc, "f2_norm"] = (sub["f2_unrecovered_value"] - ideal[1]) / rng[1]
    df.loc[df["scenario"] == sc, "f3_norm"] = (sub["f3_vehicles"]          - ideal[2]) / rng[2]

df["vehicles"] = df["f3_vehicles"].astype(float)

# ---------------------------------------------------------------------------
# KDE helper (from plot_shadow.py)
# ---------------------------------------------------------------------------

def plot_kde(ax, x, y, colour):
    if len(x) < 5:
        return
    if len(x) < MIN_KDE_POINTS:
        ax.scatter(x, y, c=colour, s=15, alpha=0.4, linewidths=0)
        return
    try:
        kde  = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xg   = np.linspace(x.min(), x.max(), 150)
        yg   = np.linspace(y.min(), y.max(), 150)
        X, Y = np.meshgrid(xg, yg)
        Z    = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        z_flat = Z[Z > Z.max() * 0.05]
        if len(z_flat) < 3:
            ax.scatter(x, y, c=colour, s=15, alpha=0.4, linewidths=0)
            return
        # 3 levels: inner 50%, 80%, 95% density bands
        levels = np.unique(np.percentile(z_flat, [50, 80, 95]))
        if len(levels) < 2:
            ax.scatter(x, y, c=colour, s=15, alpha=0.4, linewidths=0)
            return
        ax.contour(X, Y, Z, levels=levels, colors=[colour], alpha=0.85,
                   linewidths=1.8)
    except Exception:
        ax.scatter(x, y, c=colour, s=15, alpha=0.4, linewidths=0)

# ---------------------------------------------------------------------------
# Legend handles
# ---------------------------------------------------------------------------

legend_handles = [
    mlines.Line2D([], [], color=ALG_COLOURS[a], linewidth=2.5, marker="o",
                  markersize=7, label=ALG_LABELS[a])
    for a in ALGOS
]

# ---------------------------------------------------------------------------
# Figure definitions
# ---------------------------------------------------------------------------

FIGURES = [
    {
        "xcol": "f2_norm", "ycol": "f1_norm",
        "xlabel": "Lost Value (normalised)", "ylabel": "$\\Delta V$ (normalised)",
        "outfile": "shadow_dv_vs_value.pdf",
        "legend_loc": "upper right",
    },
    {
        "xcol": "f3_norm", "ycol": "f1_norm",
        "xlabel": "Vehicles (normalised)", "ylabel": "$\\Delta V$ (normalised)",
        "outfile": "shadow_dv_vs_vehicles.pdf",
        "legend_loc": "upper left",
    },
    {
        "xcol": "f3_norm", "ycol": "f2_norm",
        "xlabel": "Vehicles (normalised)", "ylabel": "Lost Value (normalised)",
        "outfile": "shadow_value_vs_vehicles.pdf",
        "legend_loc": "upper right",
    },
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

active_scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]

for fdef in FIGURES:
    n = len(active_scenarios)

    # Compute data-driven axis limits across all scenarios and algorithms
    all_x = df[fdef["xcol"]].dropna().values
    all_y = df[fdef["ycol"]].dropna().values
    xpad  = (all_x.max() - all_x.min()) * AXIS_PAD
    ypad  = (all_y.max() - all_y.min()) * AXIS_PAD
    xlim  = (all_x.min() - xpad, all_x.max() + xpad)
    ylim  = (all_y.min() - ypad, all_y.max() + ypad)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7), sharey=False)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.35)

    for i, (ax, sc) in enumerate(zip(axes, active_scenarios)):
        sub = df[df["scenario"] == sc]
        for algo in ALGOS:
            alg_df = sub[sub["algorithm"] == ALG_LABELS[algo]]
            if alg_df.empty:
                continue
            plot_kde(ax,
                     alg_df[fdef["xcol"]].values,
                     alg_df[fdef["ycol"]].values,
                     ALG_COLOURS[algo])
        ax.set_title(SCENARIO_LABELS.get(sc, sc), fontsize=FS_TITLE)
        ax.set_xlabel(fdef["xlabel"], fontsize=FS)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
        if i == 0:
            ax.set_ylabel(fdef["ylabel"], fontsize=FS)
        else:
            ax.tick_params(labelleft=False)
        ax.grid(False)
        ax.tick_params(labelsize=FS_TICK)

    axes[-1].legend(handles=legend_handles, fontsize=FS - 2, loc=fdef["legend_loc"])
    outpath = os.path.join(OUT, fdef["outfile"])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
