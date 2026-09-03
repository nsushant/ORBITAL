"""
plot_shadow.py — Two panels showing Pareto front projections as KDE contours.

Panel 1 (shadow_dv_vs_unassigned.pdf):
  4 subplots (one per instance) — ΔV vs unassigned service time, KDE per algorithm.

Panel 2 (shadow_dv_vs_vehicles.pdf):
  4 subplots (one per instance) — ΔV vs vehicles used, KDE per algorithm.

Run: python3 plot_shadow.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import gaussian_kde
import os

OUT = "outputs"
src = f"{OUT}/pareto_fronts.csv"

if not os.path.exists(src):
    raise FileNotFoundError(f"{src} not found — run numerical_experiments.jl first")

df        = pd.read_csv(src)
instances = ["tight_normal", "loose_uniform", "tight_low_dv", "loose_high_dv"]
instances = [i for i in instances if i in df["instance"].unique()]
alg_names = ["MDLS", "NSGA-III", "MOPSO-CD"]

SCENARIO_LABELS = {
    "tight_normal":   "Scenario 1",
    "loose_uniform":  "Scenario 2",
    "tight_low_dv":   "Scenario 3",
    "loose_high_dv":  "Scenario 4",
}

# Normalise ΔV globally so scale is consistent across instances
dv_min   = df["f1_dv"].min()
dv_max   = df["f1_dv"].max()
dv_range = max(dv_max - dv_min, 1e-10)
df["f1_dv_norm"] = (df["f1_dv"] - dv_min) / dv_range

COLOURS = {
    "MDLS":     "#1f77b4",
    "NSGA-III": "#ff7f0e",
    "MOPSO-CD": "#d62728",
}

FS       = 22   # base font size
FS_TICK  = 22
FS_TITLE = 22

legend_handles = [
    mlines.Line2D([], [], color=COLOURS[a], linewidth=2.5, marker="o",
                  markersize=7, label=a)
    for a in alg_names
]

MIN_KDE_POINTS = 60   # below this, fall back to scatter

def plot_kde(ax, x, y, colour):
    """KDE contours if enough points, otherwise scatter fallback."""
    if len(x) < 5:
        return
    if len(x) < MIN_KDE_POINTS:
        ax.scatter(x, y, c=colour, s=20, alpha=0.5, linewidths=0)
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
            ax.scatter(x, y, c=colour, s=20, alpha=0.5, linewidths=0)
            return
        ax.contour(X, Y, Z, levels=levels, colors=[colour], alpha=0.85,
                   linewidths=1.5)
    except Exception:
        ax.scatter(x, y, c=colour, s=20, alpha=0.5, linewidths=0)


# ── Panel 1: ΔV vs unassigned service time ────────────────────────────────────
fig1, axes1 = plt.subplots(1, len(instances), figsize=(6 * len(instances), 6),
                            sharey=False)
fig1.suptitle("ΔV vs Unassigned Service Time", fontsize=FS_TITLE + 2, y=1.02)
fig1.subplots_adjust(wspace=0.20)

for i, (ax, inst) in enumerate(zip(axes1, instances)):
    sub = df[df["instance"] == inst]
    for alg in alg_names:
        alg_df = sub[sub["algorithm"] == alg]
        plot_kde(ax,
                 alg_df["f2_unassigned_time"].values,
                 alg_df["f1_dv_norm"].values,
                 COLOURS[alg])
    ax.set_title(SCENARIO_LABELS.get(inst, inst), fontsize=FS_TITLE)
    ax.set_xlabel("Time Unserviced [days]", fontsize=FS)
    if i == 0:
        ax.set_ylabel("ΔV (normalised)", fontsize=FS)
    ax.grid(False)
    ax.tick_params(labelsize=FS_TICK)
    inst_dv = sub["f1_dv_norm"]
    pad = (inst_dv.max() - inst_dv.min()) * 0.05
    ax.set_ylim(inst_dv.min() - pad, inst_dv.max() + pad)

axes1[-1].legend(handles=legend_handles, fontsize=FS, loc="upper right")
out1 = f"{OUT}/shadow_dv_vs_unassigned.pdf"
fig1.savefig(out1, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")

# ── Panel 2: ΔV vs vehicles used ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, len(instances), figsize=(6 * len(instances), 6),
                            sharey=False)
fig2.suptitle("ΔV vs Vehicles Used", fontsize=FS_TITLE + 2, y=1.02)
fig2.subplots_adjust(wspace=0.20)

for i, (ax, inst) in enumerate(zip(axes2, instances)):
    sub = df[df["instance"] == inst]
    for alg in alg_names:
        alg_df = sub[sub["algorithm"] == alg]
        plot_kde(ax,
                 alg_df["f3_vehicles"].values.astype(float),
                 alg_df["f1_dv_norm"].values,
                 COLOURS[alg])
    ax.set_title(SCENARIO_LABELS.get(inst, inst), fontsize=FS_TITLE)
    ax.set_xlabel("Vehicles used", fontsize=FS)
    if i == 0:
        ax.set_ylabel("ΔV (normalised)", fontsize=FS)
    ax.grid(False)
    ax.tick_params(labelsize=FS_TICK)
    inst_dv = sub["f1_dv_norm"]
    pad = (inst_dv.max() - inst_dv.min()) * 0.05
    ax.set_ylim(inst_dv.min() - pad, inst_dv.max() + pad)

axes2[-1].legend(handles=legend_handles, fontsize=FS, loc="upper left")
out2 = f"{OUT}/shadow_dv_vs_vehicles.pdf"
fig2.savefig(out2, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")
