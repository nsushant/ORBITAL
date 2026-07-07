"""
plot_ga_tuning.py — Visualise GA hyperparameter sweep results across scenarios.

Reads outputs/tune_ga_results_{scenario}.csv for each scenario and produces
two figures (saved as PDFs):

  1. tune_hv_sensitivity.pdf   — median HV ± min/max across all 9 configs
  2. tune_knee_sensitivity.pdf — median knee unrecovered value ± min/max

Each figure has 4 panels (one per scenario).  Within each panel two groups
(NSGA-III, MOPSO-CD) show the spread due to hyperparameter choice.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np 
# ── Config ────────────────────────────────────────────────────────────────────

SCENARIOS = ["tight_normal", "loose_uniform", "tight_low_dv", "loose_high_dv"]
SCENARIO_LABELS = {
    "tight_normal":   "Scenario 1",
    "loose_uniform":  "Scenario 2",
    "tight_low_dv":   "Scenario 3",
    "loose_high_dv":  "Scenario 4",
}
ALGO_COL  = {"nsga3": "NSGA-III", "pso": "MOPSO-CD"}
COLORS    = {"nsga3": "#2166ac", "pso": "#d6604d"}
IN_DIR    = "outputs"
OUT_DIR   = "outputs"
FS        = 22
FS_TICK   = 22
FS_TITLE  = 24

# ── Load data and add normalised knee column ──────────────────────────────────

frames = {}
missing = []
for sc in SCENARIOS:
    path = os.path.join(IN_DIR, f"tune_ga_results_{sc}.csv")
    if os.path.exists(path):
        frames[sc] = pd.read_csv(path)
    else:
        missing.append(sc)

if missing:
    print(f"[warn] Missing sweep results for: {missing}  — those panels will be blank.")

if not frames:
    raise SystemExit("No sweep result files found. Run tune_ga.py for each scenario first.")

# ── Helper ────────────────────────────────────────────────────────────────────

def summary(df, algo, col):
    """Return (median, min, max) of col for the given algo across all configs."""
    vals = df[df["algo"] == algo][col].dropna().values
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return np.median(vals), vals.min(), vals.max()

# ── Generic panel plotter ─────────────────────────────────────────────────────

def make_figure(metric_col, ylabel, title, out_name, invert=False):
    """
    metric_col : column name in the CSV
    invert     : if True, lower is better — flip y-axis
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle(title, fontsize=FS_TITLE + 2, y=1.01)

    x_nsga = 0
    x_pso  = 1

    for ax, sc in zip(axes, SCENARIOS):
        ax.set_title(SCENARIO_LABELS[sc], fontsize=FS_TITLE)
        ax.set_xticks([x_nsga, x_pso])
        ax.set_xticklabels(["NSGA-III", "MOPSO-CD"], fontsize=FS_TICK, rotation=15)
        ax.set_xlim(-0.5, 1.5)
        ax.tick_params(labelsize=FS_TICK)

        if sc not in frames:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=FS)
            continue

        df = frames[sc]
        for x_pos, algo in [(x_nsga, "nsga3"), (x_pso, "pso")]:
            med, lo, hi = summary(df, algo, metric_col)
            if np.isnan(med):
                continue
            color = COLORS[algo]
            yerr = np.array([[med - lo], [hi - med]])
            ax.errorbar(x_pos, med, yerr=yerr, fmt="o", color=color,
                        capsize=6, capthick=1.5, elinewidth=1.5,
                        markersize=10, zorder=3)

        ax.set_ylabel(ylabel if ax == axes[0] else "", fontsize=FS)
        ax.grid(False)
        if invert:
            ax.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=COLORS[a], label=ALGO_COL[a])
               for a in ["nsga3", "pso"]]
    #fig.legend(handles=patches, loc="lower center", ncol=2,
    #           bbox_to_anchor=(0.5, -0.06), fontsize=FS, frameon=False)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

# ── Generate plots ────────────────────────────────────────────────────────────

make_figure(
    metric_col = "hv",
    ylabel     = "Normalised Hypervolume",
    title      = "HV Sensitivity to Hyperparameters",
    out_name   = "tune_hv_sensitivity.pdf",
    invert     = False,
)

make_figure(
    metric_col = "knee_l2",
    ylabel     = "Knee Point (normalised)",
    title      = "Knee-Point Sensitivity to Hyperparameters",
    out_name   = "tune_knee_sensitivity.pdf",
    invert     = False,   # lower L2 = closer to ideal = better
)
