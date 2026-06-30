"""
plot_knee_sensitivity.py — Knee point comparison across OFAT sensitivity runs.

Knee point = solution closest to ideal [0,0,0] in normalised objective space,
i.e. argmin sqrt(f1_dv_norm² + f2_unassigned_norm² + f3_vehicles_norm²).

Produces 3 PDFs (one per sub-experiment), matching the layout of sensitivity_hv_*.pdf:
  - Numeric factors (size, dv): line + IQR band per algorithm vs factor level
  - Categorical factor (disttype): grouped boxplot per level

Run: python plot_knee_sensitivity.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

OUT = "outputs"

COLOURS = {
    "MDLS":     "#1f77b4",
    "NSGA-III": "#ff7f0e",
    "MOPSO-CD": "#d62728",
}
ALG_NAMES = ["MDLS", "NSGA-III", "MOPSO-CD"]

FS       = 25
FS_TICK  = 25
FS_TITLE = 25

NORM_COLS = ["f1_dv_norm", "f2_unassigned_norm", "f3_vehicles_norm"]

SUBEXPS = [
    dict(
        name         = "size",
        front_csv    = "sensitivity_size_fronts.csv",
        factor_col   = "num_demands",
        factor_label = "Number of demands",
        levels       = ["10", "50", "100", "150", "200"],
        level_labels = ["10", "50", "100", "150", "200"],
        numeric      = True,
        knee_out     = "sensitivity_knee_size.pdf",
    ),
    dict(
        name         = "disttype",
        front_csv    = "sensitivity_disttype_fronts.csv",
        factor_col   = "disttype",
        factor_label = "Demand distribution",
        levels       = ["normal", "uniform"],
        level_labels = ["Normal", "Uniform"],
        numeric      = False,
        knee_out     = "sensitivity_knee_disttype.pdf",
    ),
    dict(
        name         = "dv",
        front_csv    = "sensitivity_dv_fronts.csv",
        factor_col   = "deltaV_dist",
        factor_label = "ΔV threshold [m/s]",
        levels       = ["3000", "5000", "8000", "12000"],
        level_labels = ["3000", "5000", "8000", "12000"],
        numeric      = True,
        knee_out     = "sensitivity_knee_dv.pdf",
    ),
    dict(
        name         = "dvbudget",
        front_csv    = "sensitivity_dvbudget_fronts.csv",
        factor_col   = "dv_budget",
        factor_label = "Vehicle ΔV budget [m/s]",
        levels       = ["1500", "3000", "5000", "8000"],
        level_labels = ["1500", "3000", "5000", "8000"],
        numeric      = True,
        knee_out     = "sensitivity_knee_dvbudget.pdf",
    ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_knee_df(pf, factor_col):
    """
    Given a fronts DataFrame, compute one knee distance per
    (factor_level, algorithm, trial).  Returns a new DataFrame with columns:
    {factor_col}, algorithm, trial, knee_dist.
    """
    pf = pf.copy()
    pf["_dist"] = (pf[NORM_COLS] ** 2).sum(axis=1) ** 0.5

    rows = []
    for keys, grp in pf.groupby([factor_col, "algorithm", "trial"]):
        fval, alg, trial = keys
        knee_dist = grp["_dist"].min()
        rows.append({factor_col: str(fval), "algorithm": alg,
                     "trial": trial, "knee_dist": knee_dist})
    return pd.DataFrame(rows)


def iqr_stats(vals):
    v = np.asarray(vals, dtype=float)
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


legend_handles = [
    mlines.Line2D([], [], color=COLOURS[a], linewidth=2.5,
                  marker="o", markersize=7, label=a)
    for a in ALG_NAMES
]

# ═════════════════════════════════════════════════════════════════════════════
#  Knee distance trend — one row, one panel per sub-experiment
# ═════════════════════════════════════════════════════════════════════════════

n_se     = len(SUBEXPS)
fig_knee, axes_knee = plt.subplots(1, n_se, figsize=(10 * n_se, 9), sharey=False)

for col, se in enumerate(SUBEXPS):
    path = os.path.join(OUT, se["front_csv"])
    ax   = axes_knee[col]

    if not os.path.exists(path):
        ax.set_visible(False)
        print(f"[skip] {path} not found")
        continue

    pf       = pd.read_csv(path)
    fcol     = se["factor_col"]
    pf[fcol] = pf[fcol].astype(str)
    kdf      = compute_knee_df(pf, fcol)
    levels   = se["levels"]
    lv_labels = se["level_labels"]

    if se["numeric"]:
        x_vals = [float(lv) for lv in levels]
        for alg in ALG_NAMES:
            medians, q25s, q75s = [], [], []
            for lv in levels:
                vals = kdf[(kdf[fcol] == lv) & (kdf["algorithm"] == alg)]["knee_dist"].values
                if len(vals) == 0:
                    medians.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
                else:
                    med, q25, q75 = iqr_stats(vals)
                    medians.append(med); q25s.append(q25); q75s.append(q75)
            ax.plot(x_vals, medians, marker="o", color=COLOURS[alg],
                    linewidth=2, markersize=6, label=alg)
            ax.fill_between(x_vals, q25s, q75s, color=COLOURS[alg], alpha=0.15)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(lv_labels, fontsize=FS_TICK, rotation=30, ha="right")
    else:
        n_lv    = len(levels)
        width   = 0.18
        offsets = np.linspace(-(len(ALG_NAMES) - 1) / 2 * width,
                               (len(ALG_NAMES) - 1) / 2 * width, len(ALG_NAMES))
        for ai, alg in enumerate(ALG_NAMES):
            for li, lv in enumerate(levels):
                vals = kdf[(kdf[fcol] == lv) & (kdf["algorithm"] == alg)]["knee_dist"].values
                if len(vals) == 0:
                    continue
                pos = li + offsets[ai]
                bp  = ax.boxplot(vals, positions=[pos], widths=width * 0.85,
                                 patch_artist=True,
                                 medianprops=dict(color="black", linewidth=1.5))
                for patch in bp["boxes"]:
                    patch.set_facecolor(COLOURS[alg])
                    patch.set_alpha(0.7)
        ax.set_xticks(range(n_lv))
        ax.set_xticklabels(lv_labels, fontsize=FS_TICK, rotation=30, ha="right")

    ax.set_xlabel(se["factor_label"], fontsize=FS)
    ax.set_ylabel("Distance to ideal", fontsize=FS)
    ax.set_title(se["factor_label"], fontsize=FS_TITLE)
    ax.autoscale(axis="y", tight=False)
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.08
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=FS_TICK)

fig_knee.legend(
    handles=[mlines.Line2D([], [], color=COLOURS[a], linewidth=2.5,
                           marker="o", markersize=7, label=a) for a in ALG_NAMES],
    fontsize=FS - 2, loc="upper center",
    bbox_to_anchor=(0.5, 1.06), ncols=4, frameon=False
)
fig_knee.subplots_adjust(top=0.85, wspace=0.35, bottom=0.18)
out_knee = os.path.join(OUT, "sensitivity_knee_combined.pdf")
fig_knee.savefig(out_knee, bbox_inches="tight")
plt.close(fig_knee)
print(f"Saved: {out_knee}")
print("\nAll knee plots complete.")
