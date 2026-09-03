"""
plot_sensitivity.py — OFAT sensitivity analysis plots.

For each of 3 sub-experiments produces two PDFs:
  A) HV trend  — median HV ± IQR vs factor level, one line per algorithm
  B) Shadow KDE — KDE contours of ΔV vs unassigned time, one panel per factor level

Run: python3 plot_sensitivity.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

OUT = "outputs"

COLOURS = {
    "MDLS":     "#1f77b4",
    "NSGA-III": "#ff7f0e",
    "MOPSO-CD": "#d62728",
}
ALG_NAMES = ["MDLS", "NSGA-III", "MOPSO-CD"]

FS       = 29
FS_TICK  = 29
FS_TITLE = 29

MIN_KDE_POINTS = 60   # below this, fall back to scatter

# ── Sub-experiment configuration ─────────────────────────────────────────────

SUBEXPS = [
    dict(
        name         = "size",
        hv_csv       = "sensitivity_size.csv",
        front_csv    = "sensitivity_size_fronts.csv",
        factor_col   = "num_demands",
        factor_label = "Number of demands",
        levels       = [10, 50, 100, 150, 200],
        level_labels = ["10", "50", "100", "150", "200"],
        numeric      = True,
        hv_out       = "sensitivity_hv_size.pdf",
        shadow_out   = "sensitivity_shadow_size.pdf",
    ),
    dict(
        name         = "disttype",
        hv_csv       = "sensitivity_disttype.csv",
        front_csv    = "sensitivity_disttype_fronts.csv",
        factor_col   = "disttype",
        factor_label = "Demand distribution",
        levels       = ["normal", "uniform"],
        level_labels = ["Normal", "Uniform"],
        numeric      = False,
        hv_out       = "sensitivity_hv_disttype.pdf",
        shadow_out   = "sensitivity_shadow_disttype.pdf",
    ),
    dict(
        name         = "dv",
        hv_csv       = "sensitivity_dv.csv",
        front_csv    = "sensitivity_dv_fronts.csv",
        factor_col   = "deltaV_dist",
        factor_label = "ΔV threshold [m/s]",
        levels       = [3000, 5000, 8000, 12000],
        level_labels = ["3000", "5000", "8000", "12000"],
        numeric      = True,
        hv_out       = "sensitivity_hv_dv.pdf",
        shadow_out   = "sensitivity_shadow_dv.pdf",
    ),
    dict(
        name         = "dvbudget",
        hv_csv       = "sensitivity_dvbudget.csv",
        front_csv    = "sensitivity_dvbudget_fronts.csv",
        factor_col   = "dv_budget",
        factor_label = "Vehicle ΔV budget [m/s]",
        levels       = [1500, 3000, 5000, 8000],
        level_labels = ["1500", "3000", "5000", "8000"],
        numeric      = True,
        hv_out       = "sensitivity_hv_dvbudget.pdf",
        shadow_out   = "sensitivity_shadow_dvbudget.pdf",
    ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _iqr_median(vals):
    """Return (median, q25, q75) for a 1-D array-like."""
    v = np.asarray(vals, dtype=float)
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def plot_kde(ax, x, y, colour):
    """KDE contours if enough points, scatter fallback otherwise."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return
    if len(x) < MIN_KDE_POINTS:
        ax.scatter(x, y, c=colour, s=18, alpha=0.45, linewidths=0)
        return
    try:
        kde    = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xg     = np.linspace(x.min(), x.max(), 120)
        yg     = np.linspace(y.min(), y.max(), 120)
        X, Y   = np.meshgrid(xg, yg)
        Z      = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        z_flat = Z[Z > Z.max() * 0.05]
        if len(z_flat) == 0:
            ax.scatter(x, y, c=colour, s=18, alpha=0.45, linewidths=0)
            return
        levels = np.unique(np.percentile(z_flat, [40, 65, 82, 93, 98]))
        if len(levels) < 2:
            ax.scatter(x, y, c=colour, s=18, alpha=0.45, linewidths=0)
            return
        ax.contour(X, Y, Z, levels=levels, colors=[colour], alpha=0.85,
                   linewidths=1.5)
    except Exception:
        ax.scatter(x, y, c=colour, s=18, alpha=0.45, linewidths=0)


legend_handles = [
    mlines.Line2D([], [], color=COLOURS[a], linewidth=2.5,
                  marker="o", markersize=7, label=a)
    for a in ALG_NAMES
]

# ═════════════════════════════════════════════════════════════════════════════
#  Plot A: combined HV trend — one row, one panel per sub-experiment
# ═════════════════════════════════════════════════════════════════════════════

n_se   = len(SUBEXPS)

# Compute global HV max across all sub-experiments for consistent normalisation
_hv_global_max = 0.0
for se in SUBEXPS:
    p = os.path.join(OUT, se["hv_csv"])
    if os.path.exists(p):
        _hv_global_max = max(_hv_global_max, pd.read_csv(p)["hypervolume"].max())
_hv_global_max = _hv_global_max if _hv_global_max > 0 else 1.0

fig_hv, axes_hv = plt.subplots(1, n_se, figsize=(10 * n_se, 9), sharey=False)
fig_hv.subplots_adjust(wspace=0.35)

for col, se in enumerate(SUBEXPS):
    hv_path = os.path.join(OUT, se["hv_csv"])
    ax      = axes_hv[col]

    if not os.path.exists(hv_path):
        ax.set_visible(False)
        print(f"[skip] {hv_path} not found")
        continue

    hv_df      = pd.read_csv(hv_path)
    hv_df["hypervolume"] = hv_df["hypervolume"] / _hv_global_max
    factor_col = se["factor_col"]
    levels     = [str(lv) for lv in se["levels"]]
    lv_labels  = se["level_labels"]

    if se["numeric"]:
        x_vals = [float(lv) for lv in levels]
        for alg in ALG_NAMES:
            medians, q25s, q75s = [], [], []
            for lv in levels:
                vals = hv_df[
                    (hv_df[factor_col].astype(str) == lv) &
                    (hv_df["algorithm"] == alg)
                ]["hypervolume"].values
                med, q25, q75 = _iqr_median(vals) if len(vals) > 0 else (0, 0, 0)
                medians.append(med); q25s.append(q25); q75s.append(q75)
            ax.plot(x_vals, medians, marker="o", color=COLOURS[alg],
                    linewidth=2, markersize=6, label=alg)
            ax.fill_between(x_vals, q25s, q75s, color=COLOURS[alg], alpha=0.15)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(lv_labels, fontsize=FS_TICK, rotation=30, ha="right")
    else:
        n_lv    = len(levels)
        width   = 0.18
        offsets = np.linspace(-(len(ALG_NAMES)-1)/2*width,
                               (len(ALG_NAMES)-1)/2*width, len(ALG_NAMES))
        for ai, alg in enumerate(ALG_NAMES):
            for li, lv in enumerate(levels):
                vals = hv_df[
                    (hv_df[factor_col].astype(str) == lv) &
                    (hv_df["algorithm"] == alg)
                ]["hypervolume"].values
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
    ax.set_ylabel("Hypervolume (normalised)", fontsize=FS)
    ax.set_title(se["factor_label"], fontsize=FS_TITLE)
    ax.autoscale(axis="y", tight=False)
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.08
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.grid(False)
    ax.tick_params(labelsize=FS_TICK)

fig_hv.legend(
    handles=[mlines.Line2D([], [], color=COLOURS[a], linewidth=2.5,
                           marker="o", markersize=7, label=a) for a in ALG_NAMES],
    fontsize=FS - 2, loc="upper center",
    bbox_to_anchor=(0.5, 1.06), ncols=4, frameon=False
)
fig_hv.subplots_adjust(top=0.85, wspace=0.35, bottom=0.18)
out_hv = os.path.join(OUT, "sensitivity_hv_combined.pdf")
fig_hv.savefig(out_hv, bbox_inches="tight")
plt.close(fig_hv)
print(f"Saved: {out_hv}")

# ── Plot B: Shadow KDE — one PDF per sub-experiment (unchanged) ───────────────

for se in SUBEXPS:
    front_path = os.path.join(OUT, se["front_csv"])
    if not os.path.exists(front_path):
        print(f"[skip] {front_path} not found")
        continue

    pf         = pd.read_csv(front_path)
    factor_col = se["factor_col"]
    levels     = [str(lv) for lv in se["levels"]]
    lv_labels  = se["level_labels"]
    n_lv       = len(levels)

    dv_min = pf["f1_dv"].min(); dv_max = pf["f1_dv"].max()
    pf["_dv_norm"] = (pf["f1_dv"] - dv_min) / max(dv_max - dv_min, 1e-10)

    fig, axes = plt.subplots(1, n_lv, figsize=(8 * n_lv, 5), sharey=False)
    if n_lv == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.05)
    fig.suptitle(f"ΔV vs Unrecovered Value — {se['factor_label']}",
                 fontsize=FS_TITLE + 2, y=1.02)

    for i, (ax, lv, lv_lbl) in enumerate(zip(axes, levels, lv_labels)):
        sub = pf[pf[factor_col].astype(str) == lv]
        for alg in ALG_NAMES:
            alg_df = sub[sub["algorithm"] == alg]
            plot_kde(ax,
                     alg_df["f2_unrecovered_value"].values,
                     alg_df["_dv_norm"].values,
                     COLOURS[alg])
        ax.set_title(lv_lbl, fontsize=FS_TITLE)
        ax.set_xlabel("Unrecovered value [USD]", fontsize=FS)
        if i == 0:
            ax.set_ylabel("ΔV (normalised)", fontsize=FS)
        ax.grid(False)
        ax.tick_params(labelsize=FS_TICK)
        inst_dv = sub["_dv_norm"]
        if len(inst_dv) > 0:
            pad = (inst_dv.max() - inst_dv.min()) * 0.05
            ax.set_ylim(inst_dv.min() - pad, inst_dv.max() + pad)

    axes[-1].legend(handles=legend_handles, fontsize=FS - 2, loc="upper right")
    out_b = os.path.join(OUT, se["shadow_out"])
    fig.savefig(out_b, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_b}")

print("\nAll sensitivity plots complete.")
