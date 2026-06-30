"""
plot_3d_pareto.py — Static 3D Pareto front plots (PDF) in normalised objective space.

Produces:
  outputs/pareto_3d_scenarios.pdf  — 2×2 grid, one panel per demand scenario (best trial)
  outputs/pareto_3d_starlink.pdf   — single panel for the Starlink propulsion study (best trial)

Run: python plot_3d_pareto.py
"""

import os
import numpy as np
import pandas as pd
import moocore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

OUT = "outputs"

COLOURS = {
    "MDLS":     "#1f77b4",
    "NSGA-III": "#ff7f0e",
    "MOPSO-CD": "#d62728",
}
ALG_NAMES = ["MDLS", "NSGA-III", "MOPSO-CD"]
REF       = [1.1, 1.1, 1.1]
NORM_COLS = ["f1_dv_norm", "f2_unassigned_norm", "f3_vehicles_norm"]

SCENARIO_LABELS = {
    "tight_normal":   "Scenario 1\n(tight, normal)",
    "loose_uniform":  "Scenario 2\n(loose, uniform)",
    "tight_low_dv":   "Scenario 3\n(tight, low ΔV)",
    "loose_high_dv":  "Scenario 4\n(loose, high ΔV)",
}
SCENARIOS = ["tight_normal", "loose_uniform", "tight_low_dv", "loose_high_dv"]

FS       = 9
FS_TITLE = 10

# ── Helper: best trial per (group_col, algorithm) by HV ──────────────────────

def best_trials_from_hv(hv_df, group_col):
    """Return dict (group, alg) → best trial index."""
    idx = hv_df.groupby([group_col, "algorithm"])["hypervolume"].idxmax()
    best = hv_df.loc[idx][[group_col, "algorithm", "trial"]].set_index([group_col, "algorithm"])
    return best["trial"].to_dict()


def best_trials_inline(pf, group_col):
    """Compute HV inline from normalised cols; return dict (group, alg) → best trial."""
    ref = np.array(REF)
    best = {}
    for (grp, alg, trial), sub in pf.groupby([group_col, "algorithm", "trial"]):
        pts = sub[NORM_COLS].values
        hv  = float(moocore.hypervolume(pts, ref=ref, maximise=False)) if len(pts) > 0 else 0.0
        key = (grp, alg)
        if key not in best or hv > best[key][1]:
            best[key] = (trial, hv)
    return {k: v[0] for k, v in best.items()}


def best_trials_starlink(pf):
    """Compute HV per (algorithm, trial); return dict alg → best trial."""
    ref = np.array(REF)
    best = {}
    for (alg, trial), sub in pf.groupby(["algorithm", "trial"]):
        pts = sub[NORM_COLS].values
        hv  = float(moocore.hypervolume(pts, ref=ref, maximise=False)) if len(pts) > 0 else 0.0
        if alg not in best or hv > best[alg][1]:
            best[alg] = (trial, hv)
    return {alg: v[0] for alg, v in best.items()}


def add_ref_point(ax):
    ax.scatter(*[[r] for r in REF], color="black", s=60, marker="x", zorder=5,
               label="Ref (1.1,1.1,1.1)")
    for dim in range(3):
        start = [REF[0], REF[1], REF[2]]
        end   = [REF[0], REF[1], REF[2]]
        end[dim] = 0.0
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        ax.plot(xs, ys, zs, color="black", linewidth=0.7, linestyle="--", alpha=0.5)


def style_ax(ax, title):
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.2); ax.set_zlim(0, 1.2)
    ax.set_xlabel("f₁ ΔV", fontsize=FS, labelpad=2)
    ax.set_ylabel("f₂ Unassigned", fontsize=FS, labelpad=2)
    ax.set_zlabel("f₃ Vehicles", fontsize=FS, labelpad=2)
    ax.tick_params(labelsize=FS - 1)
    ax.set_title(title, fontsize=FS_TITLE)


# ── Figure 1: 4 demand scenarios ─────────────────────────────────────────────

front_path = os.path.join(OUT, "pareto_fronts.csv")
hv_path    = os.path.join(OUT, "numerical_experiments.csv")

if not os.path.exists(front_path):
    print(f"[skip] {front_path} not found — run aggregate_results.py first")
else:
    pf   = pd.read_csv(front_path)
    pf   = pf[pf["instance"].isin(SCENARIOS)]

    # Best trial selection
    if os.path.exists(hv_path):
        hv_df  = pd.read_csv(hv_path)
        bt     = best_trials_from_hv(hv_df, "instance")
    else:
        bt = best_trials_inline(pf, "instance")

    fig = plt.figure(figsize=(14, 11))
    legend_handles = [Line2D([0],[0], color=COLOURS[a], lw=0, marker="o",
                              markersize=5, label=a) for a in ALG_NAMES]
    legend_handles += [Line2D([0],[0], color="black", lw=0, marker="x",
                               markersize=7, label="Ref (1.1,1.1,1.1)")]

    for i, sc in enumerate(SCENARIOS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        for alg in ALG_NAMES:
            trial = bt.get((sc, alg))
            if trial is None:
                continue
            pts = pf[(pf["instance"] == sc) & (pf["algorithm"] == alg) &
                     (pf["trial"] == trial)]
            ax.scatter(pts["f1_dv_norm"], pts["f2_unassigned_norm"], pts["f3_vehicles_norm"],
                       color=COLOURS[alg], s=8, alpha=0.75)
        add_ref_point(ax)
        style_ax(ax, SCENARIO_LABELS.get(sc, sc))

    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=FS, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.subplots_adjust(bottom=0.08, hspace=0.25, wspace=0.05)

    out1 = os.path.join(OUT, "pareto_3d_scenarios.pdf")
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")


# ── Figure 2: Starlink propulsion study ──────────────────────────────────────

starlink_path = os.path.join(OUT, "starlink_pareto_fronts.csv")

if not os.path.exists(starlink_path):
    print(f"[skip] {starlink_path} not found — run starlink/run_starlink.jl first")
else:
    sl    = pd.read_csv(starlink_path)
    bt_sl = best_trials_starlink(sl)

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(1, 1, 1, projection="3d")

    for alg in ALG_NAMES:
        trial = bt_sl.get(alg)
        if trial is None:
            continue
        pts = sl[(sl["algorithm"] == alg) & (sl["trial"] == trial)]
        ax.scatter(pts["f1_dv_norm"], pts["f2_unassigned_norm"], pts["f3_vehicles_norm"],
                   color=COLOURS[alg], s=12, alpha=0.8, label=alg)

    add_ref_point(ax)
    style_ax(ax, "Starlink Study — best trial per algorithm")

    legend_handles = [Line2D([0],[0], color=COLOURS[a], lw=0, marker="o",
                              markersize=6, label=a) for a in ALG_NAMES]
    legend_handles += [Line2D([0],[0], color="black", lw=0, marker="x",
                               markersize=8, label="Ref (1.1,1.1,1.1)")]
    ax.legend(handles=legend_handles, fontsize=FS, loc="upper right")

    out2 = os.path.join(OUT, "pareto_3d_starlink.pdf")
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")
