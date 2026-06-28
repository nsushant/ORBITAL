"""
plot_pareto.py — Histograms of Pareto front solution distributions in
normalised objective space, using pareto_fronts.csv from numerical experiments.

Run: python plot_pareto.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

OUT = "outputs"

COLOURS = {
    "MDLS":     "#1f77b4",
    "NSGA-III": "#ff7f0e",
    "MOEA/D":   "#2ca02c",
    "PSO":      "#d62728",
}

SCENARIO_LABELS = {
    "tight_normal":   "Scenario 1",
    "loose_uniform":  "Scenario 2",
    "tight_low_dv":   "Scenario 3",
    "loose_high_dv":  "Scenario 4",
}

FS      = 22
FS_TICK = 22
FS_TITLE = 24


OBJECTIVES = [
    ("f1_dv_norm",         "f₁  ΔV  (normalised)"),
    ("f2_unassigned_norm", "f₂  Unassigned time  (normalised)"),
    ("f3_vehicles_norm",   "f₃  Vehicles used  (normalised)"),
]

src = f"{OUT}/pareto_fronts.csv"
if not os.path.exists(src):
    raise FileNotFoundError(f"{src} not found — run numerical_experiments.jl first")

df  = pd.read_csv(src)
alg_names = list(df["algorithm"].unique())
instances = list(df["instance"].unique())

for inst in instances:
    sub = df[df["instance"] == inst]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Pareto Front Distributions — {inst} (normalised)", fontsize=13)

    for ax, (col, label) in zip(axes, OBJECTIVES):
        for alg in alg_names:
            vals = sub[sub["algorithm"] == alg][col]
            ax.hist(vals, bins=30, color=COLOURS.get(alg, "grey"), label=alg,
                    alpha=0.5, edgecolor="none", density=False)
        ax.axvline(1.1, color="black", linestyle="--", linewidth=1, label="Ref (1.1)")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Number of solutions", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = f"{OUT}/pareto_histograms_{inst}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# ── Hypervolume box plots ─────────────────────────────────────────────────────
# ── Knee point box plots ──────────────────────────────────────────────────────
# Knee point per front = solution closest to the normalised ideal [0,0,0],
# i.e. argmin ||f_norm|| over all Pareto points in that front.
# We compute one knee per (instance, nvehicles, algorithm, trial) and box-plot
# the L2 distance to ideal across trials/starts.

src = f"{OUT}/pareto_fronts.csv"
if os.path.exists(src):
    pf = pd.read_csv(src)
    norm_cols = ["f1_dv_norm", "f2_unassigned_norm", "f3_vehicles_norm"]
    alg_names = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]
    instances  = list(pf["instance"].unique())
    group_cols = ["instance", "algorithm", "trial"]
    if "init_nvehicles" in pf.columns:
        group_cols = ["instance", "init_nvehicles", "algorithm", "trial"]

    # Compute L2 distance to ideal for knee point of each front
    pf["_dist"] = (pf[norm_cols] ** 2).sum(axis=1) ** 0.5

    knee_rows = []
    for keys, grp in pf.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        knee = grp.loc[grp["_dist"].idxmin()]
        row["knee_dist"] = knee["_dist"]
        for c in norm_cols:
            row[c] = knee[c]
        knee_rows.append(row)
    knee_df = pd.DataFrame(knee_rows)

    n_inst = len(instances)
    fig, axes = plt.subplots(1, n_inst, figsize=(6 * n_inst, 6), sharey=False)
    if n_inst == 1:
        axes = [axes]
    fig.suptitle("Knee Point Distance to Ideal  (lower = better)", fontsize=FS_TITLE + 2, y=1.01)

    c = 0
    for ax, inst in zip(axes, instances):
        sub    = knee_df[knee_df["instance"] == inst]
        data   = [sub[sub["algorithm"] == a]["knee_dist"].values for a in alg_names]
        colors = [COLOURS.get(a, "grey") for a in alg_names]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(SCENARIO_LABELS.get(inst, inst.replace("_", " ")), fontsize=FS_TITLE)
        ax.set_xticks(range(1, len(alg_names) + 1))
        ax.set_xticklabels(alg_names, fontsize=FS_TICK, rotation=15)
        if c == 0: 
            ax.set_ylabel("L2 distance to ideal (normalised)", fontsize=FS)
        
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.tick_params(labelsize=FS_TICK)
        c+=1 
        
    plt.tight_layout()
    knee_plot = f"{OUT}/knee_boxplots.pdf"
    plt.savefig(knee_plot, bbox_inches="tight")
    plt.close()
    print(f"Saved: {knee_plot}")
else:
    print(f"[skip] {src} not found — skipping knee point plots")

hv_path = f"{OUT}/numerical_experiments.csv"
if os.path.exists(hv_path):
    hv = pd.read_csv(hv_path)
    alg_names = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]
    instances  = list(hv["instance"].unique())
    n_inst     = len(instances)

    fig, axes = plt.subplots(1, n_inst, figsize=(6 * n_inst, 6), sharey=False)
    if n_inst == 1:
        axes = [axes]
    fig.suptitle("Hypervolume Distribution by Instance  (higher = better)",
                 fontsize=FS_TITLE + 2, y=1.01)
    
    c2 = 0 
    for ax, inst in zip(axes, instances):
        sub    = hv[hv["instance"] == inst]
        data   = [sub[sub["algorithm"] == a]["hypervolume"].values for a in alg_names]
        colors = [COLOURS.get(a, "grey") for a in alg_names]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(SCENARIO_LABELS.get(inst, inst.replace("_", " ")), fontsize=FS_TITLE)
        ax.set_xticks(range(1, len(alg_names) + 1))
        ax.set_xticklabels(alg_names, fontsize=FS_TICK, rotation=15)
        if c2 == 0: 
            ax.set_ylabel("Hypervolume (normalised)", fontsize=FS)

        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.tick_params(labelsize=FS_TICK)
        c2 +=1 

    plt.tight_layout()
    hv_plot = f"{OUT}/hv_boxplots.pdf"
    plt.savefig(hv_plot, bbox_inches="tight")
    plt.close()
    print(f"Saved: {hv_plot}")
else:
    print(f"[skip] {hv_path} not found")
