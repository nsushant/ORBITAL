"""
plot_pure_hv.py — HV box plots for pure-comparison experiments.

Reads CSVs from outputs/pure_results/, computes normalised hypervolume per
(scenario, algo, trial), and produces box plots grouped by scenario.

Run: python3 plot_pure_hv.py
"""

import os, glob
import numpy as np
import pandas as pd
import moocore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RES_DIR  = "outputs/pure_results"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = {
    "tight_normal":  "Scenario 1",
    "loose_uniform": "Scenario 2",
    "tight_low_dv":  "Scenario 3",
    "loose_high_dv": "Scenario 4",
}

ALGOS = {
    "mdls":       "MDLS",
    "nsga3rk":    "NSGA-III",
    "nsga3rk_ot": "NSGA-III-T",
}

COLOURS = {
    "mdls":       "#1f77b4",
    "nsga3rk":    "#d62728",
    "nsga3rk_ot": "#2ca02c",
}

DV_FILTER = np.inf   # no filter — all solutions included

FS      = 28
FS_TICK = 24

# HV reference point (slightly above nadir, shared across all scenarios)
# Set after loading data
REF_POINT = None

# ---------------------------------------------------------------------------
# Load all CSVs
# ---------------------------------------------------------------------------

records = []
for scenario in SCENARIOS:
    for algo in ALGOS:
        pattern = os.path.join(RES_DIR, f"{algo}_{scenario}_*.csv")
        for path in sorted(glob.glob(pattern)):
            trial = int(os.path.basename(path).split("_")[-1].replace(".csv", ""))
            df = pd.read_csv(path)
            df = df[df["f1_dv"] < DV_FILTER]
            if df.empty:
                continue
            df["scenario"] = scenario
            df["algo"]     = algo
            df["trial"]    = trial
            records.append(df)

if not records:
    raise FileNotFoundError(f"No valid CSVs in {RES_DIR}")

data = pd.concat(records, ignore_index=True)

# ---------------------------------------------------------------------------
# Normalise objectives
# ---------------------------------------------------------------------------

REF_DV = data["f1_dv"].quantile(0.99)
REF_F2 = data["f2_unrecovered_value"].quantile(0.99)
REF_F3 = data["f3_vehicles"].quantile(0.99)

data["f1_norm"] = data["f1_dv"]               / REF_DV
data["f2_norm"] = data["f2_unrecovered_value"] / REF_F2
data["f3_norm"] = data["f3_vehicles"]          / REF_F3

# Reference point for HV: 1.1× the max normalised value across all data
REF_POINT = np.array([
    data["f1_norm"].max() * 1.1,
    data["f2_norm"].max() * 1.1,
    data["f3_norm"].max() * 1.1,
])

print(f"Ref point (normalised): {REF_POINT}")
print(f"Normalisation: dv={REF_DV:.0f}  f2={REF_F2:.3e}  f3={REF_F3:.0f}")

# ---------------------------------------------------------------------------
# Compute HV per (scenario, algo, trial)
# ---------------------------------------------------------------------------

hv_records = []
for (scenario, algo, trial), grp in data.groupby(["scenario", "algo", "trial"]):
    F = grp[["f1_norm", "f2_norm", "f3_norm"]].values
    # clip to reference point (moocore requires all points dominated by ref)
    F = F[np.all(F < REF_POINT, axis=1)]
    if len(F) == 0:
        hv = 0.0
    else:
        hv = moocore.hypervolume(F, ref=REF_POINT)
    hv_records.append({"scenario": scenario, "algo": algo, "trial": trial, "hv": hv})

hv_df = pd.DataFrame(hv_records)
hv_df["hv"] = hv_df.groupby("scenario")["hv"].transform(lambda x: x / x.max() if x.max() > 0 else x)
print("\nMedian HV per scenario/algo:")
print(hv_df.groupby(["scenario", "algo"])["hv"].median().unstack().to_string())

# ---------------------------------------------------------------------------
# Box plots — one figure per scenario
# ---------------------------------------------------------------------------

algo_keys   = list(ALGOS.keys())
algo_labels = [ALGOS[a] for a in algo_keys]

for scenario, scenario_label in SCENARIOS.items():
    sub = hv_df[hv_df["scenario"] == scenario]
    if sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(6, 5))

    box_data = [sub[sub["algo"] == a]["hv"].values for a in algo_keys]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))

    for patch, algo in zip(bp["boxes"], algo_keys):
        patch.set_facecolor(COLOURS[algo])
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(algo_keys) + 1))
    ax.set_xticklabels(algo_labels, fontsize=FS_TICK)
    ax.set_ylabel("Hypervolume (normalised, 0–1)", fontsize=FS - 2)
    ax.set_title(scenario_label, fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"pure_hv_{scenario}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# Combined figure — all scenarios side by side
# ---------------------------------------------------------------------------

n_scen = len(SCENARIOS)
fig, axes = plt.subplots(1, n_scen, figsize=(5 * n_scen, 5), sharey=True)

for ax, (scenario, scenario_label) in zip(axes, SCENARIOS.items()):
    sub = hv_df[hv_df["scenario"] == scenario]
    box_data = [sub[sub["algo"] == a]["hv"].values for a in algo_keys]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, algo in zip(bp["boxes"], algo_keys):
        patch.set_facecolor(COLOURS[algo])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(algo_keys) + 1))
    ax.set_xticklabels(algo_labels, fontsize=FS_TICK, rotation=15, ha="right")
    ax.set_title(scenario_label, fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(False)
    if ax is axes[0]:
        ax.set_ylabel("Hypervolume", fontsize=FS - 2)

plt.tight_layout()
out = os.path.join(OUT_DIR, "pure_hv_combined.pdf")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"  saved {out}")
