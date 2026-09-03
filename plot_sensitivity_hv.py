"""
plot_sensitivity_hv.py — HV sensitivity figure matching plot_pareto.py style.

Numeric sub-experiments → line plot with median + IQR shading.
Categorical sub-experiments → grouped box plots.
All panels combined in one figure with shared legend at top.

Run: python3 plot_sensitivity_hv.py
"""

import os, glob
import numpy as np
import pandas as pd
import h5py
import moocore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bcr_model import objectives3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

H5_FILE = "outputs/pure_sensitivity_results.h5"
RES_DIR = "outputs/pure_results"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ALGOS = {
    "mdls":       "MDLS",
    "nsga3rk":    "NSGA-III",
    "nsga3rk_ot": "NSGA-III-T",
}

COLOURS = {
    "mdls":       "#1f77b4",
    "nsga3rk":    "#ff7f0e",
    "nsga3rk_ot": "#d62728",
}

FS      = 18
FS_TICK = 15

SUBEXPS = [
    dict(
        key     = "size",
        title   = "Number of demands",
        xlabel  = "Number of demands",
        levels  = ["size_10", "size_50", "size_100", "size_150", "size_200"],
        labels  = [10, 50, 100, 150, 200],
        numeric = True,
    ),
    dict(
        key     = "disttype",
        title   = "Demand distribution",
        xlabel  = "Demand distribution",
        levels  = ["disttype_normal", "disttype_uniform"],
        labels  = ["Normal", "Uniform"],
        numeric = False,
    ),
    dict(
        key     = "dv",
        title   = "ΔV threshold [m/s]",
        xlabel  = "ΔV threshold [m/s]",
        levels  = ["dv_3000", "dv_5000", "dv_8000", "dv_12000"],
        labels  = [3000, 5000, 8000, 12000],
        numeric = True,
    ),
    dict(
        key     = "dvbudget",
        title   = "Vehicle ΔV budget [m/s]",
        xlabel  = "Vehicle ΔV budget [m/s]",
        levels  = ["dvbudget_1500", "dvbudget_3000", "dvbudget_5000",
                   "dvbudget_8000", "dvbudget_10000"],
        labels  = [1500, 3000, 5000, 8000, 10000],
        numeric = True,
    ),
]

# ---------------------------------------------------------------------------
# Normalisation (99th pct from pure_results)
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

f1_max = (all_data["f1_dv"]               / REF_DV).max()
f2_max = (all_data["f2_unrecovered_value"] / REF_F2).max()
f3_max = (all_data["f3_vehicles"]          / REF_F3).max()
REF_POINT = np.array([f1_max * 1.1, f2_max * 1.1, f3_max * 1.1])

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
        fronts.append(objectives3(grp[trial_key][:]))
    return fronts


def compute_hv(F):
    Fn = F.copy().astype(float)
    Fn[:, 0] /= REF_DV
    Fn[:, 1] /= REF_F2
    Fn[:, 2] /= REF_F3
    Fn = Fn[np.all(Fn < REF_POINT, axis=1)]
    return moocore.hypervolume(Fn, ref=REF_POINT) if len(Fn) > 0 else 0.0


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------

if not os.path.exists(H5_FILE):
    raise FileNotFoundError(f"{H5_FILE} not found")

algo_keys = list(ALGOS.keys())

with h5py.File(H5_FILE, "r") as fid:
    # global max for normalisation
    global_max = 0.0
    cached = []
    for se in SUBEXPS:
        all_hvs = {}
        for algo in algo_keys:
            all_hvs[algo] = [
                [compute_hv(F) for F in load_fronts(fid, algo, lv)]
                for lv in se["levels"]
            ]
            for trial_hvs in all_hvs[algo]:
                if trial_hvs:
                    q95 = np.percentile(trial_hvs, 95)
                    global_max = max(global_max, q95)
        cached.append((se, all_hvs))

    hv_max = global_max if global_max > 0 else 1.0

    # ---------------------------------------------------------------------------
    # Combined figure
    # ---------------------------------------------------------------------------

    n_se = len(SUBEXPS)
    fig, axes = plt.subplots(1, n_se, figsize=(4.5 * n_se, 4.5))

    for ax, (se, all_hvs) in zip(axes, cached):
        levels  = se["levels"]
        labels  = se["labels"]

        if se["numeric"]:
            x = np.array(labels, dtype=float)
            for algo in algo_keys:
                vals = [[h / hv_max for h in all_hvs[algo][li]]
                        for li in range(len(levels))]
                medians = np.array([np.median(v) if v else 0.0 for v in vals])
                q25     = np.array([np.percentile(v, 25) if v else 0.0 for v in vals])
                q75     = np.array([np.percentile(v, 75) if v else 0.0 for v in vals])
                q75     = np.minimum(q75, 1.05)
                ax.plot(x, medians, marker="o", color=COLOURS[algo],
                        linewidth=2, markersize=5, label=ALGOS[algo])
                ax.fill_between(x, q25, q75, color=COLOURS[algo], alpha=0.2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(l) for l in labels], fontsize=FS_TICK)

        else:
            n_algos  = len(algo_keys)
            n_levels = len(levels)
            width    = 0.8 / n_algos
            offsets  = np.linspace(-(n_algos - 1) / 2 * width,
                                    (n_algos - 1) / 2 * width, n_algos)
            for ai, algo in enumerate(algo_keys):
                positions = [i + 1 + offsets[ai] for i in range(n_levels)]
                data_per_level = [
                    [h / hv_max for h in all_hvs[algo][li]] or [0.0]
                    for li in range(n_levels)
                ]
                bp = ax.boxplot(data_per_level, positions=positions,
                                widths=width * 0.85, patch_artist=True,
                                medianprops=dict(color="black", linewidth=1.5))
                for patch in bp["boxes"]:
                    patch.set_facecolor(COLOURS[algo])
                    patch.set_alpha(0.7)
                for w in bp["whiskers"]:
                    w.set(color=COLOURS[algo], linewidth=1.0)
                for c in bp["caps"]:
                    c.set(color=COLOURS[algo], linewidth=1.0)
                for fl in bp["fliers"]:
                    fl.set(marker="o", color=COLOURS[algo], alpha=0.5, markersize=4)
            ax.set_xticks(range(1, n_levels + 1))
            ax.set_xticklabels(labels, fontsize=FS_TICK)

        ax.set_title(se["title"], fontsize=FS)
        ax.set_xlabel(se["xlabel"], fontsize=FS)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.grid(False)
        if ax is axes[0]:
            ax.set_ylabel("Hypervolume (normalised)", fontsize=FS)

    # shared legend at top
    handles = [Line2D([0], [0], color=COLOURS[a], marker="o", linewidth=2,
                      markersize=5, label=ALGOS[a]) for a in algo_keys]
    fig.legend(handles=handles, loc="upper center", ncol=len(algo_keys),
               fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "sensitivity_hv_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

print("Done.")
