"""
plot_sensitivity_oat.py — box plots of HV for each OAT parameter sweep.

Uses the same objective normalisation and HV reference point as plot_pure_hv.py
so sensitivity HV values are directly comparable to the main experiment plots.
HV is then normalised per-algorithm by the max HV across all parameter levels
(matching the per-scenario normalisation in plot_pure_hv.py).
"""

import os, glob
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import moocore

H5_FILE  = "outputs/sensitivity_oat.h5"
RES_DIR  = "outputs/pure_results"   # for computing normalisation factors
OUT_DIR  = "outputs"

ALGOS = {
    "mdls":        "MDLS",
    "nsga3rk":     "NSGA-III",
    "nsga3rk_ot":  "NSGA-III-T",
}

COLOURS = {
    "mdls":        "#1f77b4",
    "nsga3rk":     "#d62728",
    "nsga3rk_ot":  "#2ca02c",
}

PARAMS = {
    "mdls": [
        ("shift",          "Timing shift (days)", [5.0, 15.0, 30.0, 60.0]),
        ("top_pct",        "Top-leg fraction",    [0.25, 0.5, 0.75, 1.0]),
    ],
    "nsga3rk": [
        ("sbx_eta",        "SBX η",              [5.0, 10.0, 20.0, 30.0]),
        ("pm_eta",         "PM η",               [5.0, 10.0, 20.0, 30.0]),
        ("crossover_prob", "Crossover prob",      [0.5, 0.7, 0.9, 1.0]),
    ],
    "nsga3rk_ot": [
        ("sbx_eta",        "SBX η",              [5.0, 10.0, 20.0, 30.0]),
        ("pm_eta",         "PM η",               [5.0, 10.0, 20.0, 30.0]),
        ("shift",          "Timing shift (days)", [5.0, 15.0, 30.0, 60.0]),
        ("top_pct",        "Top-leg fraction",    [0.25, 0.5, 0.75, 1.0]),
    ],
}

NOMINAL = {
    "shift":          15.0,
    "top_pct":        0.5,
    "sbx_eta":        20.0,
    "pm_eta":         20.0,
    "crossover_prob": 0.9,
}

# ---------------------------------------------------------------------------
# Build normalisation factors from pure results (same as plot_pure_hv.py)
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

print(f"Normalisation: dv={REF_DV:.0f}  f2={REF_F2:.3e}  f3={REF_F3:.0f}")
print(f"HV ref point (normalised): {REF_POINT}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def level_key(v):
    return str(v).replace(".", "p")


def load_front(fid, algo, param, level):
    path = f"{algo}/{param}/{level_key(level)}"
    if path not in fid:
        return []
    grp = fid[path]
    fronts = []
    for trial_key in sorted(grp.keys()):
        data = grp[trial_key][:]
        if data.ndim == 2 and data.shape[0] == 3 and data.shape[1] != 3:
            data = data.T  # Julia HDF5 transpose
        fronts.append(data)
    return fronts


def normalise_front(F):
    """Apply same normalisation as plot_pure_hv.py."""
    Fn = F.copy().astype(float)
    Fn[:, 0] /= REF_DV
    Fn[:, 1] /= REF_F2
    Fn[:, 2] /= REF_F3
    return Fn


def compute_hvs(fronts):
    hvs = []
    for F in fronts:
        if len(F) == 0:
            continue
        Fn = normalise_front(F)
        Fn = Fn[np.all(Fn < REF_POINT, axis=1)]
        hvs.append(moocore.hypervolume(Fn, ref=REF_POINT) if len(Fn) > 0 else 0.0)
    return hvs


def compute_knee_dists(fronts):
    """Min L2 distance to ideal [0,0,0] in normalised space (same as plot_pure_knee.py)."""
    dists = []
    for F in fronts:
        if len(F) == 0:
            continue
        Fn = normalise_front(F)
        d = np.sqrt(np.sum(Fn ** 2, axis=1))
        dists.append(d.min())
    return dists


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

if not os.path.exists(H5_FILE):
    raise FileNotFoundError(f"{H5_FILE} not found — run run_sensitivity_oat.sh first")

with h5py.File(H5_FILE, "r") as fid:
    # Global HV max across ALL algorithms (mirrors per-scenario norm in plot_pure_hv.py)
    global_hvs = []
    for algo in ALGOS:
        if algo not in fid:
            continue
        for param, _, levels in PARAMS[algo]:
            for level in levels:
                global_hvs.extend(compute_hvs(load_front(fid, algo, param, level)))
    hv_max = max(global_hvs) if global_hvs else 1.0
    print(f"Global HV max (for normalisation): {hv_max:.3e}")

    for algo, algo_label in ALGOS.items():
        if algo not in fid:
            print(f"No data for {algo} — skipping")
            continue

        params    = PARAMS[algo]
        n_params  = len(params)
        colour    = COLOURS[algo]

        fig, axes = plt.subplots(1, n_params, figsize=(4.5 * n_params, 4.5), sharey=True)
        if n_params == 1:
            axes = [axes]

        for ax, (param, xlabel, levels) in zip(axes, params):
            data_per_level = []
            labels = []
            for level in levels:
                hvs = [h / hv_max for h in compute_hvs(load_front(fid, algo, param, level))]
                data_per_level.append(hvs if hvs else [0.0])
                label = str(int(level)) if level == int(level) else str(level)
                nom   = NOMINAL.get(param)
                labels.append(f"{label}*" if nom is not None and abs(level - nom) < 1e-9 else label)

            bp = ax.boxplot(data_per_level, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(colour)
                patch.set_alpha(0.7)
            for whisker in bp["whiskers"]:
                whisker.set(color=colour, linewidth=1.0)
            for cap in bp["caps"]:
                cap.set(color=colour, linewidth=1.0)
            for flier in bp["fliers"]:
                flier.set(marker="o", color=colour, alpha=0.5, markersize=4)

            ax.set_xticklabels(labels, fontsize=22)
            ax.set_xlabel(xlabel, fontsize=22)
            
            if ax is axes[0]:
                ax.set_ylabel("Hypervolume", fontsize=22)
            
            ax.tick_params(axis="y", labelsize=22)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.suptitle(f"{algo_label} — OAT sensitivity", fontsize=22, y=1.02)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"sensitivity_oat_{algo}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    # -------------------------------------------------------------------
    # Knee-point distance plots
    # -------------------------------------------------------------------
    global_knees = []
    for algo in ALGOS:
        if algo not in fid:
            continue
        for param, _, levels in PARAMS[algo]:
            for level in levels:
                global_knees.extend(compute_knee_dists(load_front(fid, algo, param, level)))
    knee_max = max(global_knees) if global_knees else 1.0
    print(f"Global knee-dist max (for normalisation): {knee_max:.3e}")

    for algo, algo_label in ALGOS.items():
        if algo not in fid:
            continue

        params    = PARAMS[algo]
        n_params  = len(params)
        colour    = COLOURS[algo]

        fig, axes = plt.subplots(1, n_params, figsize=(4.5 * n_params, 4.5), sharey=True)
        if n_params == 1:
            axes = [axes]

        for ax, (param, xlabel, levels) in zip(axes, params):
            data_per_level = []
            labels = []
            for level in levels:
                kds = [k / knee_max for k in compute_knee_dists(load_front(fid, algo, param, level))]
                data_per_level.append(kds if kds else [0.0])
                label = str(int(level)) if level == int(level) else str(level)
                nom   = NOMINAL.get(param)
                labels.append(f"{label}*" if nom is not None and abs(level - nom) < 1e-9 else label)

            bp = ax.boxplot(data_per_level, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(colour)
                patch.set_alpha(0.7)
            for whisker in bp["whiskers"]:
                whisker.set(color=colour, linewidth=1.0)
            for cap in bp["caps"]:
                cap.set(color=colour, linewidth=1.0)
            for flier in bp["fliers"]:
                flier.set(marker="o", color=colour, alpha=0.5, markersize=4)

            ax.set_xticklabels(labels, fontsize=22)
            ax.set_xlabel(xlabel, fontsize=22)

            if ax is axes[0]:
                ax.set_ylabel("Knee point", fontsize=22)

            ax.tick_params(axis="y", labelsize=22)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.suptitle(f"{algo_label} — OAT sensitivity, knee distance", fontsize=22, y=1.02)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"sensitivity_oat_knee_{algo}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

print("Done.")
