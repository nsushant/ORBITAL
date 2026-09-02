"""
aggregate_results.py — Phase 3 of the numerical experiment pipeline.

Reads all per-trial CSVs from outputs/exp_results/, computes a shared
normalised hypervolume (via moocore) and knee points per scenario, then
generates summary CSVs and plots.

Outputs
-------
outputs/numerical_experiments_hv.csv      ← consumed by plot_results.jl spider plots
outputs/pareto_fronts_all.csv             ← consumed by plot_pareto.py
outputs/hv_boxplots.pdf
outputs/kde_shadow_{scenario}.pdf
"""

import os, glob, argparse
import numpy as np
import pandas as pd
import moocore
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALG_NAMES = ["mdls", "nsga3", "pso"]
ALG_LABELS = {"mdls": "MDLS", "nsga3": "NSGA-III", "pso": "MOPSO-CD"}
SCENARIOS  = ["tight_normal", "loose_uniform", "tight_low_dv", "loose_high_dv"]
N_TRIALS   = 5

# Paths are CLI-overridable so pure / sensitivity results can be aggregated
# without editing this file. Defaults reproduce the original behaviour.
#   e.g.  python aggregate_results.py --res-dir outputs/pure_results --out-dir outputs/pure_results
_ap = argparse.ArgumentParser()
_ap.add_argument("--res-dir",  default="outputs/exp_results", help="dir with per-trial CSVs")
_ap.add_argument("--out-dir",  default="outputs",             help="dir for aggregated outputs")
_ap.add_argument("--n-trials", type=int, default=N_TRIALS)
_args, _ = _ap.parse_known_args()
RES_DIR    = _args.res_dir
OUT_DIR    = _args.out_dir
N_TRIALS   = _args.n_trials
PENALTY    = 1e6
OBJ_COLS   = ["f1_dv", "f2_unrecovered_value", "f3_vehicles"]

ALG_COLORS = {
    "mdls":   "#1f77b4",
    "nsga3":  "#ff7f0e",
    "pso":    "#d62728",  # MOPSO-CD
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load all fronts
# ---------------------------------------------------------------------------

def load_front(scenario, alg, trial):
    path = os.path.join(RES_DIR, f"{alg}_{scenario}_{trial:02d}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    # backwards-compat: rename old column if present
    if "f2_unassigned_time" in df.columns:
        df = df.rename(columns={"f2_unassigned_time": "f2_unrecovered_value"})
    df = df[df["f1_dv"] < PENALTY]
    return df if not df.empty else None

print("Loading fronts …")
# fronts[scenario][alg][trial] = DataFrame or None
fronts = {sc: {alg: {} for alg in ALG_NAMES} for sc in SCENARIOS}
missing = 0
for sc in SCENARIOS:
    for alg in ALG_NAMES:
        for t in range(1, N_TRIALS + 1):
            df = load_front(sc, alg, t)
            fronts[sc][alg][t] = df
            if df is None:
                missing += 1
                print(f"  MISSING: {alg} {sc} trial {t}")

print(f"Loaded. Missing files: {missing}")

# ---------------------------------------------------------------------------
# Per-scenario normalisation: ideal/nadir from ALL algorithms & trials
# ---------------------------------------------------------------------------

print("\nComputing per-scenario reference points …")

# Problem-motivated fixed worst-case bounds (independent of which algorithms ran).
# f1: every demand requires a full depot-return leg at max ΔV budget
# f2: expected do-nothing loss (mixed V1/V2 fleet) + 10% safety margin
# f3: maximum fleet size + small buffer
N_DEMANDS         = 200
V1_ASSET_VAL      = 864_150.0
V2_ASSET_VAL      = 2_280_000.0
V2_FRACTION       = 0.30
DV_BUDGET         = 5000.0
N_VEHICLES        = 20
EXPECTED_AVG_ASSET_VAL = V2_FRACTION * V2_ASSET_VAL + (1 - V2_FRACTION) * V1_ASSET_VAL
FIXED_NADIR  = np.array([N_DEMANDS * DV_BUDGET,                          # f1: 1 000 000 m/s
                          N_DEMANDS * EXPECTED_AVG_ASSET_VAL * 1.10,     # f2: ~407 M USD
                          N_VEHICLES + 5])                               # f3: 25 vehicles

scenario_meta = {}  # sc → {ideal, nadir, ref}
for sc in SCENARIOS:
    all_pts = []
    for alg in ALG_NAMES:
        for t in range(1, N_TRIALS + 1):
            df = fronts[sc][alg][t]
            if df is not None:
                all_pts.append(df[OBJ_COLS].values)
    if not all_pts:
        print(f"  WARNING: no data for scenario {sc}")
        scenario_meta[sc] = {"ideal": np.zeros(3), "nadir": FIXED_NADIR,
                              "range": FIXED_NADIR,  "ref":   np.full(3, 1.1)}
        continue
    all_pts = np.vstack(all_pts)
    ideal = all_pts.min(axis=0)
    nadir = FIXED_NADIR          # problem-motivated, not data-driven
    rng   = np.maximum(nadir - ideal, 1e-10)
    ref   = np.array([1.1, 1.1, 1.1])
    scenario_meta[sc] = {"ideal": ideal, "nadir": nadir, "range": rng, "ref": ref}
    print(f"  {sc}: ideal={ideal.round(2)}  nadir={nadir.round(2)}  (fixed)")

def normalise(df, sc):
    meta = scenario_meta[sc]
    return (df[OBJ_COLS].values - meta["ideal"]) / meta["range"]

# ---------------------------------------------------------------------------
# Compute HV and knee per trial
# ---------------------------------------------------------------------------

print("\nComputing hypervolumes …")

hv_rows    = []
front_rows = []

for sc in SCENARIOS:
    meta = scenario_meta[sc]
    ref  = meta["ref"]
    for alg in ALG_NAMES:
        for t in range(1, N_TRIALS + 1):
            df = fronts[sc][alg][t]
            if df is None:
                hv_rows.append({"instance": sc, "algorithm": ALG_LABELS[alg],
                                 "trial": t, "hypervolume": 0.0, "elapsed": 0.0})
                continue

            pts_raw  = df[OBJ_COLS].values
            pts_norm = normalise(df, sc)

            # Hypervolume
            hv = float(moocore.hypervolume(pts_norm, ref=ref, maximise=False))
            hv_rows.append({"instance": sc, "algorithm": ALG_LABELS[alg],
                             "trial": t, "hypervolume": hv, "elapsed": 0.0})

            # Store front points (raw + normalised)
            for i in range(len(pts_raw)):
                front_rows.append({
                    "instance":              sc,
                    "algorithm":             ALG_LABELS[alg],
                    "trial":                 t,
                    "f1_dv":                 pts_raw[i, 0],
                    "f2_unrecovered_value":  pts_raw[i, 1],
                    "f3_vehicles":           pts_raw[i, 2],
                    "f1_dv_norm":            pts_norm[i, 0],
                    "f2_unassigned_norm":    pts_norm[i, 1],
                    "f3_vehicles_norm":      pts_norm[i, 2],
                })

hv_df    = pd.DataFrame(hv_rows)
front_df = pd.DataFrame(front_rows)

hv_path     = os.path.join(OUT_DIR, "numerical_experiments.csv")
fronts_path = os.path.join(OUT_DIR, "pareto_fronts.csv")
hv_df.to_csv(hv_path, index=False)
front_df.to_csv(fronts_path, index=False)
print(f"Saved {hv_path}  ({len(hv_df)} rows)")
print(f"Saved {fronts_path}  ({len(front_df)} rows)")

# ga_pareto_{alg}.csv — all scenarios/trials combined, for radar_plots.py
for alg in ALG_NAMES:
    alg_df = front_df[front_df["algorithm"] == ALG_LABELS[alg]][OBJ_COLS]
    alg_path = os.path.join(OUT_DIR, f"ga_pareto_{alg}.csv")
    alg_df.to_csv(alg_path, index=False)
    print(f"Saved {alg_path}  ({len(alg_df)} rows)")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print("\n" + "="*72)
print(f"{'':14}{'Median HV':>14}{'IQR':>12}{'n_trials':>10}")
print("="*72)
for sc in SCENARIOS:
    print(f"\nScenario: {sc}")
    print("-"*72)
    grp = hv_df[hv_df["instance"] == sc]
    for alg_label in [ALG_LABELS[a] for a in ALG_NAMES]:
        vals = grp[grp["algorithm"] == alg_label]["hypervolume"].values
        med  = np.median(vals) if len(vals) > 0 else 0.0
        iqr  = (np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) > 1 else 0.0
        n    = int((vals > 0).sum())
        print(f"  {alg_label:<12}  {med:12.4e}  {iqr:10.2e}  {n:8d}")
print()

# ---------------------------------------------------------------------------
# Plot 1: HV boxplots (one panel per scenario)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
for idx, sc in enumerate(SCENARIOS):
    ax = axes[idx]
    grp = hv_df[hv_df["instance"] == sc]
    data   = [grp[grp["algorithm"] == ALG_LABELS[a]]["hypervolume"].values
               for a in ALG_NAMES]
    colors = [ALG_COLORS[a] for a in ALG_NAMES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(ALG_NAMES) + 1))
    ax.set_xticklabels([ALG_LABELS[a] for a in ALG_NAMES], fontsize=9)
    ax.set_title(sc.replace("_", " "), fontsize=10)
    ax.set_ylabel("Hypervolume (normalised)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

fig.suptitle("Hypervolume by Scenario and Algorithm", fontsize=13)
plt.tight_layout()
hv_plot_path = os.path.join(OUT_DIR, "hv_boxplots.pdf")
plt.savefig(hv_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved {hv_plot_path}")

# ---------------------------------------------------------------------------
# Plot 2: KDE shadow — f1_dv vs f2_unassigned_time per scenario
# ---------------------------------------------------------------------------

MIN_KDE = 15
LEVELS  = [0.40, 0.65, 0.82, 0.93, 0.98]

def plot_kde_shadow(ax, x, y, colour, label):
    if len(x) < MIN_KDE:
        ax.scatter(x, y, c=colour, label=label, alpha=0.75, s=35, zorder=3)
        return
    try:
        kde   = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xg    = np.linspace(x.min(), x.max(), 120)
        yg    = np.linspace(y.min(), y.max(), 120)
        Xg, Yg = np.meshgrid(xg, yg)
        Z     = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        z_sorted = np.sort(Z.ravel())[::-1]
        cdf   = np.cumsum(z_sorted) / z_sorted.sum()
        thresholds = [z_sorted[np.searchsorted(cdf, lv)] for lv in LEVELS]
        ax.contourf(Xg, Yg, Z, levels=sorted(thresholds) + [Z.max() * 2],
                    colors=[colour], alphas=[0.10, 0.16, 0.24, 0.35, 0.50])
        ax.scatter(x, y, c=colour, s=8, alpha=0.4, zorder=2)
        ax.plot([], [], c=colour, linewidth=4, label=label, alpha=0.7)
    except Exception:
        ax.scatter(x, y, c=colour, label=label, alpha=0.75, s=35, zorder=3)

for sc in SCENARIOS:
    fig, axes = plt.subplots(1, len(ALG_NAMES), figsize=(16, 4), sharey=True)
    for idx, alg in enumerate(ALG_NAMES):
        ax = axes[idx]
        pts_list = [fronts[sc][alg][t] for t in range(1, N_TRIALS + 1)
                    if fronts[sc][alg][t] is not None]
        if not pts_list:
            ax.set_title(ALG_LABELS[alg])
            continue
        pts = pd.concat(pts_list)
        plot_kde_shadow(ax, pts["f1_dv"].values, pts["f2_unrecovered_value"].values,
                        ALG_COLORS[alg], ALG_LABELS[alg])
        ax.set_title(ALG_LABELS[alg], fontsize=10)
        ax.set_xlabel("ΔV [m/s]")
        if idx == 0:
            ax.set_ylabel("Unrecovered value [USD]")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"KDE shadow — {sc.replace('_', ' ')}", fontsize=12)
    plt.tight_layout()
    kde_path = os.path.join(OUT_DIR, f"kde_shadow_{sc}.pdf")
    plt.savefig(kde_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {kde_path}")

print("\nAggregation complete.")
