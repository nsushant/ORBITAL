"""
plot_pure_pareto.py — Pareto front scatter plots for pure-comparison experiments.

Reads CSVs directly from outputs/pure_results/ (no aggregation step needed).
Produces one PDF per scenario with one panel per algorithm, showing the
normalised Pareto front across all trials.

Run: python plot_pure_pareto.py
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker as ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RES_DIR = "outputs/pure_results"
OUT_DIR = "outputs"
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

# Reference point for normalisation (same across all scenarios for comparability)
# Computed lazily from data if not set
REF_DV    = None   # set from data
REF_F2    = None
REF_F3    = None

DV_FILTER = np.inf   # no filter — plot all solutions including infeasible-leg

FS      = 18
FS_TICK = 14

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
            df = df[df["f1_dv"] < DV_FILTER]   # drop infeasible-leg solutions
            if df.empty:
                continue
            df["scenario"] = scenario
            df["algo"]     = algo
            df["trial"]    = trial
            records.append(df)

if not records:
    raise FileNotFoundError(f"No CSV files found in {RES_DIR}")

data = pd.concat(records, ignore_index=True)

# ---------------------------------------------------------------------------
# Normalise objectives globally (0-1 based on global max)
# ---------------------------------------------------------------------------

MAX_DV = data["f1_dv"].max()
MAX_F2 = data["f2_unrecovered_value"].max()
MAX_F3 = data["f3_vehicles"].max()

data["f1_norm"] = data["f1_dv"]               / MAX_DV
data["f2_norm"] = data["f2_unrecovered_value"] / MAX_F2
data["f3_norm"] = data["f3_vehicles"]          / MAX_F3

print(f"Global max: dv={MAX_DV:.0f}  f2={MAX_F2:.3e}  f3={MAX_F3:.0f}")

# ---------------------------------------------------------------------------
# Plot: 2×2 grid of scenarios per objective pair, algos overlaid per panel
# ---------------------------------------------------------------------------

def _apply_sci_ticks(ax):
    """Use scientific notation on both axes if values are very small."""
    for axis, getter in [("x", ax.get_xlim), ("y", ax.get_ylim)]:
        lo, hi = getter()
        if hi > 0 and hi < 0.01:
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 2))
            if axis == "x":
                ax.xaxis.set_major_formatter(fmt)
            else:
                ax.yaxis.set_major_formatter(fmt)

OBJ_PAIRS = [
    ("f1_norm", "f2_norm", "ΔV (normalised)", "Lost value (normalised)"),
    ("f1_norm", "f3_norm", "ΔV (normalised)", "Vehicles (normalised)"),
    ("f2_norm", "f3_norm", "Lost value (normalised)", "Vehicles (normalised)"),
]

scenario_list  = list(SCENARIOS.items())   # [(key, label), ...]

for scenario, scenario_label in scenario_list:
    n_algos = len(ALGOS)
    n_pairs = len(OBJ_PAIRS)
    # rows = algorithms, cols = objective pairs
    fig, axes = plt.subplots(n_algos, n_pairs, figsize=(6 * n_pairs, 5 * n_algos))
    fig.suptitle(scenario_label, fontsize=FS + 2)

    sub = data[data["scenario"] == scenario]

    for row, (algo, algo_label) in enumerate(ALGOS.items()):
        asub = sub[sub["algo"] == algo]
        for col, (xcol, ycol, xlabel, ylabel) in enumerate(OBJ_PAIRS):
            ax = axes[row, col]
            if not asub.empty:
                ax.scatter(asub[xcol], asub[ycol],
                           c=COLOURS[algo], s=30, alpha=0.8, linewidths=0)
            if row == 0:
                ax.set_title(f"{xlabel} vs {ylabel}", fontsize=FS - 2)
            if col == 0:
                ax.set_ylabel(f"{algo_label}\n{ylabel}", fontsize=FS - 2)
            else:
                ax.set_ylabel(ylabel, fontsize=FS - 2)
            ax.set_xlabel(xlabel, fontsize=FS - 2)
            ax.tick_params(labelsize=FS_TICK)
            ax.grid(False)
            _apply_sci_ticks(ax)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"pure_pareto_{scenario}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

# ---------------------------------------------------------------------------
# Summary table: median front size per algo per scenario
# ---------------------------------------------------------------------------

summary = (data.groupby(["scenario", "algo", "trial"])
               .size()
               .reset_index(name="n_points")
               .groupby(["scenario", "algo"])["n_points"]
               .agg(["median", "min", "max"]))
print("\nFront size summary (n_points):")
print(summary.to_string())
