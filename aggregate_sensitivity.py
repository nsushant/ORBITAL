"""
aggregate_sensitivity.py — Phase 3 of the sensitivity analysis pipeline.

Reads per-trial CSVs from outputs/sensitivity_results/ and produces the
8 CSV files consumed by plot_sensitivity.py and plot_knee_sensitivity.py.

Output files (must match these exact names and column layouts):
  outputs/sensitivity_size.csv          + sensitivity_size_fronts.csv
  outputs/sensitivity_disttype.csv      + sensitivity_disttype_fronts.csv
  outputs/sensitivity_dv.csv            + sensitivity_dv_fronts.csv
  outputs/sensitivity_dvbudget.csv      + sensitivity_dvbudget_fronts.csv
"""

import os
import numpy as np
import pandas as pd
import moocore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALG_NAMES  = ["mdls", "nsga3", "moead", "pso"]
ALG_LABELS = {"mdls": "MDLS", "nsga3": "NSGA-III", "moead": "MOEA/D", "pso": "PSO"}
N_TRIALS   = 10
RES_DIR    = "outputs/sensitivity_results"
OUT_DIR    = "outputs"
PENALTY    = 1e6
OBJ_COLS   = ["f1_dv", "f2_unassigned_time", "f3_vehicles"]

# Problem-motivated fixed worst-case bounds (independent of which algorithms ran).
N_DEMANDS    = 200
SERVICE_TIME = 3.0
DV_BUDGET    = 5000.0
N_VEHICLES   = 20
FIXED_NADIR  = np.array([N_DEMANDS * DV_BUDGET,    # f1: 1 000 000 m/s
                          N_DEMANDS * SERVICE_TIME,  # f2: 600 days
                          N_VEHICLES + 5])           # f3: 25 vehicles

# Sub-experiment definitions: (se_name, factor_col, levels, dv_budget_per_level)
# dv_budget_per_level only relevant for SE4 (for labelling); None means N/A
SUBEXPS = [
    dict(name       = "size",
         factor_col = "num_demands",
         levels     = ["10", "50", "100", "150", "200"],
         factor_val = lambda lv: int(lv)),

    dict(name       = "disttype",
         factor_col = "disttype",
         levels     = ["normal", "uniform"],
         factor_val = lambda lv: lv),

    dict(name       = "dv",
         factor_col = "deltaV_dist",
         levels     = ["3000", "5000", "8000", "12000"],
         factor_val = lambda lv: int(lv)),

    dict(name       = "dvbudget",
         factor_col = "dv_budget",
         levels     = ["1500", "3000", "5000", "8000"],
         factor_val = lambda lv: int(float(lv))),
]

# ---------------------------------------------------------------------------
# Load one per-trial front CSV
# ---------------------------------------------------------------------------

def load_front(se_name, level_label, alg, trial):
    path = os.path.join(RES_DIR,
                        f"{alg}_{se_name}_{level_label}_{trial:02d}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df = df[df["f1_dv"] < PENALTY]
    return df if not df.empty else None

# ---------------------------------------------------------------------------
# Process each sub-experiment
# ---------------------------------------------------------------------------

for se in SUBEXPS:
    se_name    = se["name"]
    factor_col = se["factor_col"]
    levels     = se["levels"]
    factor_val = se["factor_val"]

    print(f"\n{'='*60}")
    print(f"  Sub-experiment: {se_name}  ({factor_col})")
    print(f"{'='*60}")

    # ── Step 1: load all fronts ──────────────────────────────────────────────
    # fronts[lv][alg][trial] = DataFrame or None
    fronts = {lv: {alg: {} for alg in ALG_NAMES} for lv in levels}
    missing = 0
    for lv in levels:
        for alg in ALG_NAMES:
            for t in range(1, N_TRIALS + 1):
                df = load_front(se_name, lv, alg, t)
                fronts[lv][alg][t] = df
                if df is None:
                    missing += 1
                    print(f"  MISSING: {alg} {lv} trial {t}")

    if missing > 0:
        print(f"  Total missing: {missing}")

    # ── Step 2: global ideal/nadir for this sub-experiment ───────────────────
    all_pts = []
    for lv in levels:
        for alg in ALG_NAMES:
            for t in range(1, N_TRIALS + 1):
                df = fronts[lv][alg][t]
                if df is not None:
                    all_pts.append(df[OBJ_COLS].values)

    if not all_pts:
        print(f"  WARNING: no data for {se_name} — skipping")
        continue

    all_pts = np.vstack(all_pts)
    ideal   = all_pts.min(axis=0)
    nadir   = FIXED_NADIR          # problem-motivated, not data-driven
    rng     = np.maximum(nadir - ideal, 1e-10)
    ref     = np.array([1.1, 1.1, 1.1])

    print(f"  ideal={ideal.round(2)}  nadir={nadir.round(2)}  (fixed)")

    def normalise(pts_raw):
        return (pts_raw - ideal) / rng

    # ── Step 3: compute HV + build output rows ───────────────────────────────
    hv_rows    = []
    front_rows = []

    for lv in levels:
        fv = factor_val(lv)
        for alg in ALG_NAMES:
            alg_label = ALG_LABELS[alg]
            for t in range(1, N_TRIALS + 1):
                df = fronts[lv][alg][t]
                if df is None:
                    hv_rows.append({factor_col: fv, "algorithm": alg_label,
                                    "trial": t, "hypervolume": 0.0, "elapsed": 0.0})
                    continue

                pts_raw  = df[OBJ_COLS].values
                pts_norm = normalise(pts_raw)

                hv = float(moocore.hypervolume(pts_norm, ref=ref, maximise=False))
                hv_rows.append({factor_col: fv, "algorithm": alg_label,
                                "trial": t, "hypervolume": hv, "elapsed": 0.0})

                for i in range(len(pts_raw)):
                    front_rows.append({
                        factor_col:           fv,
                        "algorithm":          alg_label,
                        "trial":              t,
                        "f1_dv_norm":         pts_norm[i, 0],
                        "f2_unassigned_norm": pts_norm[i, 1],
                        "f3_vehicles_norm":   pts_norm[i, 2],
                        "f1_dv":              pts_raw[i, 0],
                        "f2_unassigned_time": pts_raw[i, 1],
                        "f3_vehicles":        pts_raw[i, 2],
                    })

    # ── Step 4: write CSVs ───────────────────────────────────────────────────
    hv_df    = pd.DataFrame(hv_rows)
    front_df = pd.DataFrame(front_rows)

    hv_path     = os.path.join(OUT_DIR, f"sensitivity_{se_name}.csv")
    fronts_path = os.path.join(OUT_DIR, f"sensitivity_{se_name}_fronts.csv")

    hv_df.to_csv(hv_path, index=False)
    front_df.to_csv(fronts_path, index=False)

    print(f"  Saved {hv_path}  ({len(hv_df)} rows)")
    print(f"  Saved {fronts_path}  ({len(front_df)} rows)")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n  {'Level':<12} {'Alg':<10} {'Median HV':>12}  {'IQR':>10}")
    print(f"  {'-'*50}")
    for lv in levels:
        fv = factor_val(lv)
        sub = hv_df[hv_df[factor_col] == fv]
        for alg_label in [ALG_LABELS[a] for a in ALG_NAMES]:
            vals = sub[sub["algorithm"] == alg_label]["hypervolume"].values
            med  = np.median(vals) if len(vals) > 0 else 0.0
            iqr  = (np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) > 1 else 0.0
            print(f"  {str(fv):<12} {alg_label:<10} {med:12.4e}  {iqr:10.2e}")

print("\nDone.")
