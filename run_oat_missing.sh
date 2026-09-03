#!/usr/bin/env bash
# run_oat_missing.sh — run only the missing OAT sweeps (n_ref_dirs + crossover_prob).
#
# Existing sbx_eta / pm_eta results are preserved.
#
# Usage:
#   bash run_oat_missing.sh
#   N_TRIALS=3 bash run_oat_missing.sh

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
N_EVAL=${N_EVAL:-10000}
SCENARIO="tight_low_dv"
DV_BUDGET=5000.0
DEM_DIR="outputs/exp_demands"
H5_FILE="outputs/sensitivity_oat.h5"
PYTHON="python3"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Running missing OAT sweeps                             ║"
echo "╚══════════════════════════════════════════════════════════╝"


# ── NSGA-III-RK: n_ref_dirs ────────────────────────────────────────────────

#echo "── NSGA-III: sweeping n_ref_dirs ──"
#for level in 4 8 12 16 20; do
#    for trial in $(seq 1 $N_TRIALS); do
#        echo "  nsga3rk  n_ref_dirs=$level  trial=$trial"
#        $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga3rk" "n_ref_dirs" "$level" "$H5_FILE" \
#            --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
#    done
#done

# ── NSGA-III-T: crossover_prob ─────────────────────────────────────────────

echo "── NSGA-III-T: sweeping crossover_prob ──"
for level in 0.5 0.7 0.9 1.0; do
    for trial in $(seq 1 $N_TRIALS); do
        echo "  nsga3rk_ot  crossover_prob=$level  trial=$trial"
        $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga3rk_ot" "crossover_prob" "$level" "$H5_FILE" \
            --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
    done
done

# ── NSGA-III-T: n_ref_dirs ─────────────────────────────────────────────────

echo "── NSGA-III-T: sweeping n_ref_dirs ──"
for level in 4 8 12 16 20; do
    for trial in $(seq 1 $N_TRIALS); do
        echo "  nsga3rk_ot  n_ref_dirs=$level  trial=$trial"
        $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga3rk_ot" "n_ref_dirs" "$level" "$H5_FILE" \
            --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Done. Plot with: python3 plot_sensitivity_oat.py        ║"
echo "╚══════════════════════════════════════════════════════════╝"
