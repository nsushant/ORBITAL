#!/usr/bin/env bash
# run_sensitivity_oat.sh — OAT sensitivity analysis on tight_low_dv scenario.
#
# Sweeps one parameter at a time while holding others at nominal values.
# Results written to outputs/sensitivity_oat.h5
#
# Usage:
#   bash run_sensitivity_oat.sh              # all parameters
#   bash run_sensitivity_oat.sh mdls         # MDLS params only
#   bash run_sensitivity_oat.sh nsga2rk      # NSGA-II params only
#   bash run_sensitivity_oat.sh nsga2rk_ot   # NSGA-II-T params only

set -euo pipefail

SCENARIO="tight_low_dv"
N_TRIALS=${N_TRIALS:-5}
N_EVAL=${N_EVAL:-10000}
DV_BUDGET=5000.0
DEM_DIR="outputs/exp_demands"
H5_FILE="outputs/sensitivity_oat.h5"

JULIA="julia --project=."
PYTHON="python"

FILTER=("$@")

should_run() {
    local name=$1
    [[ ${#FILTER[@]} -eq 0 ]] && return 0
    for f in "${FILTER[@]}"; do [[ "$f" == "$name" ]] && return 0; done
    return 1
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  OAT Sensitivity Analysis — scenario: $SCENARIO         ║"
echo "║  Trials: $N_TRIALS  n_eval (GA): $N_EVAL                ║"
echo "║  Output: $H5_FILE                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── MDLS ────────────────────────────────────────────────────────────────────

if should_run "mdls"; then
    echo "── MDLS: sweeping shift ──"
    for level in 5.0 15.0 30.0 60.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  mdls  shift=$level  trial=$trial"
            $JULIA run_mdls_sensitivity.jl "$SCENARIO" "$trial" "shift" "$level" "$H5_FILE" "$DEM_DIR" "$DV_BUDGET"
        done
    done

    echo "── MDLS: sweeping top_pct ──"
    for level in 0.25 0.5 0.75 1.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  mdls  top_pct=$level  trial=$trial"
            $JULIA run_mdls_sensitivity.jl "$SCENARIO" "$trial" "top_pct" "$level" "$H5_FILE" "$DEM_DIR" "$DV_BUDGET"
        done
    done
fi

# ── NSGA-II ────────────────────────────────────────────────────────────────

if should_run "nsga2rk"; then
    echo "── NSGA-II: sweeping sbx_eta ──"
    for level in 5.0 10.0 20.0 30.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk  sbx_eta=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk" "sbx_eta" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done

    echo "── NSGA-II: sweeping pm_eta ──"
    for level in 5.0 10.0 20.0 30.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk  pm_eta=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk" "pm_eta" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done

    echo "── NSGA-II: sweeping crossover_prob ──"
    for level in 0.5 0.7 0.9 1.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk  crossover_prob=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk" "crossover_prob" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done

    echo "── NSGA-II: sweeping n_ref_dirs ──"
    for level in 4 8 12 16; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk  n_ref_dirs=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk" "n_ref_dirs" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done
fi

# ── NSGA-II-T ──────────────────────────────────────────────────────────────

if should_run "nsga2rk_ot"; then
    echo "── NSGA-II-T: sweeping sbx_eta ──"
    for level in 5.0 10.0 20.0 30.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk_ot  sbx_eta=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk_ot" "sbx_eta" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done

    echo "── NSGA-II-T: sweeping pm_eta ──"
    for level in 5.0 10.0 20.0 30.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk_ot  pm_eta=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk_ot" "pm_eta" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done

    echo "── NSGA-II-T: sweeping crossover_prob ──"
    for level in 0.5 0.7 0.9 1.0; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk_ot  crossover_prob=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk_ot" "crossover_prob" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done

    echo "── NSGA-II-T: sweeping n_ref_dirs ──"
    for level in 4 8 12 16; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "  nsga2rk_ot  n_ref_dirs=$level  trial=$trial"
            $PYTHON run_ga_sensitivity.py "$SCENARIO" "$trial" "nsga2rk_ot" "n_ref_dirs" "$level" "$H5_FILE" \
                --demand-dir "$DEM_DIR" --dv-budget "$DV_BUDGET" --n-eval "$N_EVAL"
        done
    done
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Done. Results in $H5_FILE                              ║"
echo "║  Plot with: python plot_sensitivity_oat.py              ║"
echo "╚══════════════════════════════════════════════════════════╝"
