#!/usr/bin/env bash
# run_rkot_sensitivity.sh — re-run sensitivity analysis for NSGA-III-RK-OT only.
#
# Uses existing demand files and leaves MDLS / NSGA-III-RK results untouched.
# Only nsga3rk_ot_*.csv files and HDF5 entries get overwritten.
#
# Usage:
#   bash run_rkot_sensitivity.sh                        # all SEs, 5 trials
#   bash run_rkot_sensitivity.sh dvbudget               # SE4 only
#   N_TRIALS=3 bash run_rkot_sensitivity.sh             # all SEs, 3 trials

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
N_EVAL=${N_EVAL:-10000}
PYTHON="python3"
DEM_DIR="outputs/sensitivity_demands"
RES_DIR="outputs/pure_sensitivity_results"
H5_FILE="outputs/pure_sensitivity_results.h5"

SE_FILTER=()
[[ $# -gt 0 ]] && SE_FILTER=("$@")

run_se() {
    local name=$1
    [[ ${#SE_FILTER[@]} -eq 0 ]] && return 0
    for f in "${SE_FILTER[@]}"; do [[ "$f" == "$name" ]] && return 0; done
    return 1
}

mkdir -p "$RES_DIR"

run_rkot() {
    local key=$1
    local trial=$2
    local dv_budget=${3:-5000.0}

    echo "  [RK-OT] key=$key trial=$trial dv_budget=$dv_budget"
    $PYTHON run_ga_trial.py "$key" "$trial" \
        --demand-dir "$DEM_DIR" --result-dir "$RES_DIR" \
        --dv-budget "$dv_budget" --n-eval "$N_EVAL" --h5-file "$H5_FILE" \
        --algos nsga3rk_ot
}

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Running NSGA-III-RK-OT sensitivity (only)              ║"
echo "╚══════════════════════════════════════════════════════════╝"

# SE1 — instance size
if run_se "size"; then
    for lv in 10 50 100 150 200; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE1 size  lv=$lv  trial=$trial ──"
            run_rkot "size_${lv}" "$trial"
        done
    done
fi

# SE2 — distribution type
if run_se "disttype"; then
    for lv in normal uniform; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE2 disttype  lv=$lv  trial=$trial ──"
            run_rkot "disttype_${lv}" "$trial"
        done
    done
fi

# SE3 — ΔV threshold
if run_se "dv"; then
    for lv in 3000 5000 8000 12000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE3 dv  lv=$lv  trial=$trial ──"
            run_rkot "dv_${lv}" "$trial"
        done
    done
fi

# SE4 — vehicle ΔV budget
if run_se "dvbudget"; then
    for lv in 1500 3000 5000 8000 10000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE4 dvbudget  lv=$lv  trial=$trial ──"
            run_rkot "dvbudget_${lv}" "$trial" "$lv"
        done
    done
fi

# SE6 — mixed fleet
if run_se "mixed_fleet"; then
    for lv in tight_normal loose_uniform; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE6 mixed_fleet  lv=$lv  trial=$trial ──"
            run_rkot "mixed_fleet_${lv}" "$trial" "10000"
        done
    done
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Done. Only nsga3rk_ot results updated.                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
