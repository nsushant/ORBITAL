#!/usr/bin/env bash
# run_long_horizon.sh — 5-year horizon BCR study (MDLS only).
#
# Uses a separate demand directory (outputs/long_horizon_demands/) and
# results directory (outputs/long_horizon_results/) so existing 1-year
# experiment outputs are untouched.
#
# Prerequisites:
#   1. julia --project=. -t auto extend_depot_cost_table.jl   (done once)
#   2. COST_TABLE_PERIOD = 1825.0 in sol_utils.jl              (done)
#
# Usage:
#   bash run_long_horizon.sh                  # all SEs
#   bash run_long_horizon.sh dvbudget         # SE4 only
#   N_TRIALS=3 bash run_long_horizon.sh       # 3 trials

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
REGEN_DEMANDS=${REGEN_DEMANDS:-true}
JULIA="julia --project=. -t auto"
PYTHON="python"
DEM_DIR="outputs/long_horizon_demands"
RES_DIR="outputs/long_horizon_results"
H5_FILE="outputs/long_horizon_results.h5"

SE_FILTER=()
[[ $# -gt 0 ]] && SE_FILTER=("$@")

run_se() {
    local name=$1
    [[ ${#SE_FILTER[@]} -eq 0 ]] && return 0
    for f in "${SE_FILTER[@]}"; do [[ "$f" == "$name" ]] && return 0; done
    return 1
}

mkdir -p "$DEM_DIR" "$RES_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate demand files (5-year time_dist)
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Generating 5-year demand files                ║"
echo "╚══════════════════════════════════════════════════════════╝"

if [ "$REGEN_DEMANDS" = "true" ]; then
    if [[ ${#SE_FILTER[@]} -eq 0 ]]; then
        $JULIA generate_sensitivity_demands.jl --outdir="$DEM_DIR"
    else
        $JULIA generate_sensitivity_demands.jl --outdir="$DEM_DIR" "${SE_FILTER[@]}"
    fi
else
    echo "Skipped (set REGEN_DEMANDS=true to regenerate)."
fi

echo "Phase 1 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Run MDLS
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Running MDLS (5-year horizon)                 ║"
echo "╚══════════════════════════════════════════════════════════╝"

run_mdls() {
    local key=$1
    local trial=$2
    local dv_budget=${3:-5000.0}
    echo "  [MDLS]  key=$key trial=$trial dv_budget=$dv_budget"
    $JULIA run_mdls_trial.jl "$key" "$trial" "$DEM_DIR" "$RES_DIR" "$dv_budget" "$H5_FILE"
}

# SE1 — instance size
if run_se "size"; then
    for lv in 10 50 100 150 200; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE1 size  lv=$lv  trial=$trial ──"
            run_mdls "size_${lv}" "$trial"
        done
    done
fi

# SE2 — distribution type
if run_se "disttype"; then
    for lv in normal uniform; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE2 disttype  lv=$lv  trial=$trial ──"
            run_mdls "disttype_${lv}" "$trial"
        done
    done
fi

# SE3 — ΔV threshold
if run_se "dv"; then
    for lv in 3000 5000 8000 12000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE3 dv  lv=$lv  trial=$trial ──"
            run_mdls "dv_${lv}" "$trial"
        done
    done
fi

# SE4 — vehicle ΔV budget
if run_se "dvbudget"; then
    for lv in 1500 3000 5000 8000 10000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE4 dvbudget  lv=$lv  trial=$trial ──"
            run_mdls "dvbudget_${lv}" "$trial" "$lv"
        done
    done
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Done.                                                   ║"
echo "║  Results in: $RES_DIR                                   ║"
echo "║  HDF5:       $H5_FILE                                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
