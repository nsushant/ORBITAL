#!/usr/bin/env bash
# run_pure_sensitivity.sh — sensitivity analysis with pure algorithms.
#
# MDLS: no LNS (opt_times only).
# GAs:  no repair — pymoo native constraint handling, 20k eval budget.
#
# Outputs go to outputs/pure_sensitivity_results (separate from
# outputs/sensitivity_results).
#
# Usage:
#   bash run_pure_sensitivity.sh                        # all SEs, 5 trials
#   bash run_pure_sensitivity.sh dvbudget               # SE4 only
#   bash run_pure_sensitivity.sh size disttype          # SE1 + SE2 only
#   N_TRIALS=3 bash run_pure_sensitivity.sh dvbudget    # SE4, 3 trials

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
REGEN_DEMANDS=${REGEN_DEMANDS:-false}
N_EVAL=${N_EVAL:-10000}
JULIA="julia --project=. -t auto"
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

mkdir -p "$DEM_DIR" "$RES_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate demand files
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Generating sensitivity demand files           ║"
echo "╚══════════════════════════════════════════════════════════╝"

if [ "$REGEN_DEMANDS" = "true" ]; then
    if [[ ${#SE_FILTER[@]} -eq 0 ]]; then
        $JULIA generate_sensitivity_demands.jl
    else
        $JULIA generate_sensitivity_demands.jl "${SE_FILTER[@]}"
    fi
else
    echo "Skipped (set REGEN_DEMANDS=true to regenerate)."
fi

echo ""
echo "Phase 1 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Run algorithms
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Running pure algorithms (no LNS / no repair)  ║"
echo "╚══════════════════════════════════════════════════════════╝"

run_pair() {
    local key=$1
    local trial=$2
    local dv_budget=${3:-5000.0}

    echo "  [MDLS]  key=$key trial=$trial dv_budget=$dv_budget"
    $JULIA run_mdls_trial.jl "$key" "$trial" "$DEM_DIR" "$RES_DIR" "$dv_budget" "$H5_FILE"

    echo "  [GAs]   key=$key trial=$trial dv_budget=$dv_budget"
    $PYTHON run_ga_trial.py "$key" "$trial" \
        --demand-dir "$DEM_DIR" --result-dir "$RES_DIR" \
        --dv-budget "$dv_budget" --n-eval "$N_EVAL" --h5-file "$H5_FILE" \
        --algos nsga3rk nsga3rk_ot
}

# SE1 — instance size
if run_se "size"; then
    for lv in 10 50 100 150 200; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE1 size  lv=$lv  trial=$trial ──"
            run_pair "size_${lv}" "$trial"
        done
    done
fi

# SE2 — distribution type
if run_se "disttype"; then
    for lv in normal uniform; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE2 disttype  lv=$lv  trial=$trial ──"
            run_pair "disttype_${lv}" "$trial"
        done
    done
fi

# SE3 — ΔV threshold
if run_se "dv"; then
    for lv in 3000 5000 8000 12000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE3 dv  lv=$lv  trial=$trial ──"
            run_pair "dv_${lv}" "$trial"
        done
    done
fi

# SE4 — vehicle ΔV budget
if run_se "dvbudget"; then
    for lv in 1500 3000 5000 8000 10000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE4 dvbudget  lv=$lv  trial=$trial ──"
            run_pair "dvbudget_${lv}" "$trial" "$lv"
        done
    done
fi

# SE6 — mixed fleet
if run_se "mixed_fleet"; then
    for lv in tight_normal loose_uniform; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE6 mixed_fleet  lv=$lv  trial=$trial ──"
            run_pair "mixed_fleet_${lv}" "$trial" "10000"
        done
    done
fi

echo ""
echo "Phase 2 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Aggregate
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 3: Aggregating sensitivity results               ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo "NOTE: aggregate_sensitivity.py reads from a hardcoded path."
echo "      To aggregate pure results, temporarily set RES_DIR in that script"
echo "      or copy CSVs from $RES_DIR into outputs/sensitivity_results/ before running."

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Done. Raw CSVs in $RES_DIR                             ║"
echo "║  After aggregating, run:                                ║"
echo "║    python3 plot_sensitivity.py                           ║"
echo "║    python3 plot_knee_sensitivity.py                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
