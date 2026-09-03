#!/usr/bin/env bash
# run_sensitivity_analysis.sh — full sensitivity analysis pipeline.
#
# Phase 1: Julia generates demand JLD2 + greedy JSON for selected sub-experiments.
# Phase 2: For each (se, level, trial), run Julia MDLS then Python GAs.
# Phase 3: Python aggregates into 8 CSV files for plot_sensitivity.py.
#
# Usage:
#   bash run_sensitivity_analysis.sh                        # all SEs, 10 trials
#   bash run_sensitivity_analysis.sh dvbudget               # SE4 only
#   bash run_sensitivity_analysis.sh size disttype          # SE1 + SE2 only
#   N_TRIALS=3 bash run_sensitivity_analysis.sh dvbudget    # SE4, 3 trials

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
REGEN_DEMANDS=${REGEN_DEMANDS:-false}
JULIA="julia --project=. -t auto"
PYTHON="python3"
DEM_DIR="outputs/sensitivity_demands"
RES_DIR="outputs/sensitivity_results"

# Sub-experiments to run (default: all)
SE_FILTER=()
[[ $# -gt 0 ]] && SE_FILTER=("$@")   # e.g. ("dvbudget") or ("size" "disttype") or empty = all

run_se() {
    # Returns 0 if this SE should run, 1 otherwise
    local name=$1
    [[ ${#SE_FILTER[@]} -eq 0 ]] && return 0
    for f in "${SE_FILTER[@]}"; do [[ "$f" == "$name" ]] && return 0; done
    return 1
}

mkdir -p "$DEM_DIR" "$RES_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate demand files (filtered to selected SEs)
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
# Sub-experiments and levels must match generate_sensitivity_demands.jl
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Running algorithms                            ║"
echo "╚══════════════════════════════════════════════════════════╝"

run_pair() {
    local key=$1
    local trial=$2
    local dv_budget=${3:-5000.0}

    echo "  [MDLS]  key=$key trial=$trial dv_budget=$dv_budget"
    $JULIA run_mdls_trial.jl "$key" "$trial" "$DEM_DIR" "$RES_DIR" "$dv_budget"

    echo "  [GAs]   key=$key trial=$trial dv_budget=$dv_budget"
    $PYTHON run_ga_trial.py "$key" "$trial" \
        --demand-dir "$DEM_DIR" --result-dir "$RES_DIR" --dv-budget "$dv_budget"
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

# SE4 — vehicle ΔV budget (dv_budget passed to MDLS; GAs unaffected)
if run_se "dvbudget"; then
    for lv in 1500 3000 5000 8000 10000; do
        for trial in $(seq 1 $N_TRIALS); do
            echo "── SE4 dvbudget  lv=$lv  trial=$trial ──"
            run_pair "dvbudget_${lv}" "$trial" "$lv"
        done
    done
fi

# SE6 — mixed fleet: 100 Starlink + 100 Planet Labs, 10k m/s budget, 5-year horizon
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

$PYTHON aggregate_sensitivity.py

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Done. Run:                                             ║"
echo "║    python3 plot_sensitivity.py                           ║"
echo "║    python3 plot_knee_sensitivity.py                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
