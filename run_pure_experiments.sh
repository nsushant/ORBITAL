#!/usr/bin/env bash
# run_pure_experiments.sh — numerical experiments with pure algorithms.
#
# MDLS: no LNS (opt_times only).
# GAs:  no repair — pymoo native constraint handling, 20k eval budget.
#
# Outputs go to outputs/pure_results (separate from outputs/exp_results).
#
# Usage:
#   bash run_pure_experiments.sh            # 5 trials (default)
#   N_TRIALS=3 bash run_pure_experiments.sh # quick test

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
REGEN_DEMANDS=${REGEN_DEMANDS:-false}
N_EVAL=${N_EVAL:-20000}
SCENARIOS="tight_normal loose_uniform tight_low_dv loose_high_dv"
JULIA="julia --project=. -t auto"
PYTHON="python3"
DEM_DIR="outputs/exp_demands"
RES_DIR="outputs/pure_results"

mkdir -p "$DEM_DIR" "$RES_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate demand files (shared with standard experiments)
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Generating demand files                       ║"
echo "╚══════════════════════════════════════════════════════════╝"

if [ "$REGEN_DEMANDS" = "true" ]; then
    $JULIA generate_experiment_demands.jl
else
    echo "Skipped (set REGEN_DEMANDS=true to regenerate)."
fi

echo ""
echo "Phase 1 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Run algorithms per (scenario, trial)
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Running pure algorithms (no LNS / no repair)  ║"
echo "╚══════════════════════════════════════════════════════════╝"

total=$(echo "$SCENARIOS" | wc -w | tr -d ' ')
total=$((total * N_TRIALS))
count=0

for scenario in $SCENARIOS; do
    for trial in $(seq 1 $N_TRIALS); do
        count=$((count + 1))
        echo ""
        echo "── [$count/$total] scenario=$scenario  trial=$trial ──────────────────"

        # Julia MDLS (no LNS — opt_times only)
        echo "  [MDLS] starting …"
        $JULIA run_mdls_trial.jl "$scenario" "$trial" "$DEM_DIR" "$RES_DIR"

        # Python GAs — no repair, 20k evals
        echo "  [GAs] starting …"
        $PYTHON run_ga_trial.py "$scenario" "$trial" \
            --demand-dir "$DEM_DIR" \
            --result-dir "$RES_DIR" \
            --dv-budget 5000.0 \
            --n-eval "$N_EVAL"
    done
done

echo ""
echo "Phase 2 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Aggregate
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 3: Aggregating results                           ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo "NOTE: aggregate_results.py reads from a hardcoded path."
echo "      To aggregate pure results, temporarily set RES_DIR in that script"
echo "      or copy CSVs from $RES_DIR into outputs/exp_results/ before running."

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All done.  Raw CSVs in $RES_DIR                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
