#!/usr/bin/env bash
# run_numerical_experiments.sh — full numerical experiment pipeline.
#
# Phase 1: Julia generates all demand JLD2 + greedy JSON files.
# Phase 2: For each (scenario, trial), run Julia MDLS then Python GAs.
# Phase 3: Python aggregates results, computes HV, generates plots.
#
# Usage:
#   bash run_numerical_experiments.sh            # 10 trials (default)
#   N_TRIALS=3 bash run_numerical_experiments.sh # quick test with 3 trials

set -euo pipefail

N_TRIALS=${N_TRIALS:-10}
SCENARIOS="tight_normal loose_uniform tight_low_dv loose_high_dv"
JULIA="julia --project=. -t auto"
PYTHON="python"

mkdir -p outputs/exp_demands outputs/exp_results

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Generate all demand files
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Generating demand files                       ║"
echo "╚══════════════════════════════════════════════════════════╝"

$JULIA generate_experiment_demands.jl

echo ""
echo "Phase 1 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Run algorithms per (scenario, trial)
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: Running algorithms                            ║"
echo "╚══════════════════════════════════════════════════════════╝"

total=$(echo "$SCENARIOS" | wc -w | tr -d ' ')
total=$((total * N_TRIALS))
count=0

for scenario in $SCENARIOS; do
    for trial in $(seq 1 $N_TRIALS); do
        count=$((count + 1))
        echo ""
        echo "── [$count/$total] scenario=$scenario  trial=$trial ──────────────────"

        # Julia MDLS
        echo "  [MDLS] starting …"
        $JULIA run_mdls_trial.jl "$scenario" "$trial"

        # Python GAs (NSGA-III, MOEA-D, PSO)
        echo "  [GAs] starting …"
        $PYTHON run_ga_trial.py "$scenario" "$trial"
    done
done

echo ""
echo "Phase 2 complete."

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Aggregate, compute HV, generate plots
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 3: Aggregating results                           ║"
echo "╚══════════════════════════════════════════════════════════╝"

$PYTHON aggregate_results.py

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All done.  Results in outputs/                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
