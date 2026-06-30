#!/usr/bin/env bash
# run_moead_rerun.sh — re-run only MOEA-D trials (PBI + larger ref dirs).
#
# Demand and greedy files must already exist from a prior Phase 1 run.
# Usage:
#   bash run_moead_rerun.sh
#   N_TRIALS=3 bash run_moead_rerun.sh   # quick test

set -euo pipefail

N_TRIALS=${N_TRIALS:-5}
SCENARIOS="tight_normal loose_uniform tight_low_dv loose_high_dv"
PYTHON="python"

# ── Guard: demand files must already exist ───────────────────────────────────

echo "Checking demand files …"
for scenario in $SCENARIOS; do
    for trial in $(seq 1 $N_TRIALS); do
        trial_str=$(printf "%02d" $trial)
        demand_file="outputs/exp_demands/${scenario}_${trial_str}.jld2"
        greedy_file="outputs/exp_demands/${scenario}_${trial_str}_greedy.json"
        if [ ! -f "$demand_file" ] || [ ! -f "$greedy_file" ]; then
            echo "ERROR: missing demand/greedy files for $scenario trial $trial"
            echo "  expected: $demand_file"
            echo "  expected: $greedy_file"
            echo "Run Phase 1 first: julia generate_experiment_demands.jl"
            exit 1
        fi
    done
done
echo "All demand files present."

# ── Run MOEA-D only ──────────────────────────────────────────────────────────

total=$(echo "$SCENARIOS" | wc -w | tr -d ' ')
total=$((total * N_TRIALS))
count=0

for scenario in $SCENARIOS; do
    for trial in $(seq 1 $N_TRIALS); do
        count=$((count + 1))
        echo ""
        echo "── [$count/$total] moead  scenario=$scenario  trial=$trial ──"
        $PYTHON run_ga_trial.py "$scenario" "$trial" --dv-budget 5000.0 --algos moead
    done
done

echo ""
echo "MOEA-D re-run complete."
echo "Re-aggregate results: python aggregate_results.py"
