#!/usr/bin/env bash
# measure_budget.sh — Phase 0.3 of PAPER_COMPLETION_PLAN.md
#
# Measures what each algorithm actually spends, so the paper's "10,000
# evaluations for all algorithms" claim can be stated correctly (finding F1).
#
# Two units are counted, both instrumented identically in the two stacks:
#   full objective evaluations — one complete (dV, unrecovered, fleet) triple
#                                for one whole solution. This is what pymoo
#                                terminates on.
#   transfer-cost lookups      — one cost-table query. The atomic work unit,
#                                and the one that captures the incremental
#                                evaluation local search gets for free.
#
# Usage:  ./measure_budget.sh [scenario] [trial]
# Writes: outputs/budget_measurement.txt

set -uo pipefail
SCEN="${1:-tight_normal}"
TRIAL="${2:-1}"
OUT="outputs/budget_measurement.txt"
export OOS_COUNT_EVALS=1

mkdir -p outputs
{
  echo "Budget measurement — scenario=$SCEN trial=$TRIAL — $(date -u +%FT%TZ)"
  echo "==========================================================="
  echo
  echo "--- MDLS (Julia, 1000 iterations, no evaluation cap) ---"
} > "$OUT"

julia --project=. -t auto run_mdls_trial.jl "$SCEN" "$TRIAL" 2>&1 \
  | grep -E "evaluation budget|full objective|transfer-cost|iterations" >> "$OUT"

{
  echo
  echo "--- NSGA-III and NSGA-III-T (Python, n_eval=10000 cap) ---"
} >> "$OUT"

python run_ga_trial.py "$SCEN" "$TRIAL" 2>&1 \
  | grep -E "^\[budget\]" >> "$OUT"

echo
echo "Wrote $OUT:"
cat "$OUT"
