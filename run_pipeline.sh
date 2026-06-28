#!/bin/bash
# Pipeline: MDLS + NSGA3 + MOEAD + comparison
# Usage: ./run_pipeline.sh [budget] [smoke]
#   budget: total number of evaluations (default 8000)
#   smoke:  if "smoke", use small budget for testing

set -e

BUDGET=${1:-8000}
SMOKE=${2:-""}

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUTS_DIR="$PROJECT_DIR/outputs"
PYTHON_DIR="$PROJECT_DIR/multi_objectove_GA_tests"

PYTHON="/Users/sushantnigudkar/miniforge3/bin/python3"

# Compute GA pop_size: use full 91 for budget >= 4550, scale down for smaller budgets
POP_SIZE=$(( BUDGET / 50 ))
if [ "$POP_SIZE" -lt 15 ]; then POP_SIZE=15; fi
if [ "$POP_SIZE" -gt 91 ]; then POP_SIZE=91; fi

if [ "$SMOKE" = "smoke" ]; then
    echo "=== SMOKE TEST mode (budget=$BUDGET) ==="
else
    echo "=== Full run (budget=$BUDGET) ==="
fi

echo "  GA pop_size=$POP_SIZE ($(( (BUDGET - POP_SIZE) / POP_SIZE )) gens)"
echo ""

echo "=== 1. MDLS (Julia) ==="
MDLS_ITER=$(( BUDGET / 8 ))
echo "  maxiter=$MDLS_ITER (${MDLS_ITER}x8=$(( MDLS_ITER * 8 )) evals)"
julia --project="$PROJECT_DIR" "$PROJECT_DIR/run_mdls.jl" "$MDLS_ITER"

echo ""
echo "=== 2. NSGA3 (Python) ==="
echo "  max_evals=$BUDGET  pop_size=$POP_SIZE"
$PYTHON "$PYTHON_DIR/run_nsga3.py" \
    --max_evals "$BUDGET" \
    --pop_size "$POP_SIZE" \
    --save_csv "$OUTPUTS_DIR/ga_pareto_nsga3.csv"

echo ""
echo "=== 3. MOEAD (Python) ==="
echo "  max_evals=$BUDGET  pop_size=$POP_SIZE"
$PYTHON "$PYTHON_DIR/run_moead.py" \
    --max_evals "$BUDGET" \
    --pop_size "$POP_SIZE" \
    --save_csv "$OUTPUTS_DIR/ga_pareto_moead.csv"

echo ""
echo "=== 4. Comparison (HV + plots) ==="
$PYTHON "$PYTHON_DIR/compare.py" \
    --mdls "$OUTPUTS_DIR/ga_pareto_mdls.csv" \
    --nsga3 "$OUTPUTS_DIR/ga_pareto_nsga3.csv" \
    --moead "$OUTPUTS_DIR/ga_pareto_moead.csv" \
    --out_dir "$OUTPUTS_DIR"

echo ""
echo "=== Pipeline complete ==="
echo "  MDLS  Pareto: $OUTPUTS_DIR/ga_pareto_mdls.csv"
echo "  NSGA3 Pareto: $OUTPUTS_DIR/ga_pareto_nsga3.csv"
echo "  MOEAD Pareto: $OUTPUTS_DIR/ga_pareto_moead.csv"
echo "  Plots:        $OUTPUTS_DIR/comparison_3d.png"
