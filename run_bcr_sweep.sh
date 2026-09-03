#!/usr/bin/env bash
# run_bcr_sweep.sh — BCR sweep: MDLS, 1 trial, dvbudget 1000:1000:5000 m/s
# Uses SE5 (bcr_dvbudget) demand files; generates them if missing.

set -euo pipefail

JULIA="julia --project=. -t auto"
PYTHON="python"
DEM_DIR="outputs/sensitivity_demands"
RES_DIR="outputs/pure_sensitivity_results"
H5_FILE="outputs/pure_sensitivity_results.h5"
TRIAL=1

mkdir -p "$DEM_DIR" "$RES_DIR"

# Phase 1: generate demand files if any are missing
for budget in 1000 2000 3000 4000 5000; do
    dem="$DEM_DIR/bcr_dvbudget_${budget}_0${TRIAL}.jld2"
    if [ ! -f "$dem" ]; then
        echo "── Generating demand files (bcr_dvbudget) ──"
        $JULIA generate_sensitivity_demands.jl bcr_dvbudget
        break
    fi
done

# Phase 2: run MDLS for each budget level
for budget in 1000 2000 3000 4000 5000; do
    echo "── bcr_dvbudget=$budget  trial=$TRIAL ──"
    $JULIA run_mdls_trial.jl "bcr_dvbudget_${budget}" "$TRIAL" "$DEM_DIR" "$RES_DIR" "$budget" "$H5_FILE"
done

echo "Done. Run: python plot_bcr_architecture.py"
