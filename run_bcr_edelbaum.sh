#!/usr/bin/env bash
# BCR MDLS run on the Edelbaum cost table. Does not overwrite Lu tables or results.
# Servicer: 10 mN, Isp 2500 s, 7000 m/s tank.
#
#   bash run_bcr_edelbaum.sh              # rebuild table, then MDLS
#   SKIP_TABLE=1 bash run_bcr_edelbaum.sh # MDLS only (reuse existing table)

set -euo pipefail

JULIA="julia --project=. -t auto"
DEM_DIR="outputs/long_horizon_demands"
RES_DIR="outputs/long_horizon_results_edelbaum"
H5_FILE="outputs/long_horizon_results_edelbaum.h5"
TRIAL=1
BUDGET=7000

export COST_TABLE_PATH="outputs/cost_table_edelbaum.jld2"
export PHASING_TABLE_PATH="outputs/phasing_time_table_edelbaum.jld2"
export MIN_TOF_PATH="outputs/min_tof_table_edelbaum.jld2"
export COST_TABLE_PERIOD=400

mkdir -p "$DEM_DIR" "$RES_DIR"

if [ "${SKIP_TABLE:-0}" != "1" ]; then
    echo "── Building Edelbaum cost table (10 mN / 2500 s) ──"
    $JULIA build_edelbaum_cost_table.jl
fi

echo "── Generating bcr_mixed demands (60–90/yr, mixed 6–18 mo / 2–3 yr windows) ──"
$JULIA generate_sensitivity_demands.jl --outdir="$DEM_DIR" bcr_mixed

echo "── MDLS bcr_mixed=$BUDGET trial=$TRIAL (Edelbaum table) ──"
$JULIA run_mdls_trial.jl "bcr_mixed_${BUDGET}" "$TRIAL" "$DEM_DIR" "$RES_DIR" "$BUDGET" "$H5_FILE"

echo ""
echo "Done. Results in $H5_FILE"
echo "Plot: python3 plot_bcr_architecture.py --h5 $H5_FILE --out-dir $RES_DIR"
echo "      python3 plot_coverage_fleet.py --h5 $H5_FILE --out-dir $RES_DIR"
echo "      python3 plot_bcr_mass_trade.py --h5 $H5_FILE --out-dir $RES_DIR"
echo "      python3 plot_bcr_vs_fee.py --h5 $H5_FILE --out-dir $RES_DIR"
