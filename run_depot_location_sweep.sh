#!/usr/bin/env bash
# Depot (a, i) sweep on the Edelbaum sat↔sat table, then BCR crossover map.
#
#   bash run_depot_location_sweep.sh
#
# Requires outputs/cost_table_edelbaum.jld2 (from run_bcr_edelbaum.sh).
# Does not overwrite Lu tables or the Edelbaum sat↔sat file.

set -euo pipefail

JULIA="julia --project=. -t auto"
CT_PATH="outputs/cost_table_edelbaum.jld2"
DEM="outputs/long_horizon_demands/bcr_mixed_7000_01.jld2"
H5="outputs/depot_location_sweep.h5"
OUT_DIR="outputs/depot_location_sweep"

if [ ! -f "$CT_PATH" ]; then
    echo "Missing $CT_PATH"
    echo "Build it first: bash run_bcr_edelbaum.sh"
    exit 1
fi

if [ ! -f "$DEM" ]; then
    echo "── Generating bcr_mixed demands ──"
    $JULIA generate_sensitivity_demands.jl --outdir=outputs/long_horizon_demands bcr_mixed
fi

mkdir -p "$OUT_DIR"

echo "── Depot (a, i) MDLS sweep ──"
$JULIA sweep_depot_location.jl

echo "── Plot implied-contract F* map (knee, BCR* ≥ 1) ──"
python plot_depot_bcr_map.py --h5 "$H5" --out-dir "$OUT_DIR"

echo ""
echo "Done. Map: $OUT_DIR/depot_bcr_map.pdf"
