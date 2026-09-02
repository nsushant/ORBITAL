#!/usr/bin/env bash
# Depot BCR crossover maps at several Edelbaum thrust levels.
# Each thrust rebuilds the Edelbaum table (burn time ∝ 1/T), then the (a, i) MDLS sweep.
#
#   bash run_depot_thrust_maps.sh
#   THRUSTS="0.01 1.0" bash run_depot_thrust_maps.sh   # subset (Newtons)
#   SKIP_TABLE=1 bash run_depot_thrust_maps.sh         # reuse tagged tables
#
# Default thrusts: 10 mN, 100 mN, 1 N. Isp 2500 s, tank 7000 m/s.
# 10 mN can reuse outputs/cost_table_edelbaum.jld2 if the tagged copy is missing.

set -euo pipefail

JULIA="julia --project=. -t auto"
DEM="outputs/long_horizon_demands/bcr_mixed_7000_01.jld2"
# Newtons: 10 mN, 100 mN, 1 N
THRUSTS="${THRUSTS:-0.01 0.1 1.0}"

thrust_tag() {
    python -c "t=float('$1'); print(f'{t:.0f}N' if t>=0.999 else f'{t*1e3:.0f}mN')"
}

if [ ! -f "$DEM" ]; then
    echo "── Generating bcr_mixed_7000 demands ──"
    $JULIA generate_sensitivity_demands.jl --outdir=outputs/long_horizon_demands bcr_mixed
fi

tags=()
for T in $THRUSTS; do
    tag=$(thrust_tag "$T")
    tags+=("$tag")
    ct="outputs/cost_table_edelbaum_${tag}.jld2"
    pt="outputs/phasing_time_table_edelbaum_${tag}.jld2"
    mt="outputs/min_tof_table_edelbaum_${tag}.jld2"
    h5="outputs/depot_location_sweep_${tag}.h5"
    res="outputs/depot_location_sweep_${tag}"

    echo ""
    echo "════════ ${tag}  (THRUST_N=${T} N) ════════"
    mkdir -p "$res"

    if [ ! -f "$ct" ] && [ "$T" = "0.01" ] && [ -f outputs/cost_table_edelbaum.jld2 ]; then
        echo "── Reusing 10 mN production table as $ct ──"
        cp -n outputs/cost_table_edelbaum.jld2 "$ct"
        cp -n outputs/phasing_time_table_edelbaum.jld2 "$pt" 2>/dev/null || true
        cp -n outputs/min_tof_table_edelbaum.jld2 "$mt" 2>/dev/null || true
    fi

    export THRUST_N="$T"
    export COST_TABLE_PATH="$ct"
    export PHASING_TABLE_PATH="$pt"
    export MIN_TOF_PATH="$mt"
    export DEPOT_H5="$h5"
    export DEPOT_RES_DIR="$res"

    if [ "${SKIP_TABLE:-0}" != "1" ] && [ ! -f "$ct" ]; then
        echo "── Building Edelbaum table ${tag} ──"
        $JULIA build_edelbaum_cost_table.jl
    elif [ ! -f "$ct" ]; then
        echo "Missing $ct (SKIP_TABLE=1)"
        exit 1
    fi

    echo "── Depot (a, i) MDLS sweep ${tag} ──"
    $JULIA sweep_depot_location.jl

    echo "── Plot ${tag} ──"
    python plot_depot_bcr_map.py --h5 "$h5" --out-dir "$res" --title "$(python -c "t='$tag'; print(t[:-2]+' mN' if t.endswith('mN') else t[:-1]+' N' if t.endswith('N') else t)")"
done

echo ""
echo "── Combined thrust panel ──"
python plot_depot_bcr_maps_thrust.py --tags "${tags[@]}"

echo ""
echo "Done. Combined: outputs/depot_location_sweep/depot_bcr_maps_thrust.pdf"
