#!/usr/bin/env bash
# Reproduce the three depot-placement cases reported in the paper. The
# scenario-independent depot-legs cache is built by the first sweep and reused
# by the other two. Set JOBS in the environment to change parallelism.
set -euo pipefail

LEGS=outputs/depot_legs.h5
JOBS=${JOBS:-4}

for s in S1_repair S2_refuel S3_deorbit; do
  echo "=== sweep $s ==="
  python run_depot_sweep.py "$s" \
      --n-eval 1000 --jobs "$JOBS" --legs-cache "$LEGS"
  echo "=== lexicographic MILP $s ==="
  python run_depot_fronts_milp.py "$s" \
      --budgets 2e9 \
      --starlink-loss-values 0.4 \
      --planet-loss-values 0.4 \
      --dv-cap 12000 \
      --solver scipy \
      --mip-gap 0 \
      --time-limit 600 \
      --out "outputs/depot_milp_verified_${s}"
done

python plot_depot_coverage_map.py \
    --sweep-dir outputs \
    --surface-dir outputs \
    --surface-prefix depot_milp_verified \
    --B 2e9 \
    --starlink-loss 0.4 \
    --planet-loss 0.4 \
    --out figures/depot_coverage_map_grouped.pdf

echo "=== all scenarios done ==="
