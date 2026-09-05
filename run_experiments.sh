#!/bin/bash
# run_experiments.sh — the Section 4 comparison: MDLS against NSGA-II,
# three scenarios, five trials each, equal evaluation budget.
#
# Both methods start from the same schedule (the Sec. 3.6 constructive
# heuristic), score through the same evaluator (oos/schedule.py), and are held
# to the same number of full objective evaluations. MDLS spends three per
# iteration, so its iteration count is the budget divided by three; the runners
# report what they actually spent so the claim can be checked.
#
# Run from basic_project/:
#   bash run_experiments.sh                 # 5 trials, 10k evaluations
#   bash run_experiments.sh 15 10000        # 15 trials (TODO item 3)
#   bash run_experiments.sh 1 900           # quick smoke
#
# Results: outputs/exp_results/{mdls,nsga2}_{scenario}_{trial}.csv

set -e
cd "$(dirname "$0")"

TRIALS=${1:-5}
N_EVAL=${2:-10000}

# Tuned parameters, read from file rather than pasted in, so the numbers in a
# result can be traced to the race that produced them. Comment lines are
# stripped; an absent or empty file means defaults, which is what the untuned
# comparison uses.
strip () { [ -f "$1" ] && grep -vE '^\s*#|^\s*$' "$1" | tr '\n' ' ' || true; }
MDLS_PARAMS=$(strip tune/best-mdls.txt)
NSGA_PARAMS=$(strip tune/best-nsga2.txt)
if [ -n "$MDLS_PARAMS" ] || [ -n "$NSGA_PARAMS" ]; then
    echo "=== tuned parameters ==="
    echo "  MDLS    ${MDLS_PARAMS:-(defaults)}"
    echo "  NSGA-II ${NSGA_PARAMS:-(defaults)}"
    echo ""
fi
SCENARIOS="S1_repair S2_refuel S3_deorbit"
RES_DIR=outputs/exp_results
H5=outputs/experiment_fronts.h5

echo "=== gates ==="
python tests/test_schedule.py  > /dev/null && echo "  schedule   OK"
python tests/test_archive.py   > /dev/null && echo "  archive    OK"
python tests/test_ga_problem.py > /dev/null && echo "  ga_problem OK"
python tests/test_greedy.py    > /dev/null && echo "  greedy     OK"
python tests/test_mdls.py      > /dev/null && echo "  mdls       OK"

echo ""
echo "=== smoke: one short run of each, so failures surface in seconds ==="
python run_mdls_trial.py S2_refuel 1 --n-eval 300 --result-dir /tmp/smoke $MDLS_PARAMS > /dev/null
echo "  MDLS     OK"
python run_ga_trial.py   S2_refuel 1 --n-eval 300 --result-dir /tmp/smoke $NSGA_PARAMS > /dev/null
echo "  NSGA-II  OK"

echo ""
echo "=== $TRIALS trials x 3 scenarios x 2 algorithms, $N_EVAL evaluations each ==="
mkdir -p "$RES_DIR"
for scen in $SCENARIOS; do
    for t in $(seq 1 "$TRIALS"); do
        printf "  %-11s trial %2d  " "$scen" "$t"
        python run_mdls_trial.py "$scen" "$t" --n-eval "$N_EVAL" \
               --result-dir "$RES_DIR" --h5-file "$H5" $MDLS_PARAMS > /dev/null
        printf "MDLS done  "
        python run_ga_trial.py "$scen" "$t" --n-eval "$N_EVAL" \
               --result-dir "$RES_DIR" --h5-file "$H5" $NSGA_PARAMS > /dev/null
        printf "NSGA-II done\n"
    done
done

echo ""
echo "=== summary ==="
python - <<'PY'
import glob, os
import numpy as np, pandas as pd

rows = []
for f in sorted(glob.glob("outputs/exp_results/*_S*_*.csv")):
    base = os.path.basename(f)[:-4]
    algo, scen, trial = base.split("_")[0], "_".join(base.split("_")[1:-1]), base.split("_")[-1]
    d = pd.read_csv(f)
    if not len(d):
        rows.append(dict(algo=algo, scenario=scen, trial=trial, points=0))
        continue
    rows.append(dict(algo=algo, scenario=scen, trial=trial, points=len(d),
                     best_dv=d.f1_dv.min(),
                     best_unrec=d.f2_unrecovered_value.min() / 1e6,
                     min_fleet=d.f3_vehicles.min()))
df = pd.DataFrame(rows)
if df.empty:
    print("no results found")
else:
    g = df.groupby(["scenario", "algo"]).agg(
        trials=("trial", "count"), front_points=("points", "mean"),
        best_dv=("best_dv", "mean"), best_unrec_M=("best_unrec", "mean"),
        min_fleet=("min_fleet", "mean")).round(1)
    print(g.to_string())
    print("\nfront_points is the mean number of non-dominated solutions per trial;")
    print("best_* are the per-trial extremes averaged over trials. These are")
    print("descriptive only -- the hypervolume and knee-point comparison of")
    print("Sec. 4.1, and the Wilcoxon tests of TODO item 3, come next.")
PY

echo ""
echo "=== done ==="
echo "Fronts in $RES_DIR/ and $H5"
echo "Next: hypervolume + knee-point per trial, then Wilcoxon rank-sum"
echo "(TODO item 3 wants 10-15 trials; at 5 v 5 the smallest attainable"
echo " two-sided p is 0.0079, which leaves no margin)."
