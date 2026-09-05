#!/bin/bash
# smoke_test_nsga2.sh — quick sanity check that the NSGA-III -> NSGA-II swap
# (D21, PAPER_COMPLETION_PLAN.md) didn't break anything, before spending real
# compute on the full sensitivity/experiment sweeps.
#
# Uses a tiny evaluation budget and an existing demand file, just to exercise
# the import, the decoder, and the CSV writer end to end. Takes seconds, not
# hours.
#
# Run from basic_project/:
#   bash smoke_test_nsga2.sh

set -e

cd "$(dirname "$0")"

echo "=== 1. syntax check the touched files ==="
python -c "import ast; [ast.parse(open(f).read()) for f in [
    'run_ga_trial.py', 'run_ga_sensitivity.py', 'pymoo_stuff.py',
    'plot_sensitivity_oat.py', 'plot_pure_knee.py', 'plot_sensitivity_knee.py',
    'plot_sensitivity_pareto.py', 'plot_pure_hv.py', 'plot_pure_pareto.py',
    'plot_sensitivity_hv.py',
]]"
echo "  OK"

echo ""
echo "=== 2. NSGA2 imports cleanly, NSGA3 is gone from the active pipeline ==="
python -c "from pymoo.algorithms.moo.nsga2 import NSGA2; print('  pymoo.algorithms.moo.nsga2.NSGA2 OK')"
if grep -q "NSGA3" run_ga_trial.py run_ga_sensitivity.py; then
    echo "  FAIL: NSGA3 still referenced in the trial/sensitivity runners"
    exit 1
fi
echo "  OK: no NSGA3 references left in run_ga_trial.py / run_ga_sensitivity.py"

echo ""
echo "=== 3. tiny real run: run_ga_trial.py with --n-eval 300 ==="
python run_ga_trial.py loose_high_dv 1 --n-eval 300 \
    --result-dir outputs/exp_results_smoketest \
    --algos nsga2rk nsga2rk_ot

echo ""
echo "=== 4. tiny real run: run_ga_sensitivity.py (n_ref_dirs -> pop_size) ==="
python run_ga_sensitivity.py loose_high_dv 1 nsga2rk n_ref_dirs 40 \
    outputs/sensitivity_smoketest.h5 --n-eval 300

echo ""
echo "=== smoke test complete ==="
echo "Check outputs/exp_results_smoketest/nsga2rk_loose_high_dv_01.csv and"
echo "outputs/exp_results_smoketest/nsga2rk_ot_loose_high_dv_01.csv exist and"
echo "are non-empty, and that outputs/sensitivity_smoketest.h5 has a"
echo "nsga2rk/n_ref_dirs/40p0/trial_01 dataset (h5dump -n or h5py to check)."
echo "If all of that looks right, the full sweeps in run_pure_sensitivity.sh /"
echo "run_sensitivity_oat.sh / run_oat_missing.sh / run_rkot_sensitivity.sh"
echo "are safe to run for real."
