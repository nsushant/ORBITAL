#!/bin/bash
# run_irace.sh — tune MDLS and NSGA-II, symmetrically, on instances neither
# will be tested on (TODO item 2).
#
# Three stages, each skippable if its output already exists:
#
#   1. a tuning set drawn with --trial-offset 100, so it shares no demand
#      realisation with the Sec. 4 trials 1-15;
#   2. a hypervolume reference box per instance, from pilot runs of *both*
#      algorithms at their default parameters, fixed before tuning begins and
#      never recomputed -- a box derived from the run being scored would move
#      with the configuration;
#   3. two irace races, same budget, same instances, same seed.
#
# Requires R with the irace package for stage 3:
#   Rscript -e 'install.packages("irace", repos="https://cloud.r-project.org")'
# Stages 1 and 2 need only Python, and are worth running first: they are the
# slow part and the part that can be checked.
#
#   bash run_irace.sh              # full: 30 instances, 10k evals, 1000 experiments
#   bash run_irace.sh --smoke      # 3 instances, 900 evals, 60 experiments
#   bash run_irace.sh --stages 12  # prepare only, no race

set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE

TRIALS=10; N_EVAL=10000; MAXEXP=1000; PILOT=2; STAGES=123
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)  TRIALS=1; N_EVAL=900; MAXEXP=300; PILOT=1 ;;
        --stages) STAGES="$2"; shift ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
    shift
done

DEM=outputs/tune_demands
REF=outputs/tune_reference.json
export OOS_N_EVAL="$N_EVAL" OOS_DEMAND_DIR="$DEM" OOS_REFERENCE="$REF"

echo "=== 0. memory-mappable cost table ==="
python - <<NPY
import sys, os
sys.path.insert(0, ".")
from oos.schedule import _npy_sidecars, export_npy
h5 = "outputs/cost_table.h5"
dv, ph, rc = _npy_sidecars(h5)
if (os.path.exists(dv) and os.path.exists(ph) and os.path.exists(rc)
        and min(os.path.getmtime(f) for f in (dv, ph, rc))
            >= os.path.getmtime(h5)):
    print("  sidecars present and current")
else:
    print("  writing .npy sidecars (once, ~20 s) so every worker shares one")
    print("  copy of the table instead of reading its own 2.6 GB")
    export_npy(h5)
NPY
echo ""

if [[ "$STAGES" == *1* ]]; then
    echo "=== 1. tuning instances (offset 100: disjoint from the test set) ==="
    if [ -d "$DEM" ] && [ -n "$(ls -A "$DEM" 2>/dev/null)" ]; then
        echo "  $DEM already populated ($(ls "$DEM" | wc -l | tr -d ' ') files); leaving it"
    else
        python generate_demands.py --trials "$TRIALS" --trial-offset 100 \
                                   --out-dir "$DEM"
    fi
fi

if [[ "$STAGES" == *2* ]]; then
    echo ""
    echo "=== 2. hypervolume reference box per instance ==="
    # "It exists" is not a reason to keep it. The box must have been built from
    # both algorithms and from the code as it stands: a stale or one-sided box
    # scores every configuration against a scale drawn from a superseded
    # algorithm, and nothing downstream would say so. One was found on disk
    # carrying both_algorithms=false and predating the 2-regret repair, left
    # behind by a test (F46).
    REUSE=no
    if [ -f "$REF" ]; then
        REUSE=$(python - "$REF" <<'CHK'
import json, sys, os, glob
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("no"); raise SystemExit
if not d or not all(v.get("both_algorithms") for v in d.values()):
    print("no"); raise SystemExit
src = glob.glob("oos/*.py") + glob.glob("tune/*.py")
if src and os.path.getmtime(sys.argv[1]) < max(os.path.getmtime(f) for f in src):
    print("no"); raise SystemExit
print("yes")
CHK
)
    fi
    if [ "$REUSE" == "yes" ]; then
        echo "  $REF is current and built from both algorithms; reusing it"
    else
        if [ -f "$REF" ]; then
            echo "  the box on disk is stale or one-sided; rebuilding"
            mv -f "$REF" "$REF.superseded"
        fi
        python tune/make_reference.py --demand-dir "$DEM" --out "$REF" \
                                      --n-eval "$N_EVAL" --pilot-seeds "$PILOT"
    fi
fi

if [[ "$STAGES" == *3* ]]; then
    echo ""
    echo "=== 3. the two races ==="
    if ! command -v irace >/dev/null 2>&1 && ! Rscript -e 'library(irace)' >/dev/null 2>&1; then
        echo "  irace is not installed. Stages 1 and 2 are done and their output"
        echo "  is on disk, so this is the only thing missing:"
        echo "    Rscript -e 'install.packages(\"irace\", repos=\"https://cloud.r-project.org\")'"
        exit 1
    fi
    # Preflight. A runner that fails for every configuration still returns a
    # number, so irace races happily for a thousand experiments and reports a
    # winner drawn entirely from ties -- which is what happened (F43).
    #
    # Two things the first version of this check got wrong, both now fixed.
    # It ran from the project root, while irace runs the target runner from
    # execDir, so it never exercised the path resolution that was broken. And
    # it tried one configuration, so it could not tell "this works" from "this
    # returns the same number whatever you ask it".
    for algo in mdls nsga2; do
        mkdir -p "outputs/irace-$algo"
        inst=$(ls "$DEM" | head -1)
        run_from_execdir () {
            ( cd "outputs/irace-$algo" && "../../tune/target-runner-$algo" "$@" )
        }
        a=$(run_from_execdir 0 1 1 "$inst" 0 2>/tmp/pf.err || true)
        if [ "$algo" == "mdls" ]; then
            b=$(run_from_execdir 0 1 1 "$inst" 0 --shift 55 --top-pct 0.1 \
                                 --remove-lo 0.5 --remove-hi 0.55 2>>/tmp/pf.err || true)
        else
            b=$(run_from_execdir 0 1 1 "$inst" 0 --pop-size 40 --sbx-eta 3 \
                                 --sbx-prob 0.3 --pm-eta 3 2>>/tmp/pf.err || true)
        fi
        if [ "$a" == "1.0" ] || [ -z "$a" ]; then
            echo "  the $algo runner returns the failure value when run from"
            echo "  outputs/irace-$algo, which is where irace runs it. Every"
            echo "  experiment in its race would tie. Fix this before racing:"
            sed 's/^/    /' /tmp/pf.err
            exit 1
        fi
        if [ "$a" == "$b" ]; then
            echo "  the $algo runner gives the same score ($a) for two very"
            echo "  different configurations, so the race cannot discriminate"
            echo "  and its winner would be arbitrary. Fix this before racing."
            exit 1
        fi
        echo "  preflight $algo: $a and $b for two configurations -- discriminating"
    done

    for algo in mdls nsga2; do
        echo "  --- $algo ---  (one row per iteration; the first takes a while)"
        mkdir -p "outputs/irace-$algo"
        # Not piped through tail: tail buffers until the race ends, and a race
        # is hours, so the terminal sat silent at "--- mdls ---" and looked
        # hung. tee lets it through and keeps a copy.
        Rscript -e "library(irace); \
            abs <- function(p) file.path(normalizePath('.'), p); \
            s <- readScenario(abs('tune/scenario-$algo.txt')); \
            s\$parameterFile <- abs('tune/parameters-$algo.txt'); \
            s\$trainInstancesDir <- abs('outputs/tune_demands'); \
            s\$execDir <- abs('outputs/irace-$algo'); \
            s\$logFile <- abs('outputs/irace-$algo/irace.Rdata'); \
            s\$targetRunner <- abs('tune/target-runner-$algo'); \
            s\$maxExperiments <- $MAXEXP; \
            irace_main(s)" 2>&1 | tee "outputs/irace-$algo/irace.log"
    done
    echo ""
    echo "Best configurations are in outputs/irace-{mdls,nsga2}/irace.Rdata."
    echo "Re-run Sec. 4 with them, on the test set, before quoting anything:"
    echo "  bash run_experiments.sh 15 10000   # with the tuned parameters wired in"
fi
