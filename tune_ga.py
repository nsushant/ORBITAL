"""
tune_ga.py — hyperparameter sweep for NSGA-III and MOPSO-CD.

Sweeps:
  n_partitions : [12, 15, 20]   (population size via Das-Dennis ref dirs)
  eta          : [5, 10, 20]    (SBX crossover + PM mutation distribution index)

One run per combination per algorithm per scenario. Saves per-scenario CSVs to
outputs/tune_ga_results_{scenario}.csv for use by plot_ga_tuning.py.

Usage:
  python tune_ga.py                          # all 4 scenarios, trial 1
  python tune_ga.py --scenario tight_normal  # single scenario
  python tune_ga.py --trial 2
"""

import argparse, os, time, itertools, glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--scenario",   default="all",
                    help="Scenario name or 'all' to run all 4 scenarios")
parser.add_argument("--trial",      type=int, default=1)
parser.add_argument("--demand-dir", default="outputs/exp_demands")
parser.add_argument("--dv-budget",  type=float, default=5000.0)
parser.add_argument("--n-eval",     type=int, default=10_000)
args = parser.parse_args()

ALL_SCENARIOS = ["tight_normal", "loose_uniform", "tight_low_dv", "loose_high_dv"]
scenarios_to_run = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]

# ── Imports ───────────────────────────────────────────────────────────────────

from loaders import load_sim_name_map, load_demands, load_cost_table_jld2, load_min_tof_table
from pymoo_stuff import OOSProblem, OOSRepair, OOSCrossover, OOSMutation, MOPSO_CD_Repair
from greedy_init import load_greedy_from_json, GreedySampling

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import moocore

# ── Shared resources (loaded once) ────────────────────────────────────────────

COST_FILE     = "outputs/cost_table.jld2"
REFUEL_TIME   = 0.5
ct, ct_meta   = load_cost_table_jld2(COST_FILE)
nm            = load_sim_name_map("outputs/simulation.h5", depot_idx=ct_meta["depot_idx"])
min_tof_table = load_min_tof_table()

# ── Normalisation constants (match aggregate_results.py) ──────────────────────

PENALTY    = 1e6
_V1        = 864_150.0
_V2        = 2_280_000.0
_V2_FRAC   = 0.30
_AVG_ASSET = _V2_FRAC * _V2 + (1 - _V2_FRAC) * _V1
NADIR      = np.array([200 * 5000.0, 200 * _AVG_ASSET * 1.10, 25.0])
REF_NORM   = np.array([1.1, 1.1, 1.1])
OBJ_COLS   = ["f1_dv", "f2_unrecovered_value", "f3_vehicles"]
N_PARTITIONS = [12, 15, 20]
ETAS         = [5, 10, 20]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scenario_ideal(scenario, exp_dir="outputs/exp_results"):
    """Data-driven ideal: per-objective minimum across all algorithms/trials.
    Matches aggregate_results.py exactly. Falls back to zeros if no files exist."""
    files = glob.glob(os.path.join(exp_dir, f"*_{scenario}_*.csv"))
    if not files:
        print(f"  [warn] No exp_results for {scenario} — using ideal=0")
        return np.zeros(3)
    all_pts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if all(c in df.columns for c in OBJ_COLS):
                all_pts.append(df[OBJ_COLS].values)
        except Exception:
            pass
    return np.vstack(all_pts).min(axis=0) if all_pts else np.zeros(3)


def compute_hv_and_knee(F, ideal, rng):
    """Normalised HV + knee matching aggregate_results.py: (pts - ideal) / rng.

    Returns (hv, knee_dv, knee_unrecovered, knee_vehicles) in raw units.
    """
    nan4 = (float("nan"),) * 4
    mask = F[:, 0] < PENALTY
    F = F[mask]
    if len(F) == 0:
        return 0.0, *nan4
    F_ord    = F[:, [0, 2, 1]]          # [dv, vehicles, unrecovered] → [dv, unrecovered, vehicles]
    pts_norm = (F_ord - ideal) / rng
    valid    = np.all(pts_norm < REF_NORM, axis=1)
    if not valid.any():
        return 0.0, *nan4
    F_valid   = F_ord[valid]
    norms_sq  = (pts_norm[valid] ** 2).sum(axis=1)
    hv        = float(moocore.hypervolume(pts_norm[valid], ref=REF_NORM, maximise=False))
    knee_idx  = int(np.argmin(norms_sq))
    k         = F_valid[knee_idx]
    knee_l2   = float(np.sqrt(norms_sq[knee_idx]))
    return hv, float(k[0]), float(k[1]), float(k[2]), knee_l2

# ── Per-scenario sweep ────────────────────────────────────────────────────────

def run_scenario(scenario):
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario}")
    print(f"{'='*70}")

    trial_str   = f"{args.trial:02d}"
    dem_path    = os.path.join(args.demand_dir, f"{scenario}_{trial_str}.jld2")
    greedy_path = os.path.join(args.demand_dir, f"{scenario}_{trial_str}_greedy.json")

    d        = load_demands(dem_path)
    sat_ids  = [nm[s] for s in d["sat_identifiers"]]
    depot_id = nm["depot_1"]
    Ndems    = len(d["demand_deadlines"])

    problem = OOSProblem(
        Ndems         = Ndems,
        maxV          = 25,
        deadlines     = d["demand_deadlines"],
        service_times = d["service_times"],
        asset_values  = d.get("asset_values", None),
        sat_ids       = sat_ids,
        depot_id      = depot_id,
        cost_table    = ct,
        dep_grid      = ct_meta["dep_grid"],
        arr_grid      = ct_meta["arr_grid"],
        min_tof_table = min_tof_table,
        refuel_time   = REFUEL_TIME,
        dv_budget     = args.dv_budget,
    )

    greedy_x = load_greedy_from_json(greedy_path, Ndems) if os.path.exists(greedy_path) else None

    ideal = load_scenario_ideal(scenario)
    rng   = np.maximum(NADIR - ideal, 1e-10)
    print(f"  ideal={ideal.round(2)}  nadir={NADIR.round(2)}")

    records = []

    for n_part, eta in itertools.product(N_PARTITIONS, ETAS):
        ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=n_part)
        pop_size = len(ref_dirs)
        sampling = GreedySampling(greedy_x) if greedy_x is not None else None

        # NSGA-III
        print(f"\n  [NSGA-III] n_partitions={n_part} (pop={pop_size})  eta={eta}")
        algo = NSGA3(
            ref_dirs  = ref_dirs,
            pop_size  = pop_size,
            sampling  = sampling,
            crossover = OOSCrossover(eta=eta),
            mutation  = OOSMutation(eta=eta),
            repair    = OOSRepair(stochastic=True),
        )
        t0 = time.time()
        res = minimize(problem, algo, termination=("n_eval", args.n_eval),
                       seed=args.trial, verbose=False)
        elapsed = time.time() - t0
        hv, k_dv, k_unrecov, k_veh, k_l2 = compute_hv_and_knee(res.F, ideal, rng) if res.F is not None else (0.0, float("nan"), float("nan"), float("nan"), float("nan"))
        n_sol = int((res.F[:, 0] < PENALTY).sum()) if res.F is not None else 0
        print(f"    HV={hv:.4e}  knee_l2={k_l2:.4f}  knee=[dv={k_dv:.0f} unrecov={k_unrecov/1e6:.1f}M veh={k_veh:.0f}]  n_sol={n_sol}  time={elapsed:.1f}s")
        records.append(dict(scenario=scenario, algo="nsga3", n_partitions=n_part,
                            pop_size=pop_size, eta=eta, hv=hv, knee_l2=k_l2,
                            knee_dv=k_dv, knee_unrecovered_M=k_unrecov/1e6,
                            knee_vehicles=k_veh, n_solutions=n_sol, time_s=elapsed))

        # MOPSO-CD
        print(f"  [MOPSO-CD] n_partitions={n_part} (pop={pop_size})  eta={eta}")
        algo_pso = MOPSO_CD_Repair(
            repair       = OOSRepair(stochastic=True),
            pop_size     = pop_size,
            archive_size = pop_size * 2,
            sampling     = sampling,
        )
        t0 = time.time()
        res_pso = minimize(problem, algo_pso, termination=("n_eval", args.n_eval),
                           seed=args.trial, verbose=False)
        elapsed = time.time() - t0
        hv, k_dv, k_unrecov, k_veh, k_l2 = compute_hv_and_knee(res_pso.F, ideal, rng) if res_pso.F is not None else (0.0, float("nan"), float("nan"), float("nan"), float("nan"))
        n_sol = int((res_pso.F[:, 0] < PENALTY).sum()) if res_pso.F is not None else 0
        print(f"    HV={hv:.4e}  knee_l2={k_l2:.4f}  knee=[dv={k_dv:.0f} unrecov={k_unrecov/1e6:.1f}M veh={k_veh:.0f}]  n_sol={n_sol}  time={elapsed:.1f}s")
        records.append(dict(scenario=scenario, algo="pso", n_partitions=n_part,
                            pop_size=pop_size, eta=eta, hv=hv, knee_l2=k_l2,
                            knee_dv=k_dv, knee_unrecovered_M=k_unrecov/1e6,
                            knee_vehicles=k_veh, n_solutions=n_sol, time_s=elapsed))

    df = pd.DataFrame(records)

    print(f"\n  Results for {scenario}:")
    for algo in ["nsga3", "pso"]:
        sub  = df[df["algo"] == algo].sort_values("hv", ascending=False)
        best = sub.iloc[0]
        print(f"  {algo.upper()} best: n_partitions={int(best.n_partitions)}  eta={int(best.eta)}  HV={best.hv:.4e}  knee_unrecov={best.knee_unrecovered_M:.1f}M")

    out_path = f"outputs/tune_ga_results_{scenario}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    return df

# ── Main ──────────────────────────────────────────────────────────────────────

all_records = []
for sc in scenarios_to_run:
    all_records.append(run_scenario(sc))

print(f"\nDone. Ran {len(scenarios_to_run)} scenario(s).")
