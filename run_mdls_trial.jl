# run_mdls_trial.jl — run MDLS on one (key, trial) pair.
# CLI: julia --project=. -t auto run_mdls_trial.jl <key> <trial> [demand_dir] [result_dir] [dv_budget]
#
# Positional args:
#   key         — identifies the demand file, e.g. "tight_normal" or "size_10"
#   trial       — integer trial number
#   demand_dir  — (optional) directory containing demand JLD2 files
#                 default: outputs/exp_demands
#   result_dir  — (optional) directory to write result CSVs
#                 default: outputs/exp_results
#   dv_budget   — (optional) ΔV budget per inter-depot leg [m/s]
#                 default: 5000.0

length(ARGS) >= 2 || error("Usage: julia run_mdls_trial.jl <key> <trial> [demand_dir] [result_dir] [dv_budget]")
scenario_name = ARGS[1]
trial_num     = parse(Int, ARGS[2])

const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2

const MDLS_ITERS   = 3334   # 3 ops/iter × 3334 ≈ 10 000 evals
const N_VEHICLES   = 20
const REFUEL_TIME  = 0.5    # days — must match GA (run_ga_trial.py)

EXP_DIR   = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "outputs", "exp_demands")
RES_DIR   = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "outputs", "exp_results")
DV_BUDGET = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 5000.0
mkpath(RES_DIR)

trial_str = lpad(trial_num, 2, '0')
dem_path  = joinpath(EXP_DIR, "$(scenario_name)_$(trial_str).jld2")
isfile(dem_path) ||
    error("Demand file not found: $dem_path — run generate_experiment_demands.jl first")

@info "Loading demands" scenario=scenario_name trial=trial_num
local demands
@load dem_path demands
demands["UIDs"] = collect(1:length(demands["sat_identifiers"]))

@info "Loading Starlink tables …"
ct     = load(joinpath(@__DIR__, "outputs", "cost_table.jld2"), "CostTable")
n_sats = maximum(k[1] for k in keys(ct))
mintof = build_min_tof_table()
min_dv = build_min_dv_table(ct, n_sats)
sim    = load_sim()

@info "Building greedy warm start …"
init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=N_VEHICLES, refuel_time=REFUEL_TIME, dv_budget=DV_BUDGET)

@info "Running MDLS" scenario=scenario_name trial=trial_num iters=MDLS_ITERS
t0 = time()
archive = MDLS(MDLS_ITERS, demands, sim, ct, mintof, min_dv;
               nvehicles        = N_VEHICLES,
               dv_budget        = DV_BUDGET,
               init_sol         = init_sol,
               init_unassigned  = init_unas)
elapsed = time() - t0

@info "MDLS done" elapsed_sec=round(elapsed; digits=1) n_solutions=length(archive.solutions)

outpath = joinpath(RES_DIR, "mdls_$(scenario_name)_$(trial_str).csv")
rows = [(archive.total_deltaV[i],
         archive.total_serv_time_unassigned[i],
         archive.total_vehicles_used[i])
        for i in eachindex(archive.solutions)
        if archive.total_deltaV[i] < INFEASIBLE_LEG_COST]

open(outpath, "w") do io
    println(io, "f1_dv,f2_unrecovered_value,f3_vehicles")
    for (dv, us, veh) in rows
        println(io, "$(round(dv; digits=4)),$(round(us; digits=6)),$veh")
    end
end
@info "Saved front" path=outpath valid=length(rows) elapsed_sec=round(elapsed; digits=1)
