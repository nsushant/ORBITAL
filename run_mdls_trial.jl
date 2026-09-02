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
include("bcr_clients.jl")

using JLD2

const MDLS_ITERS   = 1000
const N_VEHICLES   = 100
const REFUEL_TIME  = 0.5    # days — must match GA (run_ga_trial.py)

EXP_DIR   = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "outputs", "exp_demands")
RES_DIR   = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "outputs", "exp_results")
DV_BUDGET = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 5000.0
H5_FILE   = length(ARGS) >= 6 ? ARGS[6] : nothing
mkpath(RES_DIR)

trial_str = lpad(trial_num, 2, '0')
dem_path  = joinpath(EXP_DIR, "$(scenario_name)_$(trial_str).jld2")
isfile(dem_path) ||
    error("Demand file not found: $dem_path — run generate_experiment_demands.jl first")

@info "Loading demands" scenario=scenario_name trial=trial_num
local demands
@load dem_path demands
demands["UIDs"] = collect(1:length(demands["sat_identifiers"]))

@info "Loading cost tables …"
ct_path     = get(ENV, "COST_TABLE_PATH", joinpath(@__DIR__, "outputs", "cost_table.jld2"))
pt_path     = get(ENV, "PHASING_TABLE_PATH", joinpath(@__DIR__, "outputs", "phasing_time_table.jld2"))
mintof_path = get(ENV, "MIN_TOF_PATH", nothing)

ct     = load(ct_path, "CostTable")
n_sats = maximum(k[1] for k in keys(ct))
mintof = if mintof_path === nothing
    build_min_tof_table()
elseif isfile(mintof_path)
    load(mintof_path, "MinTOFTable")
else
    build_min_tof_table(; ct_path=ct_path, pt_path=pt_path, out_path=mintof_path, force=true)
end
min_dv = build_min_dv_table(ct, n_sats)
sim    = load_sim()
@info "Tables loaded" ct_path n_entries=length(ct)

@info "Building greedy warm start …"
init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=N_VEHICLES, refuel_time=REFUEL_TIME, dv_budget=DV_BUDGET, min_tof_table=mintof)

@info "Running MDLS" scenario=scenario_name trial=trial_num iters=MDLS_ITERS
t0 = time()
archive = MDLS(MDLS_ITERS, demands, sim, ct, mintof, min_dv;
               nvehicles        = N_VEHICLES,
               dv_budget        = DV_BUDGET,
               init_sol         = init_sol,
               init_unassigned  = init_unas)
elapsed = time() - t0

@info "MDLS done" elapsed_sec=round(elapsed; digits=1) n_solutions=length(archive.solutions)

sat_clients = load_sat_clients(sim=sim)
client_ids  = ordered_client_ids(sat_clients)
tdv_k       = demand_client_tdv(demands, sat_clients)
data, valid = client_front_matrix(archive, sat_clients, client_ids)

outpath = joinpath(RES_DIR, "mdls_$(scenario_name)_$(trial_str).csv")
open(outpath, "w") do io
    println(io, client_column_names(client_ids))
    for j in 1:size(data, 2)
        print(io, join((round(data[r, j]; digits=6) for r in 1:size(data, 1)), ","))
        println(io)
    end
end
@info "Saved front" path=outpath valid=size(data, 2) elapsed_sec=round(elapsed; digits=1) clients=client_ids

if H5_FILE !== nothing && size(data, 2) > 0
    using HDF5
    group_path = "mdls/$(scenario_name)/trial_$(trial_str)"
    total_demand_value = haskey(demands, "asset_values") ?
        sum(demands["asset_values"]) : 0.0
    h5open(H5_FILE, "cw") do fid
        haskey(fid, group_path) && delete_object(fid, group_path)
        fid[group_path] = data
        attrs(fid[group_path])["columns"]            = client_column_names(client_ids)
        attrs(fid[group_path])["total_demand_value"] = total_demand_value
        attrs(fid[group_path])["clients"]            = join(client_ids, ",")
        attrs(fid[group_path])["fee_model"]          = "implied_contract"
        for c in client_ids
            attrs(fid[group_path])["total_demand_value_$(c)"] = get(tdv_k, c, 0.0)
        end
    end
    @info "Saved to HDF5" path=H5_FILE group=group_path
end
