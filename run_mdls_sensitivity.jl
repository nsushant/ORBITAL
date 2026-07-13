# run_mdls_sensitivity.jl — run MDLS with a single OAT parameter override.
# CLI: julia --project=. run_mdls_sensitivity.jl <key> <trial> <param_name> <param_level> <h5_file> [demand_dir] [dv_budget]
#
# param_name  : "shift" | "top_pct"
# param_level : float value to use for that parameter
# h5_file     : path to the shared HDF5 output file
#
# Writes dataset /{algo}/{param_name}/{param_level}/trial_{nn} → (n_solutions × 3)
# Columns: f1_dv, f2_unrecovered_value, f3_vehicles

length(ARGS) >= 5 || error("Usage: julia run_mdls_sensitivity.jl <key> <trial> <param_name> <param_level> <h5_file> [demand_dir] [dv_budget]")

scenario_name = ARGS[1]
trial_num     = parse(Int, ARGS[2])
param_name    = ARGS[3]
param_level   = parse(Float64, ARGS[4])
h5_file       = ARGS[5]
EXP_DIR       = length(ARGS) >= 6 ? ARGS[6] : joinpath(@__DIR__, "outputs", "exp_demands")
DV_BUDGET     = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 5000.0

const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2, HDF5

const MDLS_ITERS  = 3334
const N_VEHICLES  = 20
const REFUEL_TIME = 0.5

# Nominal parameters
opt_shift   = 15.0
opt_top_pct = 0.5

if param_name == "shift"
    opt_shift = param_level
elseif param_name == "top_pct"
    opt_top_pct = param_level
else
    error("Unknown param_name: $param_name. Choose 'shift' or 'top_pct'.")
end

trial_str = lpad(trial_num, 2, '0')
dem_path  = joinpath(EXP_DIR, "$(scenario_name)_$(trial_str).jld2")
isfile(dem_path) || error("Demand file not found: $dem_path")

@info "Loading demands" scenario=scenario_name trial=trial_num param=param_name level=param_level
local demands
@load dem_path demands
demands["UIDs"] = collect(1:length(demands["sat_identifiers"]))

@info "Loading tables …"
ct     = load(joinpath(@__DIR__, "outputs", "cost_table.jld2"), "CostTable")
n_sats = maximum(k[1] for k in keys(ct))
mintof = build_min_tof_table()
min_dv = build_min_dv_table(ct, n_sats)
sim    = load_sim()

init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=N_VEHICLES, refuel_time=REFUEL_TIME, dv_budget=DV_BUDGET)

@info "Running MDLS" param=param_name level=param_level
t0 = time()
archive = MDLS(MDLS_ITERS, demands, sim, ct, mintof, min_dv;
               nvehicles          = N_VEHICLES,
               dv_budget          = DV_BUDGET,
               init_sol           = init_sol,
               init_unassigned    = init_unas,
               opt_times_shift    = opt_shift,
               opt_times_top_pct  = opt_top_pct)
elapsed = time() - t0
@info "Done" elapsed_sec=round(elapsed; digits=1) n_solutions=length(archive.solutions)

rows = [(archive.total_deltaV[i],
         archive.total_serv_time_unassigned[i],
         archive.total_vehicles_used[i])
        for i in eachindex(archive.solutions)
        if archive.total_deltaV[i] < INFEASIBLE_LEG_COST]

data = Matrix{Float64}(undef, length(rows), 3)
for (i, (dv, us, veh)) in enumerate(rows)
    data[i, 1] = dv
    data[i, 2] = us
    data[i, 3] = Float64(veh)
end

# Write to shared HDF5 file
level_str  = replace(string(param_level), "." => "p")
group_path = "mdls/$(param_name)/$(level_str)/trial_$(trial_str)"

h5open(h5_file, "cw") do fid
    if haskey(fid, group_path)
        delete_object(fid, group_path)
    end
    fid[group_path] = collect(data')
    attrs(fid[group_path])["columns"] = "f1_dv,f2_unrecovered_value,f3_vehicles"
end

@info "Saved to HDF5" path=h5_file group=group_path n_solutions=length(rows)
