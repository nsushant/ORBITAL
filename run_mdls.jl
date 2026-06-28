const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2

# Override with adaptive cost table
@info "Loading adaptive cost table …"
cost_table_adaptive = load(joinpath(@__DIR__, "outputs", "cost_table_adaptive.jld2"), "d")
@info "Loaded adaptive cost table" n_entries=length(cost_table_adaptive)

min_dv_tab_adaptive = build_min_dv_table(cost_table_adaptive, maximum(k[1] for k in keys(cost_table_adaptive)))

maxiter = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
@info "Running MDLS" maxiter

sim_obj = load_sim()
archive = MDLS(maxiter, demands, sim_obj, cost_table_adaptive, mintof_table, min_dv_tab_adaptive)

@info "MDLS complete" n_solutions=length(archive.solutions)

csv_path = joinpath(@__DIR__, "outputs", "ga_pareto_mdls.csv")
open(csv_path, "w") do io
    println(io, "f1_dv,f2_unassigned_time,f3_vehicles")
    for i in eachindex(archive.solutions)
        line = "$(round(archive.total_deltaV[i]; digits=4))," *
               "$(round(archive.total_serv_time_unassigned[i]; digits=6))," *
               "$(archive.total_vehicles_used[i])"
        println(io, line)
    end
end
@info "Saved MDLS Pareto front" path=csv_path n=length(archive.solutions)
