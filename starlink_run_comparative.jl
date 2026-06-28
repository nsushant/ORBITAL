const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2, JSON3

# ── Load shared data ─────────────────────────────────────────────────────────
@info "Loading demands and TOF table …"
demands      = load_demands()
mintof_table = build_min_tof_table()

# ── Load basic cost table ─────────────────────────────────────────────────────
@info "Loading basic cost table …"
cost_table_basic = load(joinpath(@__DIR__, "outputs", "cost_table.jld2"), "CostTable")
n_sats = maximum(k[1] for k in keys(cost_table_basic))
min_dv_tab_basic = build_min_dv_table(cost_table_basic, n_sats)

# ── Greedy initial solution ───────────────────────────────────────────────────
@info "Building greedy initial solution …"
sim_obj = load_sim()
init_sol, init_unassigned = make_init_schedule(demands, sim_obj; nvehicles=20)
@info "Greedy done" n_vehicles=length(init_sol)

greedy_path = joinpath(@__DIR__, "outputs", "greedy_init.json")
save_schedule_json(init_sol, init_unassigned, greedy_path)
@info "Saved greedy init" path=greedy_path

# ── MDLS ─────────────────────────────────────────────────────────────────────
# 2-3 operators fire per iteration (~2.5 avg) → 4000 iters ≈ 10 000 solution evaluations
maxiter = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4000
@info "Running MDLS" maxiter

archive = MDLS(maxiter, demands, sim_obj, cost_table_basic, mintof_table, min_dv_tab_basic;
               nvehicles=20, init_sol=init_sol, init_unassigned=init_unassigned)

@info "MDLS complete" n_solutions=length(archive.solutions)

csv_path = joinpath(@__DIR__, "outputs", "mdls_pareto_comparative.csv")
rows = [(archive.total_deltaV[i],
         archive.total_serv_time_unassigned[i],
         archive.total_vehicles_used[i])
        for i in eachindex(archive.solutions)
        if archive.total_deltaV[i] < INFEASIBLE_LEG_COST]

open(csv_path, "w") do io
    println(io, "f1_dv,f2_unassigned_time,f3_vehicles")
    for (dv, us, veh) in rows
        println(io, "$(round(dv; digits=4)),$(round(us; digits=6)),$veh")
    end
end
n_penalty = length(archive.solutions) - length(rows)
@info "Saved MDLS Pareto" path=csv_path valid=length(rows) penalty_filtered=n_penalty
