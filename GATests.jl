# GATests.jl — apples-to-apples benchmark: MDLS vs NSGA-II, NSGA-III, MOEA/D-DE
#
# Run: julia --project=. -t auto GATests.jl
#
# Each algorithm gets BUDGET_SEC wall-clock seconds.
# All three objectives match MDLS:
#   f1 = total ΔV [m/s]
#   f2 = total unassigned service time [days]
#   f3 = vehicles used (active routes)
# Ranked by hypervolume indicator (shared reference point = 1.1 × global max per objective).

import Metaheuristics
import Metaheuristics.PerformanceIndicators: hypervolume
using CSV, DataFrames, Printf, CairoMakie, JLD2

# include infrastructure without running MDLS 
const GATESTS_INCLUDE = true
include("algoMDLS.jl")

#  benchmark settings 
const BUDGET_SEC   = 100.0  # wall-clock seconds per algorithm
const GA_POP       = 100
const MAX_VEHICLES = 30     # maximum fleet size GA decoder may open

# ── precompute decoder data once (avoids rebuilding on every GA evaluation) ───
const _ga_sim            = load_sim()
const _ga_name_to_idx    = Dict(_ga_sim.names[i] => i for i in eachindex(_ga_sim.names))
const _ga_sat_ids        = demands["sat_identifiers"]
const _ga_svc_times      = demands["service_times"]
const _ga_deadlines      = demands["demand_deadlines"]
const _ga_uids           = demands["UIDs"]
const _ga_sim_idx        = [_ga_name_to_idx[_ga_sat_ids[uid]] for uid in _ga_uids]
const _ga_uid_to_pos     = Dict(uid => i for (i, uid) in enumerate(_ga_uids))
const _ga_depot_idxs     = findall(n -> startswith(n, "depot"), _ga_sim.names)
const _ga_depot_names    = [_ga_sim.names[i] for i in _ga_depot_idxs]

# GA decoder
# Encoding: x ∈ [0,1]^n_demands, one priority score per demand UID.
# Assigns each demand in priority order (high score first) to the cheapest
# available vehicle (greedy). Mirrors make_init_schedule: tracks cumulative ΔV
# per vehicle leg (leg_dv) and inserts an intermediate depot return when the
# budget is exceeded, so the GA can represent the same solution space as MDLS.
function ga_decode(x::Vector{Float64};
                   start_time::Float64 = 0.0,
                   refuel_time::Float64 = 0.5,
                   dv_budget::Float64  = 5000.0)

    uids      = _ga_uids
    sat_ids   = _ga_sat_ids
    svc_times = _ga_svc_times
    deadlines = _ga_deadlines

    depot_idxs = _ga_depot_idxs

    # sort demand UIDs by priority score descending
    priority_order = uids[sortperm(x; rev=true)]

    # routes: (dep_idx, cur_sat_idx, cur_dep_time, veh, leg_dv)
    # leg_dv = cumulative ΔV since the last depot visit on this vehicle
    routes = Tuple{Int,Int,Float64,vehicle,Float64}[]
    depot_uid_ctr = 0

    unassigned_uids = Int[]

    for uid in priority_order
        pos     = _ga_uid_to_pos[uid]
        new_idx = _ga_sim_idx[pos]
        svc     = svc_times[uid]
        dl      = deadlines[uid]

        best_vi        = -1
        best_cost      = Inf
        best_arr       = 0.0
        best_via_depot = false
        best_arr_depot = 0.0
        best_dep_depot = 0.0
        best_dv_leg1   = 0.0
        best_dv_leg2   = 0.0

        for (vi, (dep_idx, cur_idx, cur_dep, _, leg_dv)) in enumerate(routes)
            # Option A: direct append (only if within ΔV budget)
            if haskey(mintof_table, (cur_idx, new_idx))
                tof = mintof_table[(cur_idx, new_idx)][1]
                arr = cur_dep + tof
                if arr + svc <= dl
                    dv_direct = min_dv_tab[cur_idx, new_idx]
                    if !isinf(dv_direct) && leg_dv + dv_direct <= dv_budget
                        if dv_direct < best_cost
                            best_cost      = dv_direct
                            best_vi        = vi
                            best_arr       = arr
                            best_via_depot = false
                        end
                    end
                end
            end

            # Option B: return to depot first, then serve demand (budget reset)
            if haskey(mintof_table, (cur_idx, dep_idx)) && haskey(mintof_table, (dep_idx, new_idx))
                tof1  = mintof_table[(cur_idx, dep_idx)][1]
                tof2  = mintof_table[(dep_idx, new_idx)][1]
                arr_d = cur_dep + tof1
                dep_d = arr_d + refuel_time
                arr_n = dep_d + tof2
                if arr_n + svc <= dl
                    dv1 = min_dv_tab[cur_idx, dep_idx]
                    dv2 = min_dv_tab[dep_idx, new_idx]
                    if !isinf(dv1) && !isinf(dv2) && dv1 <= dv_budget && dv2 <= dv_budget
                        total_via = dv1 + dv2
                        if total_via < best_cost
                            best_cost      = total_via
                            best_vi        = vi
                            best_arr       = arr_n
                            best_via_depot = true
                            best_arr_depot = arr_d
                            best_dep_depot = dep_d
                            best_dv_leg1   = dv1
                            best_dv_leg2   = dv2
                        end
                    end
                end
            end
        end

        if best_vi != -1
            dep_idx, cur_idx, cur_dep, veh, leg_dv = routes[best_vi]

            if best_via_depot
                # insert intermediate depot return
                dv1 = snap_cost(cost_table, cur_idx, dep_idx, cur_dep, best_arr_depot)
                isinf(dv1) && (dv1 = best_dv_leg1)
                depot_uid_ctr -= 1
                push!(veh.visitedUID,  depot_uid_ctr)
                push!(veh.visitedSAT,  _ga_sim.names[dep_idx])
                push!(veh.arrivals,    best_arr_depot)
                push!(veh.departures,  best_dep_depot)
                push!(veh.costs,       dv1)
                # then append demand leg from depot
                dv2 = snap_cost(cost_table, dep_idx, new_idx, best_dep_depot, best_arr)
                isinf(dv2) && (dv2 = best_dv_leg2)
                push!(veh.visitedUID,  uid)
                push!(veh.visitedSAT,  sat_ids[uid])
                push!(veh.arrivals,    best_arr)
                new_dep = best_arr + svc
                push!(veh.departures,  new_dep)
                push!(veh.costs,       dv2)
                routes[best_vi] = (dep_idx, new_idx, new_dep, veh, dv2)  # leg_dv resets to just dv2
            else
                dv = snap_cost(cost_table, cur_idx, new_idx, cur_dep, best_arr)
                isinf(dv) && (dv = best_cost)
                push!(veh.visitedUID,  uid)
                push!(veh.visitedSAT,  sat_ids[uid])
                push!(veh.arrivals,    best_arr)
                new_dep = best_arr + svc
                push!(veh.departures,  new_dep)
                push!(veh.costs,       dv)
                routes[best_vi] = (dep_idx, new_idx, new_dep, veh, leg_dv + dv)
            end

        elseif length(routes) < MAX_VEHICLES
            dep_idx    = depot_idxs[(length(routes)) % length(depot_idxs) + 1]
            depot_name = _ga_sim.names[dep_idx]
            haskey(mintof_table, (dep_idx, new_idx)) || (push!(unassigned_uids, uid); continue)
            tof = mintof_table[(dep_idx, new_idx)][1]
            arr = start_time + tof
            arr + svc > dl && (push!(unassigned_uids, uid); continue)
            isinf(min_dv_tab[dep_idx, new_idx]) && (push!(unassigned_uids, uid); continue)

            depot_uid_ctr -= 1
            dv = snap_cost(cost_table, dep_idx, new_idx, start_time, arr)
            isinf(dv) && (dv = min_dv_tab[dep_idx, new_idx])
            new_dep = arr + svc
            veh = vehicle(
                [depot_uid_ctr, uid],
                [depot_name,    sat_ids[uid]],
                [start_time,    arr],
                [start_time,    new_dep],
                [0.0,           dv],
            )
            push!(routes, (dep_idx, new_idx, new_dep, veh, dv))
        else
            push!(unassigned_uids, uid)
        end
    end

    # close each route with a final return-to-depot leg
    for (dep_idx, cur_idx, cur_dep, veh, _) in routes
        depot_name = _ga_sim.names[dep_idx]
        tof_ret = haskey(mintof_table, (cur_idx, dep_idx)) ?
                  mintof_table[(cur_idx, dep_idx)][1] : 0.0
        arr_ret = cur_dep + tof_ret
        dv_ret  = snap_cost(cost_table, cur_idx, dep_idx, cur_dep, arr_ret)
        isinf(dv_ret) && (dv_ret = min_dv_tab[cur_idx, dep_idx])
        depot_uid_ctr -= 1
        push!(veh.visitedUID,  depot_uid_ctr)
        push!(veh.visitedSAT,  depot_name)
        push!(veh.arrivals,    arr_ret)
        push!(veh.departures,  arr_ret + refuel_time)
        push!(veh.costs,       dv_ret)
    end

    schedule = [r[4] for r in routes]

    if isempty(unassigned_uids)
        return schedule, nothing
    else
        unas = Dict{String,Any}(
            "UIDs"             => unassigned_uids,
            "sat_identifiers"  => [demands["sat_identifiers"][u] for u in unassigned_uids],
            "service_times"    => [demands["service_times"][u]   for u in unassigned_uids],
            "demand_deadlines" => [demands["demand_deadlines"][u] for u in unassigned_uids],
        )
        return schedule, unas
    end
end

# ── cumulative objective buffer (collects every GA evaluation) ────────────────
# Each GA run resets this buffer, then every evaluation appends its objectives.
# After the run we compute the Pareto front from the full set — matching the
# cumulative archive that MDLS builds across all its iterations.
const _ga_obj_buffer = Ref{Vector{NTuple{3,Float64}}}(NTuple{3,Float64}[])
const _ga_obj_lock   = ReentrantLock()

function _reset_ga_buffer!()
    lock(_ga_obj_lock) do; _ga_obj_buffer[] = NTuple{3,Float64}[]; end
end

function _pareto_front_from_buffer()
    objs = lock(_ga_obj_lock) do; copy(_ga_obj_buffer[]); end
    objs = filter(o -> all(isfinite, o), objs)
    isempty(objs) && return Matrix{Float64}(undef, 0, 3)
    n = length(objs)
    dominated = falses(n)
    for i in 1:n
        dominated[i] && continue
        fi = objs[i]
        for j in 1:n
            (i == j || dominated[j]) && continue
            fj = objs[j]
            if fi[1] <= fj[1] && fi[2] <= fj[2] && fi[3] <= fj[3] &&
               (fi[1] < fj[1]  || fi[2] < fj[2]  || fi[3] < fj[3])
                dominated[j] = true
            end
        end
    end
    nd  = [objs[i] for i in 1:n if !dominated[i]]
    mat = Matrix{Float64}(undef, length(nd), 3)
    for (i, o) in enumerate(nd)
        mat[i, 1] = o[1]; mat[i, 2] = o[2]; mat[i, 3] = o[3]
    end
    return mat
end

#  GA objectives wrapper 
function ga_objectives(x::Vector{Float64})
    sched, unas = ga_decode(x)
    f1 = isempty(sched) ? 0.0 : sum(sum(v.costs) for v in sched)
    f2 = unas === nothing ? 0.0 : sum(unas["service_times"])
    f3 = Float64(length(sched))
    lock(_ga_obj_lock) do; push!(_ga_obj_buffer[], (f1, f2, f3)); end
    return [f1, f2, f3], Float64[], Float64[]
end

# batch method called by Metaheuristics when parallel_evaluation=true
# X is N×D (rows = individuals); returns N×3, N×0, N×0
function ga_objectives(X::Matrix{Float64})
    n = size(X, 1)
    F = Matrix{Float64}(undef, n, 3)
    Threads.@threads for i in 1:n
        f, _, _ = ga_objectives(X[i, :])
        F[i, :] = f
    end
    return F, Matrix{Float64}(undef, n, 0), Matrix{Float64}(undef, n, 0)
end

# ── memetic objectives: decode → polish (2 iters of MDLS operators) → record ─
function ga_objectives_memetic(x::Vector{Float64})
    sched, unas = ga_decode(x)
    isempty(sched) || (sched = polish(sched, demands, cost_table, mintof_table, _ga_sim; n_iters=2))
    f1 = isempty(sched) ? 0.0 : sum(sum(v.costs) for v in sched)
    f2 = unas === nothing ? 0.0 : sum(unas["service_times"])
    f3 = Float64(length(sched))
    lock(_ga_obj_lock) do; push!(_ga_obj_buffer[], (f1, f2, f3)); end
    return [f1, f2, f3], Float64[], Float64[]
end

function ga_objectives_memetic(X::Matrix{Float64})
    n = size(X, 1)
    F = Matrix{Float64}(undef, n, 3)
    Threads.@threads for i in 1:n
        f, _, _ = ga_objectives_memetic(X[i, :])
        F[i, :] = f
    end
    return F, Matrix{Float64}(undef, n, 0), Matrix{Float64}(undef, n, 0)
end

# ── initial solution seeding ──────────────────────────────────────────────────
# Encodes a schedule (Vector{vehicle}) as a GA priority vector x ∈ [0,1]^n.
# Demands served first get the highest x value so ga_decode reproduces the
# same visitation order when sorted descending — giving all GAs the same
# starting point as MDLS (which internally calls make_init_schedule).
function encode_schedule_as_x(sched::Vector{vehicle})
    n = length(_ga_uids)
    x = zeros(n)
    rank = n
    for veh in sched
        for uid in veh.visitedUID
            uid > 0 || continue
            haskey(_ga_uid_to_pos, uid) || continue
            x[_ga_uid_to_pos[uid]] = rank / n
            rank -= 1
        end
    end
    return x
end

const _init_sched, _init_unas = make_init_schedule(demands, _ga_sim; nvehicles=20)
const x_seed   = encode_schedule_as_x(_init_sched)
const _init_f1 = sum(sum(v.costs) for v in _init_sched; init=0.0)
const _init_f2 = _init_unas === nothing ? 0.0 : sum(_init_unas["service_times"])
const _init_f3 = Float64(length(_init_sched))
const _init_fx = ([_init_f1, _init_f2, _init_f3], Float64[], Float64[])

#  helpers
function _save_csv(mat::Matrix{Float64}, path::String)
    open(path, "w") do io
        println(io, "f1_dv,f2_unassigned_time,f3_vehicles")
        for i in 1:size(mat, 1)
            @printf io "%.4f,%.6f,%.0f\n" mat[i,1] mat[i,2] mat[i,3]
        end
    end
    @info "CSV saved" path
end

function _run_ga(alg, label::String, obj_fn=ga_objectives)
    n_vars = length(demands["UIDs"])
    bounds = [zeros(n_vars) ones(n_vars)]'
    _reset_ga_buffer!()
    # seed with the exact same initial solution MDLS starts from (no re-decoding);
    # manually push to buffer so it counts in the cumulative Pareto front
    Metaheuristics.set_user_solutions!(alg, x_seed, _init_fx)
    lock(_ga_obj_lock) do; push!(_ga_obj_buffer[], (_init_f1, _init_f2, _init_f3)); end
    @info "Running $label …" budget_sec=BUDGET_SEC
    t0 = time()
    Metaheuristics.optimize(obj_fn, bounds, alg)
    elapsed = time() - t0
    front = _pareto_front_from_buffer()
    n_eval = length(_ga_obj_buffer[])
    @info "$label done" elapsed_sec=round(elapsed; digits=1) n_evaluated=n_eval front_size=size(front, 1)
    return front
end

#  run MDLS 
@info "Running MDLS …" budget_sec=BUDGET_SEC
t0_mdls    = time()
mdls_arch  = MDLS(100_000, demands, _ga_sim, cost_table, mintof_table, min_dv_tab;
                  time_limit=BUDGET_SEC,
                  init_sol=copy_schedule(_init_sched), init_unassigned=_init_unas)
elapsed_mdls = time() - t0_mdls
@info "MDLS done" elapsed_sec=round(elapsed_mdls; digits=1) n_solutions=length(mdls_arch.solutions)

front_mdls = hcat(
    mdls_arch.total_deltaV,
    mdls_arch.total_serv_time_unassigned,
    Float64.(mdls_arch.total_vehicles_used),
)
front_mdls = front_mdls[vec(all(isfinite, front_mdls; dims=2)), :]

# ── run GA algorithms ─────────────────────────────────────────────────────────
n_vars = length(demands["UIDs"])

alg_nsga2 = Metaheuristics.NSGA2(N=GA_POP, p_m=1.0/n_vars)
alg_nsga2.options.time_limit          = BUDGET_SEC
alg_nsga2.options.parallel_evaluation = true
front_nsga2 = _run_ga(alg_nsga2, "NSGA-II")

alg_smsemoa = Metaheuristics.SMS_EMOA(N=GA_POP, p_m=1.0/n_vars)
alg_smsemoa.options.time_limit = BUDGET_SEC
front_smsemoa = _run_ga(alg_smsemoa, "SMS-EMOA")

H           = 12   # Das-Dennis partitions → C(14,2) = 91 weight vectors for 3 objectives
weights     = Metaheuristics.gen_ref_dirs(3, H)
alg_moead   = Metaheuristics.MOEAD_DE(weights, F=0.5, CR=1.0, p_m=1.0/n_vars)
alg_moead.options.time_limit          = BUDGET_SEC
alg_moead.options.parallel_evaluation = true
front_moead = _run_ga(alg_moead, "MOEA/D-DE")

# memetic NSGA-II: same GA but each decoded solution is polished with 2 iters
# of MDLS operators before recording — fewer evaluations but higher quality
alg_memetic = Metaheuristics.NSGA2(N=GA_POP, p_m=1.0/n_vars)
alg_memetic.options.time_limit          = BUDGET_SEC
alg_memetic.options.parallel_evaluation = true
front_memetic = _run_ga(alg_memetic, "Memetic NSGA-II", ga_objectives_memetic)

# ── hypervolume comparison ────────────────────────────────────────────────────
fronts = [
    ("MDLS",           front_mdls),
    ("NSGA-II",        front_nsga2),
    ("SMS-EMOA",       front_smsemoa),
    ("MOEA/D-DE",      front_moead),
    ("Memetic NSGA-II",front_memetic),
]

# drop empty fronts and rows with any Inf/NaN objective (prevents poisoned reference point)
_finite_rows(m) = m[vec(all(isfinite, m; dims=2)), :]
fronts = [(lbl, _finite_rows(m)) for (lbl, m) in fronts]
fronts = filter(((_, m),) -> size(m, 1) > 0, fronts)

all_pts = vcat([m for (_, m) in fronts]...)
ref     = maximum(all_pts; dims=1)[:] .* 1.1

@printf "\nShared reference point:\n"
@printf "  f1 (ΔV [m/s])              = %.1f\n"  ref[1]
@printf "  f2 (unassigned time [days]) = %.4f\n" ref[2]
@printf "  f3 (vehicles)               = %.0f\n\n" ref[3]

results = [(lbl, size(m, 1), hypervolume(m, ref)) for (lbl, m) in fronts]
sort!(results; by=r -> r[3], rev=true)
best_hv = results[1][3]

@printf "%-12s  %10s  %16s  %8s\n" "Algorithm" "Front size" "Hypervolume" "vs best"
println("─"^54)
for (lbl, n, hv) in results
    @printf "%-12s  %10d  %16.6e  %7.1f%%\n" lbl n hv 100.0 * hv / best_hv
end
println()

# ── save CSVs ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "outputs")
_save_csv(front_mdls,     joinpath(outdir, "ga_pareto_mdls.csv"))
_save_csv(front_nsga2,    joinpath(outdir, "ga_pareto_nsga2.csv"))
_save_csv(front_smsemoa,  joinpath(outdir, "ga_pareto_smsemoa.csv"))
_save_csv(front_moead,    joinpath(outdir, "ga_pareto_moead.csv"))
_save_csv(front_memetic,  joinpath(outdir, "ga_pareto_memetic.csv"))

# ── overlay Pareto plot ───────────────────────────────────────────────────────
fig       = plot_comparison(fronts; budget_sec=BUDGET_SEC)
plot_path = joinpath(outdir, "comparison_pareto.png")
save(plot_path, fig)
@info "Comparison plot saved" plot_path
