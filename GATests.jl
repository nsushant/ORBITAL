# GATests.jl — benchmark: MDLS vs NSGA-III vs MOEA/D
#
# All three algorithms share the same domain operators (MDLS operator functions).
# NSGA-III and MOEA/D maintain populations of Vector{vehicle} schedules directly —
# no real-valued encoding. MDLS operator functions (NOT the MDLS() algorithm) are
# used as the mutation step inside both EAs.
#
# Run: julia --project=. -t auto GATests.jl

import Metaheuristics.PerformanceIndicators: hypervolume
using CSV, DataFrames, Printf, CairoMakie, JLD2, LinearAlgebra

const GATESTS_INCLUDE = true
include("algoMDLS.jl")

# ── benchmark settings ────────────────────────────────────────────────────────
const BUDGET_SEC    = 100.0
const BUDGET_EVALS  = 10_000  # default eval budget (one eval = one solution objective call)
const POP_SIZE      = 91     # matches Das-Dennis H=12 → C(14,2) = 91 reference points
const REF_H         = 12
const DV_BUDGET     = 5000.0 # m/s ΔV budget per leg between depot visits (matches MDLS)
const REFUEL_TIME   = 0.5    # days refuelling dwell at depot (matches MDLS)
const MAX_VEHICLES = 30


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

struct SchedIndividual
    schedule     :: Vector{vehicle}
    unassigned   :: Union{Nothing, Dict{String,Any}}
    f1           :: Float64   # total ΔV [m/s]
    f2           :: Float64   # total unrecovered asset value [$]
    f3           :: Float64   # vehicles used
    n_unassigned :: Int       # number of unserviced demands
end

function sched_eval(sched::Vector{vehicle},
                    unas::Union{Nothing,Dict{String,Any}}) :: SchedIndividual
    f1           = isempty(sched) ? 0.0 : sum(sum(v.costs) for v in sched)
    f2           = unas === nothing ? 0.0 : sum(unas["asset_values"])
    f3           = Float64(length(sched))
    n_unassigned = unas === nothing ? 0 : length(unas["UIDs"])
    return SchedIndividual(sched, unas, f1, f2, f3, n_unassigned)
end

# Extract non-dominated front from a flat list of evaluated (f1,f2,f3) tuples.
function _extract_pareto(objs::Vector{NTuple{3,Float64}}) :: Matrix{Float64}
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
            if fj[1] <= fi[1] && fj[2] <= fi[2] && fj[3] <= fi[3] &&
               (fj[1] < fi[1]  || fj[2] < fi[2]  || fj[3] < fi[3])
                dominated[i] = true; break
            end
        end
    end
    nd  = [objs[i] for i in 1:n if !dominated[i]]
    mat = Matrix{Float64}(undef, length(nd), 3)
    for (i, o) in enumerate(nd); mat[i, :] = collect(o); end
    return mat
end

_dominates(a, b) =
    a.f1 <= b.f1 && a.f2 <= b.f2 && a.f3 <= b.f3 &&
    (a.f1 < b.f1  || a.f2 < b.f2  || a.f3 < b.f3)

# Fast non-dominated sort → Vector of fronts (each front = Vector{Int} of pop indices).
function nondominated_sort(pop::AbstractVector)
    n         = length(pop)
    dom_count = zeros(Int, n)
    dom_by    = [Int[] for _ in 1:n]

    for i in 1:n, j in 1:n
        i == j && continue
        if _dominates(pop[i], pop[j])
            push!(dom_by[i], j)
        elseif _dominates(pop[j], pop[i])
            dom_count[i] += 1
        end
    end

    fronts = [Int[]]
    for i in 1:n
        dom_count[i] == 0 && push!(fronts[1], i)
    end
    k = 1
    while !isempty(fronts[k])
        next = Int[]
        for i in fronts[k], j in dom_by[i]
            dom_count[j] -= 1
            dom_count[j] == 0 && push!(next, j)
        end
        push!(fronts, next)
        k += 1
    end
    pop!(fronts)   # remove trailing empty front
    return fronts
end

# Das-Dennis uniform weight vectors on the unit simplex.
# M objectives, H divisions → C(M+H-1, H) points, each row sums to 1.
function das_dennis(M::Int, H::Int) :: Matrix{Float64}
    function _gen(m, rem, prefix)
        m == 1 && return [vcat(prefix, rem)]
        vcat([_gen(m-1, rem-i, vcat(prefix, i)) for i in 0:rem]...)
    end
    pts = _gen(M, H, Int[])
    mat = Matrix{Float64}(undef, length(pts), M)
    for (i, p) in enumerate(pts)
        mat[i, :] = p ./ H
    end
    return mat
end

# ═════════════════════════════════════════════════════════════════════════════
#  PER-INSTANCE CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

struct RunContext
    demands             :: Dict{String,Any}
    sim                 :: Any
    cost_table          :: Dict
    mintof_table        :: Dict
    min_dv_tab          :: Matrix{Float64}
    name_to_idx         :: Dict{String,Int}
    svc_times           :: Vector{Float64}
    deadlines           :: Vector{Float64}
    sat_ids             :: Vector{String}
    uids                :: Vector{Int}
    uid_to_pos          :: Dict{Int,Int}
    depot_idxs          :: Vector{Int}
    depot_names         :: Vector{String}
    init_sched          :: Vector{vehicle}
    init_unas           :: Union{Nothing,Dict{String,Any}}
    adaptive_cost_table :: Union{Nothing,AdaptiveGrid}
    dv_budget           :: Float64
end

function make_context(demands, sim, cost_table, mintof_table, min_dv_tab;
                      nvehicles::Int = 20,
                      dv_budget::Float64 = DV_BUDGET) :: RunContext
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    depot_idxs  = findall(n -> startswith(n, "depot"), sim.names)
    init_sched, init_unas = make_init_schedule(demands, sim; nvehicles=nvehicles,
                                               dv_budget=dv_budget)
    uids        = demands["UIDs"]
    uid_to_pos  = Dict(uid => i for (i, uid) in enumerate(uids))
    adap_ct     = try
        isfile(ADAPTIVE_GRID_PATH) ? load_adaptive_cost_table() : nothing
    catch
        nothing
    end
    RunContext(demands, sim, cost_table, mintof_table, min_dv_tab,
               name_to_idx,
               demands["service_times"], demands["demand_deadlines"],
               demands["sat_identifiers"], uids,
               uid_to_pos,
               depot_idxs, [sim.names[i] for i in depot_idxs],
               init_sched, init_unas,
               adap_ct, dv_budget)
end

# ── MDLS mutation (p = 1.0) ───────────────────────────────────────────────────
# Applies one randomly selected MDLS operator function to a schedule individual.
# These are standalone operator functions from algoMDLS.jl — NOT the MDLS() algorithm.
function sched_mutate(ind::SchedIndividual, ctx::RunContext) :: SchedIndividual
    s, u = ind.schedule, ind.unassigned
    op   = rand(1:7)
    new_s, new_u = try
        if     op == 1; (opt_times_combined(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.sim; ag=ctx.adaptive_cost_table), u)
        elseif op == 2;  consolidate_demands(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.sim,              u; dv_budget=ctx.dv_budget)
        elseif op == 3; (swap_cross_vehicle(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.sim),              u)
        elseif op == 4; (swap_intratour(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.sim),                  u)
        elseif op == 5;  create_vehicle(s, ctx.demands, u, ctx.cost_table, ctx.mintof_table, ctx.sim; dv_budget=ctx.dv_budget)
        elseif op == 6;  destroy_and_repair(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab, ctx.sim, u)
        else;            shaw_removal_repair(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab, ctx.sim, u)
        end
    catch
        return ind   # operator failed — return parent unchanged
    end
    isempty(new_s) && return ind
    return sched_eval(new_s, new_u)
end

# ── Best-Route Crossover (BRX) ────────────────────────────────────────────────
# Transfers the most ΔV-efficient route from parent A into a child built from
# parent B. Displaced demands are reinserted via regret-2 repair.
function brx_crossover(pa::SchedIndividual, pb::SchedIndividual, ctx::RunContext) :: SchedIndividual
    # 1. Find route in pa with best service-time / total-cost ratio
    best_vi, best_ratio = 0, -Inf
    for (vi, veh) in enumerate(pa.schedule)
        real_uids = [uid for uid in veh.visitedUID if uid > 0]
        isempty(real_uids) && continue
        cost = sum(veh.costs)
        cost <= 0.0 && continue
        ratio = sum(ctx.svc_times[uid] for uid in real_uids) / cost
        ratio > best_ratio && (best_ratio = ratio; best_vi = vi)
    end
    best_vi == 0 && return sched_mutate(pb, ctx)   # fallback: no valid donor route

    donor     = pa.schedule[best_vi]
    donor_set = Set(uid for uid in donor.visitedUID if uid > 0)

    # 2. Deep copy pb's schedule; strip donor UIDs and record them as displaced
    child     = copy_schedule(pb.schedule)
    displaced = Int[]
    for veh in child
        to_del = [p for p in eachindex(veh.visitedUID) if veh.visitedUID[p] in donor_set]
        for p in reverse(to_del)
            push!(displaced, veh.visitedUID[p])
            deleteat!(veh.visitedUID,  p)
            deleteat!(veh.visitedSAT,  p)
            deleteat!(veh.arrivals,    p)
            deleteat!(veh.departures,  p)
            deleteat!(veh.costs,       p)
        end
    end
    filter!(veh -> any(uid -> uid > 0, veh.visitedUID), child)

    # 3. Edge case: all pb routes served only the donor's demands
    if isempty(child)
        child = [vehicle(copy(donor.visitedUID), copy(donor.visitedSAT),
                         copy(donor.arrivals),   copy(donor.departures),
                         copy(donor.costs))]
        unas = _merge_unassigned(pb.unassigned, displaced, ctx.demands)
        return sched_eval(child, unas)
    end

    # 4. Splice in donor route; regret-2 repair for displaced demands
    push!(child, vehicle(copy(donor.visitedUID), copy(donor.visitedSAT),
                         copy(donor.arrivals),   copy(donor.departures),
                         copy(donor.costs)))
    out_unassigned = Int[]
    _regret2_insert!(child, displaced, ctx.demands, ctx.cost_table, ctx.mintof_table,
                     ctx.name_to_idx, out_unassigned)
    unas = _merge_unassigned(pb.unassigned, out_unassigned, ctx.demands)
    length(child) > MAX_VEHICLES && (child = child[1:MAX_VEHICLES])
    return sched_eval(child, unas)
end

# ═════════════════════════════════════════════════════════════════════════════
#  NSGA-III
# ═════════════════════════════════════════════════════════════════════════════

# Associate each individual (indexed by front_indices into pop) to the nearest
# reference point via perpendicular distance in normalised objective space.
# Returns (assoc, dist, niche_count).
function nsga3_associate(pop::AbstractVector,
                          front_indices::Vector{Int},
                          refs::Matrix{Float64},
                          ideal::Vector{Float64},
                          nadir::Vector{Float64})
    n_refs = size(refs, 1)
    m      = length(front_indices)
    assoc  = ones(Int, m)   # default to ref 1 — safe fallback for non-finite objectives
    dist   = fill(Inf, m)

    for (k, i) in enumerate(front_indices)
        ind = pop[i]
        f   = [ind.f1, ind.f2, ind.f3]
        fn  = (f .- ideal) ./ max.(nadir .- ideal, 1e-10)
        all(isfinite, fn) || continue   # skip individuals with Inf/NaN objectives
        for r in 1:n_refs
            ref  = refs[r, :]
            proj = dot(fn, ref) / dot(ref, ref)
            d    = norm(fn .- proj .* ref)
            isfinite(d) || continue
            if d < dist[k]
                dist[k]  = d
                assoc[k] = r
            end
        end
    end

    niche_count = zeros(Int, n_refs)
    for r in assoc; niche_count[r] += 1; end
    return assoc, dist, niche_count
end

# NSGA-III survivor selection: fill from fronts in rank order; for the last
# partial front use reference-point niching to select the remaining slots.
function nsga3_select(combined::AbstractVector, target_n::Int,
                      refs::Matrix{Float64})
    fronts = nondominated_sort(combined)
    ideal  = [minimum(ind.f1 for ind in combined),
              minimum(ind.f2 for ind in combined),
              minimum(ind.f3 for ind in combined)]
    nadir  = [maximum(ind.f1 for ind in combined),
              maximum(ind.f2 for ind in combined),
              maximum(ind.f3 for ind in combined)]

    selected_idx = Int[]
    for front in fronts
        if length(selected_idx) + length(front) <= target_n
            append!(selected_idx, front)
        else
            remaining = target_n - length(selected_idx)

            # Niche counts from already-selected individuals
            niche_count = zeros(Int, size(refs, 1))
            if !isempty(selected_idx)
                _, _, niche_count = nsga3_associate(combined, selected_idx, refs, ideal, nadir)
            end

            assoc_front, dist_front, _ = nsga3_associate(combined, front, refs, ideal, nadir)
            candidates = collect(1:length(front))

            for _ in 1:remaining
                min_nc     = minimum(niche_count[assoc_front[c]] for c in candidates)
                min_refs   = unique(assoc_front[c] for c in candidates
                                    if niche_count[assoc_front[c]] == min_nc)
                chosen_ref = rand(min_refs)
                ref_cands  = filter(c -> assoc_front[c] == chosen_ref, candidates)
                chosen_c   = ref_cands[argmin(dist_front[ref_cands])]
                push!(selected_idx, front[chosen_c])
                niche_count[chosen_ref] += 1
                filter!(c -> c != chosen_c, candidates)
            end
            break
        end
    end
    return combined[selected_idx]
end

# ═════════════════════════════════════════════════════════════════════════════
#  TOUR INDIVIDUAL — giant tour encoding for standard EA baselines
# ═════════════════════════════════════════════════════════════════════════════

struct TourIndividual
    tour     :: Vector{Int}        # demand UIDs + 0-separators (giant tour)
    arrivals :: Vector{Float64}    # arrival epoch per position (0.0 for separators)
    f1 :: Float64
    f2 :: Float64
    f3 :: Float64
end

# Encode SchedIndividual → TourIndividual (strips depot stops, inserts 0-separators)
function encode_tour(ind::SchedIndividual, ctx::RunContext) :: TourIndividual
    tour     = Int[]
    arrivals = Float64[]
    for (vi, veh) in enumerate(ind.schedule)
        vi > 1 && (push!(tour, 0); push!(arrivals, 0.0))
        for (pos, uid) in enumerate(veh.visitedUID)
            uid <= 0 && continue
            push!(tour, uid)
            push!(arrivals, veh.arrivals[pos])
        end
    end
    return TourIndividual(tour, arrivals, ind.f1, ind.f2, ind.f3)
end

# Decode TourIndividual → SchedIndividual.
# Mirrors the MDLS vehicle structure exactly:
#   • each vehicle starts AND ends at a depot
#   • intermediate depot returns are inserted whenever cumulative ΔV since the
#     last depot exceeds DV_BUDGET (same 5000 m/s limit as MDLS)
#   • multiple depots are rotated across vehicles (same as create_vehicle in MDLS)
#   • depot entries use negative UIDs (matching MDLS convention)
function decode_tour(ti::TourIndividual, ctx::RunContext;
                     refuel_time::Float64 = REFUEL_TIME) :: SchedIndividual
    ct       = ctx.adaptive_cost_table !== nothing ? ctx.adaptive_cost_table : ctx.cost_table
    n_depots = length(ctx.depot_names)

    # Split tour on 0-separators → one sub-tour per vehicle
    subtours   = Vector{Int}[]
    arr_groups = Vector{Float64}[]
    cur_uids   = Int[]
    cur_arrs   = Float64[]
    for (i, uid) in enumerate(ti.tour)
        if uid == 0
            if !isempty(cur_uids)
                push!(subtours,   copy(cur_uids))
                push!(arr_groups, copy(cur_arrs))
                empty!(cur_uids); empty!(cur_arrs)
            end
        else
            push!(cur_uids, uid)
            push!(cur_arrs, ti.arrivals[i])
        end
    end
    !isempty(cur_uids) && (push!(subtours, cur_uids); push!(arr_groups, cur_arrs))

    assigned     = Set{Int}()
    schedule     = vehicle[]
    next_dep_uid = -1          # negative UIDs for depot stops, same as MDLS

    for (sub_i, (sub, arrs)) in enumerate(zip(subtours, arr_groups))
        isempty(sub) && continue

        # Rotate through available depots by vehicle index (mirrors MDLS create_vehicle)
        depot_name = n_depots > 0 ?
            ctx.depot_names[((sub_i - 1) % n_depots) + 1] :
            ctx.sim.names[1]
        depot_idx  = ctx.name_to_idx[depot_name]
        depot_dep  = 0.5

        v_uids  = Int[next_dep_uid];  next_dep_uid -= 1
        v_sats  = String[depot_name]
        v_arrs  = Float64[depot_dep]
        v_deps  = Float64[depot_dep]
        v_costs = Float64[0.0]

        prev_dep = depot_dep
        prev_idx = depot_idx
        cum_dv   = 0.0

        for (k, uid) in enumerate(sub)
            haskey(ctx.uid_to_pos, uid) || continue
            upos = ctx.uid_to_pos[uid]
            sat  = ctx.sat_ids[upos]
            sidx = ctx.name_to_idx[sat]

            tof_info = get(ctx.mintof_table, (prev_idx, sidx), nothing)
            tof_info === nothing && continue

            leg_dv = snap_cost(ct, prev_idx, sidx, prev_dep, prev_dep + tof_info[1])

            # ΔV budget exceeded → insert intermediate depot return and refuel
            if cum_dv + leg_dv > ctx.dv_budget && prev_idx != depot_idx
                ret_info = get(ctx.mintof_table, (prev_idx, depot_idx), nothing)
                if ret_info !== nothing
                    arr_d = prev_dep + ret_info[1]
                    dep_d = arr_d + refuel_time
                    dv_d  = snap_cost(ct, prev_idx, depot_idx, prev_dep, arr_d)
                    push!(v_uids,  next_dep_uid); next_dep_uid -= 1
                    push!(v_sats,  depot_name)
                    push!(v_arrs,  arr_d)
                    push!(v_deps,  dep_d)
                    push!(v_costs, dv_d)
                    prev_dep = dep_d; prev_idx = depot_idx; cum_dv = 0.0
                    # Recompute transfer from depot to demand after refuel
                    tof_info = get(ctx.mintof_table, (prev_idx, sidx), nothing)
                    tof_info === nothing && continue
                end
            end

            earliest = prev_dep + tof_info[1]
            latest   = ctx.deadlines[upos] - ctx.svc_times[upos]
            earliest > latest && continue

            arr = clamp(arrs[k], earliest, latest)
            dep = arr + ctx.svc_times[upos]
            dv  = snap_cost(ct, prev_idx, sidx, prev_dep, arr)

            push!(v_uids, uid); push!(v_sats, sat)
            push!(v_arrs, arr); push!(v_deps, dep); push!(v_costs, dv)
            push!(assigned, uid)
            prev_dep = dep; prev_idx = sidx
            cum_dv  += dv
        end

        # Every vehicle ends at depot (mirrors MDLS create_vehicle)
        if prev_idx != depot_idx
            ret_info = get(ctx.mintof_table, (prev_idx, depot_idx), nothing)
            if ret_info !== nothing
                arr_ret = prev_dep + ret_info[1]
                dep_ret = arr_ret + refuel_time
                dv_ret  = snap_cost(ct, prev_idx, depot_idx, prev_dep, arr_ret)
                push!(v_uids,  next_dep_uid); next_dep_uid -= 1
                push!(v_sats,  depot_name)
                push!(v_arrs,  arr_ret)
                push!(v_deps,  dep_ret)
                push!(v_costs, dv_ret)
            end
        end

        length(v_uids) > 1 && push!(schedule, vehicle(v_uids, v_sats, v_arrs, v_deps, v_costs))
    end

    unassigned_uids = [uid for uid in ctx.uids if uid ∉ assigned]
    unas = if isempty(unassigned_uids)
        nothing
    else
        ups = [ctx.uid_to_pos[uid] for uid in unassigned_uids]
        _av = get(ctx.demands, "asset_values", nothing)
        _d  = Dict{String,Any}("UIDs"             => unassigned_uids,
                               "sat_identifiers"  => ctx.sat_ids[ups],
                               "service_times"    => ctx.svc_times[ups],
                               "demand_deadlines" => ctx.deadlines[ups])
        _av !== nothing && (_d["asset_values"] = _av[ups])
        _d
    end
    return sched_eval(schedule, unas)
end

# Repair arrivals array in-place: enforce earliest-feasible lower bound and deadline upper bound
function tour_arrivals_repair!(tour::Vector{Int}, arrivals::Vector{Float64}, ctx::RunContext)
    depot_name = isempty(ctx.depot_names) ? ctx.sim.names[1] : ctx.depot_names[1]
    depot_idx  = ctx.name_to_idx[depot_name]
    prev_dep = 0.5; prev_idx = depot_idx

    for i in eachindex(tour)
        uid = tour[i]
        if uid == 0
            arrivals[i] = 0.0
            prev_dep = 0.5; prev_idx = depot_idx
            continue
        end
        haskey(ctx.uid_to_pos, uid) || continue
        upos = ctx.uid_to_pos[uid]
        sat  = ctx.sat_ids[upos]
        sidx = ctx.name_to_idx[sat]
        tof_info = get(ctx.mintof_table, (prev_idx, sidx), nothing)
        min_tof  = tof_info === nothing ? 0.0 : tof_info[1]
        earliest = prev_dep + min_tof
        latest   = ctx.deadlines[upos] - ctx.svc_times[upos]
        arr      = clamp(arrivals[i], earliest, max(earliest, latest))
        arrivals[i] = arr
        prev_dep = arr + ctx.svc_times[upos]
        prev_idx = sidx
    end
end

# Order crossover (OX) on demand UIDs + arithmetic blend on arrival times
function ox_crossover(pa::TourIndividual, pb::TourIndividual, ctx::RunContext) :: TourIndividual
    a_seq  = [(uid, pa.arrivals[i]) for (i, uid) in enumerate(pa.tour) if uid > 0]
    b_seq  = [(uid, pb.arrivals[i]) for (i, uid) in enumerate(pb.tour) if uid > 0]
    n      = length(a_seq)
    n == 0 && return pa

    # Arrival lookup by UID for arithmetic blend
    a_arr = Dict(uid => arr for (uid, arr) in a_seq)
    b_arr = Dict(uid => arr for (uid, arr) in b_seq)

    # Random crossover segment from parent A
    l, r = extrema(rand(1:n, 2))
    seg_uids = Set(a_seq[i][1] for i in l:r)

    # Remaining demands from B in B's order, excluding segment
    remaining = [(uid, arr) for (uid, arr) in b_seq if uid ∉ seg_uids]

    # Child demand sequence: OX layout
    child_pairs = vcat(remaining[1:min(l-1, end)],
                       a_seq[l:r],
                       remaining[min(l, end+1):end])

    # Arithmetic blend arrivals
    child_pairs = [(uid, rand() * get(a_arr, uid, arr) + (1-rand()) * get(b_arr, uid, arr))
                   for (uid, arr) in child_pairs]

    # Reinsert separators: average vehicle count of parents
    n_seps = max(0, round(Int, (count(==(0), pa.tour) + count(==(0), pb.tour)) / 2))
    tour     = Int[]
    arrivals = Float64[]
    step     = n_seps > 0 ? max(1, n ÷ (n_seps + 1)) : n + 1
    next_sep = step

    for (k, (uid, arr)) in enumerate(child_pairs)
        if k == next_sep && n_seps > 0
            push!(tour, 0); push!(arrivals, 0.0)
            next_sep += step; n_seps -= 1
        end
        push!(tour, uid); push!(arrivals, arr)
    end

    tour_arrivals_repair!(tour, arrivals, ctx)
    decoded = decode_tour(TourIndividual(tour, arrivals, 0.0, 0.0, 0.0), ctx)
    return TourIndividual(tour, arrivals, decoded.f1, decoded.f2, decoded.f3)
end

# Tour mutation: combinatorial (swap/insert/separator) + optional arrival perturbation
function tour_mutate(ti::TourIndividual, ctx::RunContext;
                     use_timing_mutation::Bool = true) :: TourIndividual
    tour     = copy(ti.tour)
    arrivals = copy(ti.arrivals)
    demand_pos = findall(>(0), tour)

    if length(demand_pos) >= 2
        op = rand(1:3)
        if op == 1
            # Random swap of two demand positions
            i, j = rand(demand_pos), rand(demand_pos)
            tour[i], tour[j]         = tour[j], tour[i]
            arrivals[i], arrivals[j] = arrivals[j], arrivals[i]
        elseif op == 2
            # Random demand relocation
            i   = rand(demand_pos)
            uid = tour[i]; arr = arrivals[i]
            deleteat!(tour, i); deleteat!(arrivals, i)
            j = rand(1:length(tour)+1)
            insert!(tour, j, uid); insert!(arrivals, j, arr)
        else
            # Separator manipulation
            sep_pos = findall(==(0), tour)
            if rand() < 0.5 && length(sep_pos) > 1
                i = rand(sep_pos)
                deleteat!(tour, i); deleteat!(arrivals, i)
            else
                dp = findall(>(0), tour)
                if length(dp) >= 2
                    i = rand(dp[1:end-1])
                    insert!(tour, i+1, 0); insert!(arrivals, i+1, 0.0)
                end
            end
        end
    end

    if use_timing_mutation
        for i in eachindex(tour)
            tour[i] <= 0 && continue
            uid = tour[i]
            haskey(ctx.uid_to_pos, uid) || continue
            upos    = ctx.uid_to_pos[uid]
            latest  = ctx.deadlines[upos] - ctx.svc_times[upos]
            σ       = 0.1 * max(latest - arrivals[i], 1.0)
            arrivals[i] += randn() * σ
        end
    end

    tour_arrivals_repair!(tour, arrivals, ctx)
    decoded = decode_tour(TourIndividual(tour, arrivals, 0.0, 0.0, 0.0), ctx)
    return TourIndividual(tour, arrivals, decoded.f1, decoded.f2, decoded.f3)
end

# ── PSO helpers: standard inertia mutation (replaces sched_mutate) ────────────

# Recompute arrivals/departures/costs for vehicle v from from_pos to end using earliest-feasible.
# Returns false if any deadline is violated (caller should reject).
function _recompute_vehicle!(v::vehicle, ctx::RunContext, from_pos::Int,
                              ct::Union{Dict,AdaptiveGrid}) :: Bool
    for p in from_pos:length(v.visitedUID)
        from_idx = ctx.name_to_idx[v.visitedSAT[p-1]]
        to_idx   = ctx.name_to_idx[v.visitedSAT[p]]
        uid      = v.visitedUID[p]
        tof_info = get(ctx.mintof_table, (from_idx, to_idx), nothing)
        tof_info === nothing && return false
        arr = v.departures[p-1] + tof_info[1]
        if uid > 0
            haskey(ctx.uid_to_pos, uid) || return false
            upos = ctx.uid_to_pos[uid]
            arr + ctx.svc_times[upos] > ctx.deadlines[upos] && return false
            dep = arr + ctx.svc_times[upos]
        else
            dep = arr + 0.5
        end
        v.arrivals[p]   = arr
        v.departures[p] = dep
        v.costs[p]      = snap_cost(ct, from_idx, to_idx, v.departures[p-1], arr)
    end
    return true
end

# Perturb arrival times in-place on a copy of the schedule (cascade-correct, forward pass).
function perturb_arrivals(ind::SchedIndividual, ctx::RunContext) :: SchedIndividual
    ct    = ctx.adaptive_cost_table !== nothing ? ctx.adaptive_cost_table : ctx.cost_table
    sched = copy_schedule(ind.schedule)
    for veh in sched
        prev_dep = veh.departures[1]
        for i in 2:length(veh.visitedUID)
            uid = veh.visitedUID[i]
            from_idx = ctx.name_to_idx[veh.visitedSAT[i-1]]
            to_idx   = ctx.name_to_idx[veh.visitedSAT[i]]
            tof_info = get(ctx.mintof_table, (from_idx, to_idx), nothing)
            tof_info === nothing && (prev_dep = veh.departures[i]; continue)
            earliest = prev_dep + tof_info[1]
            if uid > 0 && haskey(ctx.uid_to_pos, uid)
                upos   = ctx.uid_to_pos[uid]
                latest = ctx.deadlines[upos] - ctx.svc_times[upos]
                if earliest <= latest
                    σ   = 0.1 * max(latest - earliest, 1.0)
                    arr = clamp(veh.arrivals[i] + randn() * σ, earliest, latest)
                    dep = arr + ctx.svc_times[upos]
                    veh.arrivals[i]   = arr
                    veh.departures[i] = dep
                    veh.costs[i]      = snap_cost(ct, from_idx, to_idx, prev_dep, arr)
                    prev_dep = dep
                    continue
                end
            end
            # Depot or infeasible: push forward
            veh.arrivals[i]   = earliest
            veh.departures[i] = earliest + 0.5
            veh.costs[i]      = snap_cost(ct, from_idx, to_idx, prev_dep, earliest)
            prev_dep = veh.departures[i]
        end
    end
    return sched_eval(sched, ind.unassigned)
end

# Random demand swap between two vehicles (non-improving, purely random).
function _random_demand_swap(ind::SchedIndividual, ctx::RunContext) :: SchedIndividual
    ct = ctx.adaptive_cost_table !== nothing ? ctx.adaptive_cost_table : ctx.cost_table
    length(ind.schedule) < 2 && return ind
    real_vehs = findall(v -> any(>(0), v.visitedUID), ind.schedule)
    length(real_vehs) < 2 && return ind
    sched = copy_schedule(ind.schedule)
    vi    = rand(real_vehs)
    vj    = rand(filter(!=(vi), real_vehs))
    pi_p  = findall(>(0), sched[vi].visitedUID)
    pj_p  = findall(>(0), sched[vj].visitedUID)
    isempty(pi_p) || isempty(pj_p) && return ind
    pi = rand(pi_p); pj = rand(pj_p)
    sched[vi].visitedUID[pi], sched[vj].visitedUID[pj] =
        sched[vj].visitedUID[pj], sched[vi].visitedUID[pi]
    sched[vi].visitedSAT[pi], sched[vj].visitedSAT[pj] =
        sched[vj].visitedSAT[pj], sched[vi].visitedSAT[pi]
    ok1 = _recompute_vehicle!(sched[vi], ctx, pi, ct)
    ok2 = _recompute_vehicle!(sched[vj], ctx, pj, ct)
    (!ok1 || !ok2) && return ind
    return sched_eval(sched, ind.unassigned)
end

# PSO inertia step: combinatorial swap + optional arrival perturbation
# ═════════════════════════════════════════════════════════════════════════════
#  MULTI-ALGORITHM WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

function _ctx_with_init(ctx::RunContext, sched::Vector{vehicle}, unas) :: RunContext
    RunContext(ctx.demands, ctx.sim, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab,
               ctx.name_to_idx, ctx.svc_times, ctx.deadlines, ctx.sat_ids, ctx.uids,
               ctx.uid_to_pos, ctx.depot_idxs, ctx.depot_names,
               sched, unas, ctx.adaptive_cost_table, ctx.dv_budget)
end

function _save_csv(mat::Matrix{Float64}, path::String)
    open(path, "w") do io
        println(io, "f1_dv,f2_unassigned_time,f3_vehicles")
        for i in 1:size(mat, 1)
            @printf io "%.4f,%.6f,%.0f\n" mat[i,1] mat[i,2] mat[i,3]
        end
    end
    @info "CSV saved" path
end

_save_csv(_results["MDLS"][1],     joinpath(outdir, "ga_pareto_mdls.csv"))
_save_csv(_results["NSGA-III"][1], joinpath(outdir, "ga_pareto_nsga3.csv"))
_save_csv(_results["MOEA/D"][1],   joinpath(outdir, "ga_pareto_moead.csv"))
_save_csv(_results["PSO"][1],      joinpath(outdir, "ga_pareto_pso.csv"))

# ── overlay Pareto plot ───────────────────────────────────────────────────────
fig       = plot_spider_comparison(fronts)
plot_path = joinpath(outdir, "comparison_pareto.png")
save(plot_path, fig)
@info "Comparison plot saved" plot_path

end  # if !NUMEXP_INCLUDE
