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
    schedule   :: Vector{vehicle}
    unassigned :: Union{Nothing, Dict{String,Any}}
    f1 :: Float64   # total ΔV [m/s]
    f2 :: Float64   # total unassigned service time [days]
    f3 :: Float64   # vehicles used
end

function sched_eval(sched::Vector{vehicle},
                    unas::Union{Nothing,Dict{String,Any}}) :: SchedIndividual
    f1 = isempty(sched) ? 0.0 : sum(sum(v.costs) for v in sched)
    f2 = unas === nothing ? 0.0 : sum(unas["service_times"])
    f3 = Float64(length(sched))
    return SchedIndividual(sched, unas, f1, f2, f3)
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
        Dict{String,Any}("UIDs"             => unassigned_uids,
                         "sat_identifiers"  => ctx.sat_ids[ups],
                         "service_times"    => ctx.svc_times[ups],
                         "demand_deadlines" => ctx.deadlines[ups])
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
function pso_inertia_mutate(ind::SchedIndividual, ctx::RunContext;
                             use_timing_mutation::Bool = true) :: SchedIndividual
    result = rand() < 0.5 ? _random_demand_swap(ind, ctx) : ind
    return use_timing_mutation ? perturb_arrivals(result, ctx) : result
end

function run_nsga3(ctx::RunContext;
                   pop_size::Int             = POP_SIZE,
                   budget_evals::Int         = BUDGET_EVALS,
                   H::Int                    = REF_H,
                   p_cross::Float64          = 0.8,
                   use_timing_mutation::Bool = true) :: Tuple{Matrix{Float64}, Float64}

    refs = das_dennis(3, H)
    @info "Running NSGA-III …" budget_evals=budget_evals pop_size=pop_size n_refs=size(refs,1) use_timing_mutation=use_timing_mutation

    # Initialise from greedy solution using GA operators only (no MDLS operators)
    seed_sched = sched_eval(copy_schedule(ctx.init_sched), ctx.init_unas)
    seed_tour  = encode_tour(seed_sched, ctx)
    pop        = Vector{TourIndividual}(undef, pop_size)
    pop[1]     = seed_tour
    Threads.@threads for i in 2:pop_size
        pop[i] = tour_mutate(seed_tour, ctx; use_timing_mutation=false)
    end

    n_evaluated = pop_size
    t0 = time()

    while n_evaluated < budget_evals
        fronts = nondominated_sort(pop)
        rank   = zeros(Int, pop_size)
        for (r, front) in enumerate(fronts), i in front
            rank[i] = r
        end

        offspring = Vector{TourIndividual}(undef, pop_size)
        Threads.@threads for i in 1:pop_size
            a, b = rand(1:pop_size), rand(1:pop_size)
            p1   = rank[a] <= rank[b] ? a : b
            a, b = rand(1:pop_size), rand(1:pop_size)
            p2   = rank[a] <= rank[b] ? a : b

            child = rand() < p_cross ?
                ox_crossover(pop[p1], pop[p2], ctx) : pop[p1]
            offspring[i] = tour_mutate(child, ctx; use_timing_mutation=use_timing_mutation)
        end

        n_evaluated += pop_size
        pop = nsga3_select(vcat(pop, offspring), pop_size, refs)
    end

    elapsed = time() - t0
    front   = _extract_pareto(NTuple{3,Float64}[(ind.f1, ind.f2, ind.f3) for ind in pop])
    @info "NSGA-III done" n_evaluated=n_evaluated elapsed_sec=round(elapsed; digits=1) front_size=size(front,1)
    return front, elapsed
end

# ═════════════════════════════════════════════════════════════════════════════
#  MOEA/D
# ═════════════════════════════════════════════════════════════════════════════

# Precompute T nearest-neighbour weight vectors for each subproblem (Euclidean distance).
function moead_neighbourhoods(weights::Matrix{Float64}, T::Int) :: Matrix{Int}
    K = size(weights, 1)
    B = Matrix{Int}(undef, K, T)
    for i in 1:K
        dists   = [norm(weights[i, :] .- weights[j, :]) for j in 1:K]
        B[i, :] = sortperm(dists)[1:T]
    end
    return B
end

# Weighted Tchebycheff scalarization: g(x|λ,z*) = max_i λ_i |f_i(x) - z*_i|
tchebycheff(f1, f2, f3, λ, z_star) =
    maximum(λ .* abs.([f1, f2, f3] .- z_star))

function run_moead(ctx::RunContext;
                   pop_size::Int             = POP_SIZE,
                   budget_evals::Int         = BUDGET_EVALS,
                   H::Int                    = REF_H,
                   T::Int                    = 15,
                   nr::Int                   = 2,
                   p_cross::Float64          = 0.8,
                   use_timing_mutation::Bool = true) :: Tuple{Matrix{Float64}, Float64}

    weights = das_dennis(3, H)
    K       = size(weights, 1)
    B       = moead_neighbourhoods(weights, T)
    @info "Running MOEA/D …" budget_evals=budget_evals K=K T=T nr=nr use_timing_mutation=use_timing_mutation

    seed_sched = sched_eval(copy_schedule(ctx.init_sched), ctx.init_unas)
    seed_tour  = encode_tour(seed_sched, ctx)
    pop        = Vector{TourIndividual}(undef, K)
    pop[1]     = seed_tour
    Threads.@threads for i in 2:K
        pop[i] = tour_mutate(seed_tour, ctx; use_timing_mutation=false)
    end

    z_star = [minimum(ind.f1 for ind in pop),
              minimum(ind.f2 for ind in pop),
              minimum(ind.f3 for ind in pop)]

    n_evaluated = K
    t0 = time()

    while n_evaluated < budget_evals
        for i in 1:K
            n_evaluated >= budget_evals && break
            nb             = B[i, :]
            p1_idx, p2_idx = nb[rand(1:T)], nb[rand(1:T)]

            child = rand() < p_cross ?
                ox_crossover(pop[p1_idx], pop[p2_idx], ctx) : pop[p1_idx]
            child = tour_mutate(child, ctx; use_timing_mutation=use_timing_mutation)
            n_evaluated += 1

            z_star[1] = min(z_star[1], child.f1)
            z_star[2] = min(z_star[2], child.f2)
            z_star[3] = min(z_star[3], child.f3)

            replaced = 0
            for j in nb
                replaced >= nr && break
                λ = weights[j, :]
                if tchebycheff(child.f1, child.f2, child.f3, λ, z_star) <=
                   tchebycheff(pop[j].f1, pop[j].f2, pop[j].f3, λ, z_star)
                    pop[j]   = child
                    replaced += 1
                end
            end
        end
    end

    elapsed = time() - t0
    front   = _extract_pareto(NTuple{3,Float64}[(ind.f1, ind.f2, ind.f3) for ind in pop])
    @info "MOEA/D done" n_evaluated=n_evaluated elapsed_sec=round(elapsed; digits=1) front_size=size(front,1)
    return front, elapsed
end

# ═════════════════════════════════════════════════════════════════════════════
#  PARTICLE SWARM OPTIMISATION (MOPSO)
# ═════════════════════════════════════════════════════════════════════════════

mutable struct Particle
    position :: SchedIndividual
    pbest    :: SchedIndividual
end

mutable struct PSOArchive
    members  :: Vector{SchedIndividual}
    max_size :: Int
end
PSOArchive(max_size::Int) = PSOArchive(SchedIndividual[], max_size)

# Crowding distances within the archive over 3 objectives.
function _archive_crowding(arch::PSOArchive) :: Vector{Float64}
    m = length(arch.members)
    m <= 2 && return fill(Inf, m)
    dist = zeros(Float64, m)
    for obj_k in 1:3
        vals  = obj_k == 1 ? [ind.f1 for ind in arch.members] :
                obj_k == 2 ? [ind.f2 for ind in arch.members] :
                             [ind.f3 for ind in arch.members]
        order = sortperm(vals)
        dist[order[1]]   = Inf
        dist[order[end]] = Inf
        rng = vals[order[end]] - vals[order[1]]
        rng < 1e-10 && continue
        for j in 2:m-1
            dist[order[j]] += (vals[order[j+1]] - vals[order[j-1]]) / rng
        end
    end
    return dist
end

# Add ind to archive: reject if dominated, remove dominated members,
# trim most-crowded member if over capacity.
function _pso_archive_add!(arch::PSOArchive, ind::SchedIndividual)
    for m in arch.members
        _dominates(m, ind) && return
    end
    filter!(m -> !_dominates(ind, m), arch.members)
    push!(arch.members, ind)
    while length(arch.members) > arch.max_size
        cd    = _archive_crowding(arch)
        worst = argmin(cd)
        deleteat!(arch.members, worst)
    end
end

# Select gbest via binary tournament — prefer higher crowding distance (more diverse).
function _pso_archive_gbest(arch::PSOArchive) :: SchedIndividual
    length(arch.members) == 1 && return arch.members[1]
    cd = _archive_crowding(arch)
    a  = rand(1:length(arch.members))
    b  = rand(1:length(arch.members))
    return cd[a] >= cd[b] ? arch.members[a] : arch.members[b]
end

# Update personal best via Pareto dominance; random accept when neither dominates.
function _pso_update_pbest(particle::Particle, new_pos::SchedIndividual) :: SchedIndividual
    _dominates(new_pos, particle.pbest) && return new_pos
    _dominates(particle.pbest, new_pos) && return particle.pbest
    return rand() < 0.5 ? new_pos : particle.pbest
end

function run_pso(ctx::RunContext;
                 pop_size::Int             = POP_SIZE,
                 budget_evals::Int         = BUDGET_EVALS,
                 c1::Float64               = 0.3,
                 c2::Float64               = 0.3,
                 archive_size::Int         = 50,
                 use_timing_mutation::Bool = true) :: Tuple{Matrix{Float64}, Float64}

    @info "Running PSO …" budget_evals=budget_evals pop_size=pop_size c1=c1 c2=c2 use_timing_mutation=use_timing_mutation

    seed_sched = sched_eval(copy_schedule(ctx.init_sched), ctx.init_unas)
    seed_tour  = encode_tour(seed_sched, ctx)
    particles  = Vector{Particle}(undef, pop_size)
    particles[1] = Particle(seed_sched, seed_sched)
    Threads.@threads for i in 2:pop_size
        ind = decode_tour(tour_mutate(seed_tour, ctx; use_timing_mutation=false), ctx)
        particles[i] = Particle(ind, ind)
    end

    arch = PSOArchive(archive_size)
    for p in particles
        _pso_archive_add!(arch, p.position)
    end

    n_evaluated = pop_size
    t0 = time()

    while n_evaluated < budget_evals
        for particle in particles
            n_evaluated >= budget_evals && break
            new_pos = particle.position

            # Social: BRX toward gbest
            if rand() < c2 && !isempty(arch.members)
                new_pos = brx_crossover(_pso_archive_gbest(arch), new_pos, ctx)
            end

            # Cognitive: BRX toward pbest
            if rand() < c1
                new_pos = brx_crossover(particle.pbest, new_pos, ctx)
            end

            # Inertia: combinatorial demand swap + arrivals perturbation (replaces sched_mutate)
            new_pos = pso_inertia_mutate(new_pos, ctx; use_timing_mutation=use_timing_mutation)

            n_evaluated += 1
            particle.pbest    = _pso_update_pbest(particle, new_pos)
            particle.position = new_pos
            _pso_archive_add!(arch, new_pos)
        end
    end

    elapsed = time() - t0
    front   = _extract_pareto(NTuple{3,Float64}[(m.f1, m.f2, m.f3) for m in arch.members])
    @info "PSO done" n_evaluated=n_evaluated elapsed_sec=round(elapsed; digits=1) front_size=size(front,1) archive_size=length(arch.members)
    return front, elapsed
end

# ═════════════════════════════════════════════════════════════════════════════
#  MULTI-ALGORITHM WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

function _ctx_with_init(ctx::RunContext, sched::Vector{vehicle}, unas) :: RunContext
    RunContext(ctx.demands, ctx.sim, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab,
               ctx.name_to_idx, ctx.svc_times, ctx.deadlines, ctx.sat_ids, ctx.uids,
               ctx.uid_to_pos, ctx.depot_idxs, ctx.depot_names,
               sched, unas, ctx.adaptive_cost_table, ctx.dv_budget)
end

function run_all_algorithms(ctx::RunContext;
                            budget_evals::Int = BUDGET_EVALS,
                            init_copies = nothing) :: Dict{String, Tuple{Matrix{Float64}, Float64}}
    # Unpack 4 explicit deep copies, or make them now if not provided
    c1, c2, c3, c4 = if init_copies !== nothing
        init_copies
    else
        ntuple(_ -> (copy_schedule(ctx.init_sched), ctx.init_unas), 4)
    end

    # MDLS applies 10 operators per iteration; scale maxiter so total evals ≈ budget_evals
    mdls_iters = max(1, budget_evals ÷ 9)
    @info "Running MDLS …" budget_evals=budget_evals mdls_iters=mdls_iters
    t0 = time()
    arch = MDLS(mdls_iters, ctx.demands, ctx.sim, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab;
                init_sol        = c1[1],
                init_unassigned = c1[2],
                dv_budget       = ctx.dv_budget)
    elapsed_mdls = time() - t0
    @info "MDLS done" elapsed_sec=round(elapsed_mdls; digits=1) n_solutions=length(arch.solutions)

    front_mdls = hcat(arch.total_deltaV,
                      arch.total_serv_time_unassigned,
                      Float64.(arch.total_vehicles_used))
    front_mdls = front_mdls[vec(all(isfinite, front_mdls; dims=2)), :]

    front_nsga3, t_nsga3 = run_nsga3(_ctx_with_init(ctx, c2[1], c2[2]); budget_evals=budget_evals)
    front_moead, t_moead = run_moead(_ctx_with_init(ctx, c3[1], c3[2]); budget_evals=budget_evals)
    front_pso,   t_pso   = run_pso(_ctx_with_init(ctx,  c4[1], c4[2]); budget_evals=budget_evals)

    return Dict(
        "MDLS"     => (front_mdls, elapsed_mdls),
        "NSGA-III" => (front_nsga3, t_nsga3),
        "MOEA/D"   => (front_moead, t_moead),
        "PSO"      => (front_pso,   t_pso),
    )
end

# ═════════════════════════════════════════════════════════════════════════════
#  SINGLE-INSTANCE RUN (skipped when included from numerical_experiments.jl)
# ═════════════════════════════════════════════════════════════════════════════

if !@isdefined(NUMEXP_INCLUDE) || !NUMEXP_INCLUDE

# ── load data (standalone run only) ──────────────────────────────────────────
demands      = load_demands()
cost_table   = load_cost_table()
mintof_table = build_min_tof_table()
min_dv_tab   = build_min_dv_table(cost_table, maximum(k[1] for k in keys(cost_table)))
_ga_sim      = load_sim()

init_sched, init_unas = make_init_schedule(demands, _ga_sim; nvehicles=20)
init_unas !== nothing &&
    @warn "Initial schedule has unassigned demands" n=length(init_unas["UIDs"])

_ctx     = make_context(demands, _ga_sim, cost_table, mintof_table, min_dv_tab)
_results = run_all_algorithms(_ctx; budget_evals = BUDGET_EVALS)

# ═════════════════════════════════════════════════════════════════════════════
#  HYPERVOLUME COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

const ALG_ORDER = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]

_finite_rows(m) = m[vec(all(isfinite, m; dims=2)), :]

fronts = [(lbl, _finite_rows(_results[lbl][1])) for lbl in ALG_ORDER
          if haskey(_results, lbl)]
filter!(((_, m),) -> size(m, 1) > 0, fronts)

all_pts = vcat([m for (_, m) in fronts]...)
ref     = maximum(all_pts; dims=1)[:] .* 1.1

@printf "\nShared reference point:\n"
@printf "  f1 (ΔV [m/s])              = %.1f\n"  ref[1]
@printf "  f2 (unassigned time [days]) = %.4f\n" ref[2]
@printf "  f3 (vehicles)               = %.0f\n\n" ref[3]

_hv_rows = [(lbl, size(m, 1), hypervolume(m, ref), _results[lbl][2]) for (lbl, m) in fronts]
sort!(_hv_rows; by = r -> r[3], rev = true)
_best_hv = _hv_rows[1][3]

@printf "%-14s  %10s  %16s  %8s  %10s\n" "Algorithm" "Front size" "Hypervolume" "vs best" "Wall [s]"
println("─"^64)
for (lbl, n, hv, t) in _hv_rows
    @printf "%-14s  %10d  %16.6e  %7.1f%%  %9.1fs\n" lbl n hv 100.0 * hv / _best_hv t
end
println()

# ── save CSVs ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "outputs")

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
