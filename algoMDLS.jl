include("sim/propagator.jl")
include("demands/generate_demands.jl")
include("sol_utils.jl")
include("plots/gantt.jl")
include("plots/visualise.jl")
include("plots/gradients.jl")
include("fuel_cost_calc/adaptive_grid.jl")


const J2_MDLS = 1.08263e-3
const RE_MDLS = 6371.0
const MU_MDLS = 3.986004418e5




# operators to reduce the delta V of a schedule 

function opt_times_combined(schedule, demands, CostTable, MinTOFTable, sim; top_pct=0.50, shift=15.0, ag::Union{Nothing,AdaptiveGrid}=nothing)
    # Tries two timing block moves per leg and applies the best:
    # C: shift arrivals[i:end]+departures[i:end] later (block — preserves no-wait invariant)
    # D: shift arrivals[1:i-1]+departures[1:i-1] earlier (block — preserves no-wait invariant)

    schedule = copy_schedule(schedule)

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids   = demands["sat_identifiers"]
    svc_times = demands["service_times"]
    deadlines = demands["demand_deadlines"]
    available = demand_ready_vec(demands)

    legs = [(veh.costs[i], v, i) for (v, veh) in enumerate(schedule)
                                  for i in 2:length(veh.visitedUID)]
    sort!(legs, by=x->x[1], rev=true)
    topn = max(1, round(Int, length(legs) * top_pct))

    for n in 1:topn
        _, v, i = legs[n]
        veh = schedule[v]

        from_name = veh.visitedSAT[i-1]
        to_name   = veh.visitedSAT[i]
        from_idx  = name_to_idx[from_name]
        to_idx    = name_to_idx[to_name]
        min_tof   = get(MinTOFTable, (from_idx, to_idx), (0.0, 0.0))[1]
        # Adaptive shift: ±1 cell in the adaptive grid. Falls back to 15 days.
        shift_later, shift_earlier = if ag !== nothing
            local _idx = get(ag.lookup, (from_idx, to_idx), nothing)
            if _idx !== nothing
                _p  = ag.pairs[_idx]
                _id = clamp(searchsortedlast(_p.deps, veh.departures[i-1]), 1, length(_p.deps)-1)
                (max(1.0, _p.deps[min(_id+1, length(_p.deps))] - veh.departures[i-1]),
                 max(1.0, veh.departures[i-1] - _p.deps[max(_id-1, 1)]))
            else
                (15.0, 15.0)
            end
        else
            (shift, shift)
        end

        orig_cost   = legs[n][1]
        best_cost   = orig_cost
        best_move   = :none

        # ── Move C: shift block arrivals+departures[i:end] later ──
        c_arrs = veh.arrivals[i:end] .+ shift_later
        c_deps = veh.departures[i:end] .+ shift_later
        feas_c = c_arrs[1] - veh.departures[i-1] >= min_tof
        if feas_c
            for j in eachindex(c_arrs)
                uid = veh.visitedUID[i + j - 1]
                uid > 0 && c_arrs[j] + svc_times[uid] > deadlines[uid] && (feas_c = false; break)
            end
        end
        if feas_c
            total_c = snap_cost(CostTable, from_idx, to_idx, veh.departures[i-1], c_arrs[1])
            if total_c < best_cost
                best_cost = total_c; best_move = :C
            end
        end

        # ── Move D: shift block arrivals+departures[1:i-1] earlier ──
        d_arrs_up = veh.arrivals[1:i-1] .- shift_earlier
        d_deps_up = veh.departures[1:i-1] .- shift_earlier
        feas_d = d_arrs_up[1] >= 0.0
        if feas_d
            for j in eachindex(d_arrs_up)
                uid = veh.visitedUID[j]
                uid > 0 && d_arrs_up[j] < available[uid] && (feas_d = false; break)
            end
        end
        if feas_d
            total_d = snap_cost(CostTable, from_idx, to_idx, d_deps_up[end], veh.arrivals[i])
            if total_d < best_cost
                best_cost = total_d; best_move = :D
            end
        end

        best_move === :none && continue

        if best_move === :C
            veh.arrivals[i:end]   = c_arrs
            veh.departures[i:end] = c_deps
            veh.costs[i]          = best_cost
        else  # :D
            veh.arrivals[1:i-1]   = d_arrs_up
            veh.departures[1:i-1] = d_deps_up
            veh.costs[i]          = best_cost
        end
    end
    return schedule
end






function consolidate_demands(schedule, demands, CostTable, MinTOFTable, sim,
    unassigned_in=nothing; dv_budget=5000.0, refuel_time=0.5)

    schedule = copy_schedule(schedule)

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids    = demands["sat_identifiers"]
    svc_times  = demands["service_times"]
    deadlines  = demands["demand_deadlines"]
    available  = demand_ready_vec(demands)

    length(schedule) <= 1 && return schedule, unassigned_in

    demand_counts = [count(uid -> uid > 0, veh.visitedUID) for veh in schedule]
    victim_idx = argmin(demand_counts)
    victim = schedule[victim_idx]

    demands_to_move = [(victim.visitedUID[i], victim.visitedSAT[i])
                       for i in 1:length(victim.visitedUID) if victim.visitedUID[i] > 0]
    isempty(demands_to_move) && return schedule, unassigned_in
    sort!(demands_to_move, by = x -> deadlines[x[1]])

    unassigned_uids = Int[]
    deleteat!(schedule, victim_idx)

    all_depot_uids = [uid for veh in schedule for uid in veh.visitedUID if uid < 0]
    next_depot_uid = isempty(all_depot_uids) ? -1 : minimum(all_depot_uids) - 1

    for (i, (uid, sat_name)) in enumerate(demands_to_move)
        svc = svc_times[uid]
        dl  = deadlines[uid]

        best_v     = 0
        best_p     = 0
        best_delta = Inf
        best_mode  = 0

        best_arr_depot = 0.0
        best_dep_depot = 0.0
        best_arr_cand  = 0.0
        best_dep_cand  = 0.0
        best_arr_next  = 0.0
        best_dv_leg1   = 0.0
        best_dv_leg2   = 0.0
        best_dv_leg3   = 0.0

        for (v, veh) in enumerate(schedule)
            n = length(veh.visitedUID)
            home_depot_idx = name_to_idx[veh.visitedSAT[1]]
            cand_idx = name_to_idx[sat_name]

            # ── Phase 1: Mid-tour (1 ≤ p ≤ n-1) ──
            for p in 1:n-1
                prev_name = veh.visitedSAT[p]
                next_name = veh.visitedSAT[p+1]
                prev_idx  = name_to_idx[prev_name]
                next_idx  = name_to_idx[next_name]

                last_dep = 0
                for k in p:-1:1
                    veh.visitedUID[k] < 0 && (last_dep = k; break)
                end
                cum_dv = sum(veh.costs[last_dep+1:p])
                old_dv = veh.costs[p+1]

                tof_prev_cand, _ = get(MinTOFTable, (prev_idx, cand_idx), (Inf, Inf))
                isfinite(tof_prev_cand) || continue

                arr_cand = max(veh.departures[p] + tof_prev_cand, available[uid])
                dep_cand = arr_cand + svc
                dep_cand > dl && continue

                dv_prev_cand = snap_cost(CostTable, prev_idx, cand_idx, veh.departures[p], arr_cand)
                cum_dv + dv_prev_cand > dv_budget && continue

                # Mode A: Direct insertion (prev → cand → next)
                tof_cand_next, _ = get(MinTOFTable, (cand_idx, next_idx), (Inf, Inf))
                if isfinite(tof_cand_next)
                    arr_next_new = dep_cand + tof_cand_next
                    dv_cand_next = snap_cost(CostTable, cand_idx, next_idx, dep_cand, arr_next_new)

                    delta = dv_prev_cand + dv_cand_next - old_dv
                    if delta < best_delta
                        shift = arr_next_new - veh.arrivals[p+1]
                        feasible = true
                        for j in p+2:n
                            uid_j = veh.visitedUID[j]
                            if uid_j > 0 && veh.arrivals[j] + shift + svc_times[uid_j] > deadlines[uid_j]
                                feasible = false
                                break
                            end
                        end
                        if feasible
                            best_v = v;  best_p = p
                            best_mode = 1
                            best_delta = delta
                            best_arr_cand = arr_cand;  best_dep_cand = dep_cand
                            best_arr_next = arr_next_new
                            best_dv_leg1 = dv_prev_cand;  best_dv_leg2 = dv_cand_next
                        end
                    end
                end

                # Mode B: Depot-return insertion (prev → depot → cand → next)
                depot_idx = home_depot_idx
                tof_prev_depot, _ = get(MinTOFTable, (prev_idx, depot_idx), (Inf, Inf))
                isfinite(tof_prev_depot) || continue

                arr_depot = veh.departures[p] + tof_prev_depot
                dep_depot = arr_depot + refuel_time
                dv_prev_depot = snap_cost(CostTable, prev_idx, depot_idx, veh.departures[p], arr_depot)
                dv_prev_depot > dv_budget && continue
                cum_dv + dv_prev_depot > dv_budget && continue

                tof_depot_cand, _ = get(MinTOFTable, (depot_idx, cand_idx), (Inf, Inf))
                isfinite(tof_depot_cand) || continue

                arr_cand2 = max(dep_depot + tof_depot_cand, available[uid])
                dep_cand2 = arr_cand2 + svc
                dep_cand2 > dl && continue
                dv_depot_cand = snap_cost(CostTable, depot_idx, cand_idx, dep_depot, arr_cand2)
                dv_depot_cand > dv_budget && continue

                tof_cand_next2, _ = get(MinTOFTable, (cand_idx, next_idx), (Inf, Inf))
                isfinite(tof_cand_next2) || continue

                arr_next_new2 = dep_cand2 + tof_cand_next2
                dv_cand_next2 = snap_cost(CostTable, cand_idx, next_idx, dep_cand2, arr_next_new2)

                delta2 = dv_prev_depot + dv_depot_cand + dv_cand_next2 - old_dv
                if delta2 < best_delta
                    shift2 = arr_next_new2 - veh.arrivals[p+1]
                    feasible2 = true
                    for j in p+2:n
                        uid_j = veh.visitedUID[j]
                        if uid_j > 0 && veh.arrivals[j] + shift2 + svc_times[uid_j] > deadlines[uid_j]
                            feasible2 = false
                            break
                        end
                    end
                    if feasible2
                        best_v = v;  best_p = p
                        best_mode = 2
                        best_delta = delta2
                        best_arr_depot = arr_depot;  best_dep_depot = dep_depot
                        best_arr_cand = arr_cand2;   best_dep_cand = dep_cand2
                        best_arr_next = arr_next_new2
                        best_dv_leg1 = dv_prev_depot
                        best_dv_leg2 = dv_depot_cand
                        best_dv_leg3 = dv_cand_next2
                    end
                end
            end

            # ── Phase 2: Tour extension (p = n) ──
            if veh.visitedUID[n] < 0
                depot_idx = name_to_idx[veh.visitedSAT[n]]

                tof_depot_cand, _ = get(MinTOFTable, (depot_idx, cand_idx), (Inf, Inf))
                isfinite(tof_depot_cand) || continue

                arr_cand3 = max(veh.departures[n] + tof_depot_cand, available[uid])
                dep_cand3 = arr_cand3 + svc
                dep_cand3 > dl && continue
                dv_depot_cand = snap_cost(CostTable, depot_idx, cand_idx, veh.departures[n], arr_cand3)
                dv_depot_cand > dv_budget && continue

                tof_cand_depot, _ = get(MinTOFTable, (cand_idx, depot_idx), (Inf, Inf))
                isfinite(tof_cand_depot) || continue

                dv_cand_depot = snap_cost(CostTable, cand_idx, depot_idx, dep_cand3, dep_cand3 + tof_cand_depot)

                delta3 = dv_depot_cand + dv_cand_depot
                if delta3 < best_delta
                    best_v = v;  best_p = n
                    best_mode = 3
                    best_delta = delta3
                    best_arr_cand = arr_cand3;  best_dep_cand = dep_cand3
                    best_dv_leg1 = dv_depot_cand;  best_dv_leg2 = dv_cand_depot
                end
            end
        end

        if best_v == 0
            push!(unassigned_uids, uid)
            continue
        end

        veh = schedule[best_v]
        p = best_p
        n = length(veh.visitedUID)

        if best_mode == 2
            # Mode B: Depot-return insertion
            depot_uid = next_depot_uid
            next_depot_uid -= 1
            depot_name = veh.visitedSAT[1]

            new_uids = vcat(veh.visitedUID[1:p], [depot_uid, uid], veh.visitedUID[p+1:n])
            new_sats = vcat(veh.visitedSAT[1:p], [depot_name, sat_name], veh.visitedSAT[p+1:n])
            new_arr  = vcat(veh.arrivals[1:p], [best_arr_depot, best_arr_cand])
            new_dep  = vcat(veh.departures[1:p], [best_dep_depot, best_dep_cand])

            shift = best_arr_next - veh.arrivals[p+1]
            for j in p+1:n
                push!(new_arr, veh.arrivals[j] + shift)
                push!(new_dep, veh.departures[j] + shift)
            end

            new_costs = Float64[0.0]
            for i in 2:p
                push!(new_costs, veh.costs[i])
            end
            push!(new_costs, best_dv_leg1, best_dv_leg2, best_dv_leg3)
            for i in p+2:n
                from_idx = name_to_idx[new_sats[i+1]]
                to_idx   = name_to_idx[new_sats[i+2]]
                dv = snap_cost(CostTable, from_idx, to_idx, new_dep[i+1], new_arr[i+2])
                push!(new_costs, dv)
            end

            veh.visitedUID = new_uids
            veh.visitedSAT = new_sats
            veh.arrivals   = new_arr
            veh.departures = new_dep
            veh.costs      = new_costs

        elseif best_mode == 1
            # Mode A: Direct insertion
            new_uids = vcat(veh.visitedUID[1:p], [uid], veh.visitedUID[p+1:n])
            new_sats = vcat(veh.visitedSAT[1:p], [sat_name], veh.visitedSAT[p+1:n])
            new_arr  = vcat(veh.arrivals[1:p], [best_arr_cand])
            new_dep  = vcat(veh.departures[1:p], [best_dep_cand])

            shift = best_arr_next - veh.arrivals[p+1]
            for j in p+1:n
                push!(new_arr, veh.arrivals[j] + shift)
                push!(new_dep, veh.departures[j] + shift)
            end

            new_costs = Float64[0.0]
            for i in 2:p
                push!(new_costs, veh.costs[i])
            end
            push!(new_costs, best_dv_leg1, best_dv_leg2)
            for i in p+1:n-1
                from_idx = name_to_idx[new_sats[i+1]]
                to_idx   = name_to_idx[new_sats[i+2]]
                dv = snap_cost(CostTable, from_idx, to_idx, new_dep[i+1], new_arr[i+2])
                push!(new_costs, dv)
            end

            veh.visitedUID = new_uids
            veh.visitedSAT = new_sats
            veh.arrivals   = new_arr
            veh.departures = new_dep
            veh.costs      = new_costs

        elseif best_mode == 3
            # Mode C: Tour extension (depot → cand → new_depot)
            depot_uid = next_depot_uid
            next_depot_uid -= 1
            depot_name = veh.visitedSAT[n]
            cand_idx2 = name_to_idx[sat_name]
            depot_idx2 = name_to_idx[depot_name]

            tof_cand_depot, _ = get(MinTOFTable, (cand_idx2, depot_idx2), (Inf, Inf))
            arr_depot2 = best_dep_cand + tof_cand_depot
            dep_depot2 = arr_depot2 + refuel_time

            new_uids = vcat(veh.visitedUID, [uid, depot_uid])
            new_sats = vcat(veh.visitedSAT, [sat_name, depot_name])
            new_arr  = vcat(veh.arrivals, [best_arr_cand, arr_depot2])
            new_dep  = vcat(veh.departures, [best_dep_cand, dep_depot2])
            new_costs = vcat(veh.costs, [best_dv_leg1, best_dv_leg2])

            veh.visitedUID = new_uids
            veh.visitedSAT = new_sats
            veh.arrivals   = new_arr
            veh.departures = new_dep
            veh.costs      = new_costs
        end

        if i < length(demands_to_move)
            sort!(demands_to_move, by = x -> deadlines[x[1]])
        end
    end

    if isempty(unassigned_uids)
        unassigned = unassigned_in
    elseif unassigned_in === nothing
        unassigned = subset_unassigned(unassigned_uids, demands)
    else
        unassigned = subset_unassigned(vcat(unassigned_in["UIDs"], unassigned_uids), demands)
    end

    return schedule, unassigned
end






# ── Bulk vehicle removal + single regret-2 reinsertion pass ──────────────────
# Removes the k_remove smallest vehicles in one shot, collects all their demands
# as unassigned, then runs one regret-2 insertion pass to recover what it can.
function bulk_remove_vehicles(schedule, demands, CostTable, MinTOFTable, sim,
                               unassigned_in=nothing; k_remove=1)

    schedule = copy_schedule(schedule)
    length(schedule) <= 1 && return schedule, unassigned_in

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))

    k = min(k_remove, length(schedule) - 1)   # keep at least 1 vehicle

    # Sort vehicles by demand count ascending, remove the k smallest
    demand_counts = [count(uid -> uid > 0, veh.visitedUID) for veh in schedule]
    victim_idxs   = partialsortperm(demand_counts, 1:k)

    removed_uids = Int[]
    for vi in sort(victim_idxs, rev=true)
        for uid in schedule[vi].visitedUID
            uid > 0 && push!(removed_uids, uid)
        end
        deleteat!(schedule, vi)
    end

    # Merge with any already-unassigned demands
    if unassigned_in !== nothing && !isempty(get(unassigned_in, "UIDs", []))
        append!(removed_uids, unassigned_in["UIDs"])
    end

    # All freed demands go straight to unassigned — create_vehicle will
    # build new tours from them in subsequent iterations.
    unassigned = _merge_unassigned(unassigned_in, removed_uids, demands)
    return schedule, unassigned
end

function swap_cross_vehicle(schedule, demands, CostTable, MinTOFTable, sim;
                           k=1, top_n=20)

    schedule = copy_schedule(schedule)

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    svc_times   = demands["service_times"]
    deadlines   = demands["demand_deadlines"]

    # ── Phase 1: collect demand positions and per-position slack ───────────
    dem_positions = [Int[] for _ in schedule]
    slack_by_pos  = [Dict{Int,Float64}() for _ in schedule]

    for (v, veh) in enumerate(schedule)
        for p in 2:length(veh.visitedUID)-1
            veh.visitedUID[p] > 0 || continue
            push!(dem_positions[v], p)
            uid = veh.visitedUID[p]
            slack_by_pos[v][p] = deadlines[uid] - (veh.arrivals[p] + svc_times[uid])
        end
    end

    # ── Phase 2: generate candidate swap pairs via slack heuristic ────────
    candidates = Tuple{Int,Int,Int,Int,Float64}[]

    n_veh = length(schedule)
    for v1 in 1:n_veh
        for v2 in v1+1:n_veh
            for p1 in dem_positions[v1]
                sat1   = schedule[v1].visitedSAT[p1]
                prev1  = schedule[v1].visitedSAT[p1-1]
                next1  = schedule[v1].visitedSAT[p1+1]

                for p2 in dem_positions[v2]
                    sat2   = schedule[v2].visitedSAT[p2]
                    prev2  = schedule[v2].visitedSAT[p2-1]
                    next2  = schedule[v2].visitedSAT[p2+1]

                    # Quick TOF gate: both directions must be feasible
                    haskey(MinTOFTable, (name_to_idx[prev1], name_to_idx[sat2])) || continue
                    haskey(MinTOFTable, (name_to_idx[sat2],   name_to_idx[next1])) || continue
                    haskey(MinTOFTable, (name_to_idx[prev2], name_to_idx[sat1])) || continue
                    haskey(MinTOFTable, (name_to_idx[sat1],   name_to_idx[next2])) || continue

                    score = slack_by_pos[v1][p1] + slack_by_pos[v2][p2]
                    push!(candidates, (v1, p1, v2, p2, score))
                end
            end
        end
    end

    isempty(candidates) && return schedule

    sort!(candidates, by=x -> x[5], rev=true)
    top_n = min(top_n, length(candidates))

    best_v1 = best_p1 = best_v2 = best_p2 = 0
    best_delta = Inf

    # ── Phase 3: full feasibility + cost evaluation of top N ──────────────
    for ci in 1:top_n
        v1, p1, v2, p2 = candidates[ci][1:4]
        veh1 = schedule[v1]; veh2 = schedule[v2]
        n1 = length(veh1.visitedUID); n2 = length(veh2.visitedUID)

        uid1 = veh1.visitedUID[p1]; sat1 = veh1.visitedSAT[p1]
        uid2 = veh2.visitedUID[p2]; sat2 = veh2.visitedSAT[p2]

        prev1_idx = name_to_idx[veh1.visitedSAT[p1-1]]
        d2_idx    = name_to_idx[sat2]
        next1_idx = name_to_idx[veh1.visitedSAT[p1+1]]

        tof_prev_d2 = MinTOFTable[(prev1_idx, d2_idx)][1]
        tof_d2_next = MinTOFTable[(d2_idx, next1_idx)][1]

        arr_d2 = veh1.departures[p1-1] + tof_prev_d2
        dep_d2 = arr_d2 + svc_times[uid2]
        dep_d2 > deadlines[uid2] && continue

        new_arr_next1 = dep_d2 + tof_d2_next
        shift1 = new_arr_next1 - veh1.arrivals[p1+1]

        feasible1 = true
        for j in p1+2:n1
            uid_j = veh1.visitedUID[j]
            if uid_j > 0
                a = veh1.arrivals[j] + shift1
                a + svc_times[uid_j] > deadlines[uid_j] && (feasible1 = false; break)
            end
        end
        feasible1 || continue

        dv_prev_d2 = snap_cost(CostTable, prev1_idx, d2_idx, veh1.departures[p1-1], arr_d2)
        dv_d2_next = snap_cost(CostTable, d2_idx, next1_idx, dep_d2, new_arr_next1)

        prev2_idx = name_to_idx[veh2.visitedSAT[p2-1]]
        d1_idx    = name_to_idx[sat1]
        next2_idx = name_to_idx[veh2.visitedSAT[p2+1]]

        tof_prev_d1 = MinTOFTable[(prev2_idx, d1_idx)][1]
        tof_d1_next = MinTOFTable[(d1_idx, next2_idx)][1]

        arr_d1 = veh2.departures[p2-1] + tof_prev_d1
        dep_d1 = arr_d1 + svc_times[uid1]
        dep_d1 > deadlines[uid1] && continue

        new_arr_next2 = dep_d1 + tof_d1_next
        shift2 = new_arr_next2 - veh2.arrivals[p2+1]

        feasible2 = true
        for j in p2+2:n2
            uid_j = veh2.visitedUID[j]
            if uid_j > 0
                a = veh2.arrivals[j] + shift2
                a + svc_times[uid_j] > deadlines[uid_j] && (feasible2 = false; break)
            end
        end
        feasible2 || continue

        dv_prev_d1 = snap_cost(CostTable, prev2_idx, d1_idx, veh2.departures[p2-1], arr_d1)
        dv_d1_next = snap_cost(CostTable, d1_idx, next2_idx, dep_d1, new_arr_next2)

        old_cost = veh1.costs[p1] + veh1.costs[p1+1] + veh2.costs[p2] + veh2.costs[p2+1]
        new_cost = dv_prev_d2 + dv_d2_next + dv_prev_d1 + dv_d1_next
        delta = new_cost - old_cost

        if delta < best_delta - 1e-6
            best_delta = delta
            best_v1 = v1; best_p1 = p1
            best_v2 = v2; best_p2 = p2
        end
    end

    # ── Phase 4: apply best swap ──────────────────────────────────────
    if best_v1 > 0
        v1 = best_v1; p1 = best_p1
        v2 = best_v2; p2 = best_p2
        veh1 = schedule[v1]; veh2 = schedule[v2]
        n1 = length(veh1.visitedUID); n2 = length(veh2.visitedUID)

        uid1 = veh1.visitedUID[p1]; sat1 = veh1.visitedSAT[p1]
        uid2 = veh2.visitedUID[p2]; sat2 = veh2.visitedSAT[p2]

        # ── Mutate v1: swap in d2 at p1 ───────────────────────────────
        prev1_idx = name_to_idx[veh1.visitedSAT[p1-1]]
        d2_idx    = name_to_idx[sat2]
        next1_idx = name_to_idx[veh1.visitedSAT[p1+1]]

        tof_prev_d2 = MinTOFTable[(prev1_idx, d2_idx)][1]
        tof_d2_next = MinTOFTable[(d2_idx, next1_idx)][1]

        arr_d2  = veh1.departures[p1-1] + tof_prev_d2
        dep_d2  = arr_d2 + svc_times[uid2]
        n_arr_n1 = dep_d2 + tof_d2_next
        shift1  = n_arr_n1 - veh1.arrivals[p1+1]

        dv_prev_d2 = snap_cost(CostTable, prev1_idx, d2_idx, veh1.departures[p1-1], arr_d2)
        dv_d2_next = snap_cost(CostTable, d2_idx, next1_idx, dep_d2, n_arr_n1)

        veh1.visitedUID[p1]    = uid2
        veh1.visitedSAT[p1]    = sat2
        veh1.arrivals[p1]      = arr_d2
        veh1.departures[p1]    = dep_d2
        veh1.costs[p1]         = dv_prev_d2
        veh1.costs[p1+1]       = dv_d2_next

        for j in p1+1:n1
            veh1.arrivals[j]   += shift1
            veh1.departures[j] += shift1
        end
        for j in p1+2:n1
            fi = name_to_idx[veh1.visitedSAT[j-1]]
            ti = name_to_idx[veh1.visitedSAT[j]]
            veh1.costs[j] = snap_cost(CostTable, fi, ti, veh1.departures[j-1], veh1.arrivals[j])
        end

        # ── Mutate v2: swap in d1 at p2 ───────────────────────────────
        prev2_idx = name_to_idx[veh2.visitedSAT[p2-1]]
        d1_idx    = name_to_idx[sat1]
        next2_idx = name_to_idx[veh2.visitedSAT[p2+1]]

        tof_prev_d1 = MinTOFTable[(prev2_idx, d1_idx)][1]
        tof_d1_next = MinTOFTable[(d1_idx, next2_idx)][1]

        arr_d1  = veh2.departures[p2-1] + tof_prev_d1
        dep_d1  = arr_d1 + svc_times[uid1]
        n_arr_n2 = dep_d1 + tof_d1_next
        shift2  = n_arr_n2 - veh2.arrivals[p2+1]

        dv_prev_d1 = snap_cost(CostTable, prev2_idx, d1_idx, veh2.departures[p2-1], arr_d1)
        dv_d1_next = snap_cost(CostTable, d1_idx, next2_idx, dep_d1, n_arr_n2)

        veh2.visitedUID[p2]    = uid1
        veh2.visitedSAT[p2]    = sat1
        veh2.arrivals[p2]      = arr_d1
        veh2.departures[p2]    = dep_d1
        veh2.costs[p2]         = dv_prev_d1
        veh2.costs[p2+1]       = dv_d1_next

        for j in p2+1:n2
            veh2.arrivals[j]   += shift2
            veh2.departures[j] += shift2
        end
        for j in p2+2:n2
            fi = name_to_idx[veh2.visitedSAT[j-1]]
            ti = name_to_idx[veh2.visitedSAT[j]]
            veh2.costs[j] = snap_cost(CostTable, fi, ti, veh2.departures[j-1], veh2.arrivals[j])
        end
    end

    return schedule
end




function swap_intratour(schedule, demands, CostTable, MinTOFTable, sim)

    schedule = copy_schedule(schedule)

    # 2-opt on the highest-cost vehicle: reverse a segment to reduce total delta-V

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    svc_times   = demands["service_times"]
    deadlines   = demands["demand_deadlines"]

    total_costs = [sum(veh.costs) for veh in schedule]
    target_v    = argmax(total_costs)
    veh = schedule[target_v]
    n   = length(veh.visitedUID)

    best_delta  = Inf
    best_uids   = nothing
    best_sats   = nothing
    best_arrs   = nothing
    best_deps   = nothing
    best_costs  = nothing
    old_total   = sum(veh.costs)

    for i in 1:n-2
        for j in i+2:n-1
            new_uids = vcat(veh.visitedUID[1:i], reverse(veh.visitedUID[i+1:j]), veh.visitedUID[j+1:end])
            new_sats = vcat(veh.visitedSAT[1:i], reverse(veh.visitedSAT[i+1:j]), veh.visitedSAT[j+1:end])

            new_arrs  = Float64[veh.arrivals[1]]
            new_deps  = Float64[veh.departures[1]]
            new_costs = Float64[0.0]
            feasible  = true

            for p in 2:length(new_sats)
                from_idx = name_to_idx[new_sats[p-1]]
                to_idx   = name_to_idx[new_sats[p]]

                haskey(MinTOFTable, (from_idx, to_idx)) || (feasible = false; break)
                tof = MinTOFTable[(from_idx, to_idx)][1]

                arr = new_deps[p-1] + tof
                uid = new_uids[p]

                if uid > 0
                    dep = arr + svc_times[uid]
                    dep > deadlines[uid] && (feasible = false; break)
                else
                    dep = arr + 0.5
                end

                push!(new_arrs, arr)
                push!(new_deps, dep)
                dv = snap_cost(CostTable, from_idx, to_idx, new_deps[p-1], arr)
                push!(new_costs, dv)
            end
            feasible || continue

            delta = sum(new_costs) - old_total
            if delta < best_delta - 1e-6
                best_delta = delta
                best_uids  = new_uids
                best_sats  = new_sats
                best_arrs  = new_arrs
                best_deps  = new_deps
                best_costs = new_costs
            end
        end
    end

    if best_delta < -1e-6
        veh.visitedUID = best_uids
        veh.visitedSAT = best_sats
        veh.arrivals   = best_arrs
        veh.departures = best_deps
        veh.costs      = best_costs
    end

    return schedule
end




function create_vehicle(schedule, demands, unassigned, CostTable, MinTOFTable, sim;
                         dv_budget=5000.0, refuel_time=0.5, start_time=0.0)

    schedule = copy_schedule(schedule)

    unassigned === nothing && return schedule, unassigned
    isempty(unassigned["UIDs"]) && return schedule, unassigned

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    svc_times   = demands["service_times"]
    deadlines   = demands["demand_deadlines"]
    sat_ids     = demands["sat_identifiers"]
    available   = demand_ready_vec(demands)

    unrouted = Set{Int}(unassigned["UIDs"])
    sim_idx_for_uid = [name_to_idx[sat_ids[uid]] for uid in demands["UIDs"]]

    depot_idxs  = findall(n -> startswith(n, "depot"), sim.names)
    dep_idx     = depot_idxs[(length(schedule) % length(depot_idxs)) + 1]
    depot_name  = sim.names[dep_idx]

    all_depot  = [uid for veh in schedule for uid in veh.visitedUID if uid < 0]
    depot_uid  = isempty(all_depot) ? -1 : minimum(all_depot) - 1

    if !isempty(unassigned["UIDs"])
        start_time = max(start_time, minimum(available[uid] for uid in unassigned["UIDs"]))
    end

    visited_uids = Int[depot_uid]
    visited_sats = String[depot_name]
    arrivals     = Float64[start_time]
    departures   = Float64[start_time]
    costs        = Float64[0.0]

    current_sim_idx = dep_idx
    current_time    = start_time
    leg_dv          = 0.0

    while true
        best_uid, best_dv = get_best_next(current_sim_idx, current_time,
                                          unrouted, sat_ids, svc_times, deadlines,
                                          MinTOFTable, sim_idx_for_uid, available)
        best_uid === nothing && break

        cand_idx = sim_idx_for_uid[best_uid]
        tof, _   = MinTOFTable[(current_sim_idx, cand_idx)]
        svc_time = svc_times[best_uid]
        arr_time = max(current_time + tof, available[best_uid])

        if leg_dv + best_dv > dv_budget
            if current_sim_idx == dep_idx
                delete!(unrouted, best_uid)
                continue
            end
            depot_uid -= 1
            push!(visited_uids, depot_uid)
            push!(visited_sats, depot_name)
            tof_depot, dv_depot = get(MinTOFTable, (current_sim_idx, dep_idx), (0.0, 0.0))
            current_time += tof_depot
            push!(arrivals,   Float64(current_time))
            push!(costs,      dv_depot)
            current_time += refuel_time
            push!(departures, Float64(current_time))
            current_sim_idx = dep_idx
            leg_dv = 0.0
            continue
        end

        push!(visited_uids, best_uid)
        push!(visited_sats, sat_ids[best_uid])
        push!(arrivals,   Float64(arr_time))
        push!(departures, Float64(arr_time + svc_time))
        push!(costs,      best_dv)

        current_sim_idx = cand_idx
        current_time    = arr_time + svc_time
        leg_dv         += best_dv
        delete!(unrouted, best_uid)
    end

    # Return to depot
    depot_uid -= 1
    push!(visited_uids, depot_uid)
    push!(visited_sats, depot_name)
    tof_depot, dv_depot = get(MinTOFTable, (current_sim_idx, dep_idx), (0.0, 0.0))
    current_time += tof_depot
    push!(arrivals,   Float64(current_time))
    push!(costs,      dv_depot)
    current_time += refuel_time
    push!(departures, Float64(current_time))

    push!(schedule, vehicle(visited_uids, visited_sats, arrivals, departures, costs))


    if isempty(unrouted)
        return schedule, unassigned
    else
        remaining_uids = collect(unrouted)
        d = subset_unassigned(remaining_uids, demands)
        return schedule, d
    end
end











# ── Shared repair: regret-2 insertion ────────────────────────────────────────
# Inserts `removed_uids` into `schedule` one at a time, ordered by descending
# regret (best2_delta - best1_delta). Demands with no feasible position are
# appended to `out_unassigned`.
function _regret2_insert!(schedule, removed_uids, demands, CostTable, MinTOFTable,
                          name_to_idx, out_unassigned)
    svc_times = demands["service_times"]
    deadlines = demands["demand_deadlines"]
    sat_ids   = demands["sat_identifiers"]
    available = demand_ready_vec(demands)

    remaining = copy(removed_uids)

    while !isempty(remaining)
        best_regret = -Inf
        best_uid    = 0
        best_v      = 0
        best_p      = 0
        best_delta  = 0.0

        for uid in remaining
            r_idx = name_to_idx[sat_ids[uid]]
            dl    = deadlines[uid]
            svc   = svc_times[uid]
            b1 = Inf; b1v = 0; b1p = 0; b1delta = 0.0
            b2 = Inf

            for (v, veh) in enumerate(schedule)
                n = length(veh.visitedUID)
                for p in 1:n
                    prev_name = veh.visitedSAT[p]
                    prev_idx  = name_to_idx[prev_name]
                    haskey(MinTOFTable, (prev_idx, r_idx)) || continue
                    tof_pr = MinTOFTable[(prev_idx, r_idx)][1]
                    arr_r  = max(veh.departures[p] + tof_pr, available[uid])
                    arr_r + svc > dl && continue

                    next_idx = p < n ? name_to_idx[veh.visitedSAT[p+1]] : nothing
                    haskey(MinTOFTable, (r_idx, isnothing(next_idx) ? prev_idx : next_idx)) || continue

                    dep_r = arr_r + svc
                    if isnothing(next_idx)
                        dv_in  = snap_cost(CostTable, prev_idx, r_idx, veh.departures[p], arr_r)
                        dv_out = 0.0
                        old_dv = 0.0
                    else
                        arr_next = veh.arrivals[p+1]
                        dep_r > arr_next && continue   # waiting would delay the next stop
                        dv_in    = snap_cost(CostTable, prev_idx, r_idx, veh.departures[p], arr_r)
                        dv_out   = snap_cost(CostTable, r_idx, next_idx, dep_r, arr_next)
                        old_dv   = veh.costs[p+1]
                    end
                    delta = dv_in + dv_out - old_dv
                    if delta < b1
                        b2 = b1
                        b1 = delta; b1v = v; b1p = p; b1delta = delta
                    elseif delta < b2
                        b2 = delta
                    end
                end
            end

            regret = (b2 == Inf ? b1 : b2) - b1
            if b1v > 0 && regret > best_regret
                best_regret = regret
                best_uid    = uid
                best_v      = b1v
                best_p      = b1p
                best_delta  = b1delta
            end
        end

        if best_v == 0
            append!(out_unassigned, [uid for uid in remaining if uid == first(remaining)])
            filter!(u -> u != first(remaining), remaining)
            continue
        end

        filter!(u -> u != best_uid, remaining)
        veh     = schedule[best_v]
        r_idx   = name_to_idx[sat_ids[best_uid]]
        prev_idx = name_to_idx[veh.visitedSAT[best_p]]
        tof_pr  = MinTOFTable[(prev_idx, r_idx)][1]
        arr_r   = max(veh.departures[best_p] + tof_pr, available[best_uid])
        dep_r   = arr_r + svc_times[best_uid]
        dv_in   = snap_cost(CostTable, prev_idx, r_idx, veh.departures[best_p], arr_r)

        insert!(veh.visitedUID, best_p + 1, best_uid)
        insert!(veh.visitedSAT, best_p + 1, sat_ids[best_uid])
        insert!(veh.arrivals,   best_p + 1, arr_r)
        insert!(veh.departures, best_p + 1, dep_r)
        insert!(veh.costs,      best_p + 1, dv_in)

        if best_p + 1 < length(veh.visitedUID)
            next_idx = name_to_idx[veh.visitedSAT[best_p + 2]]
            arr_next = veh.arrivals[best_p + 2]
            veh.costs[best_p + 2] = snap_cost(CostTable, r_idx, next_idx, dep_r, arr_next)
        end
    end
end


# ── Destroy-and-repair operator ───────────────────────────────────────────────
function destroy_and_repair(schedule, demands, CostTable, MinTOFTable, min_dv_tab, sim,
                             unassigned_in=nothing; destroy_frac=nothing)

    schedule = copy_schedule(schedule)
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids   = demands["sat_identifiers"]
    deadlines = demands["demand_deadlines"]

    # Randomise destruction intensity each call: uniform in [0.25, 0.50]
    frac = isnothing(destroy_frac) ? (0.25 + rand() * 0.25) : destroy_frac

    # ── Worst-cost removal ────────────────────────────────────────────────────
    scored = Tuple{Float64, Int, Int}[]
    for (v, veh) in enumerate(schedule)
        n = length(veh.visitedUID)
        for p in 2:n-1
            veh.visitedUID[p] > 0 || continue
            prev_idx = name_to_idx[veh.visitedSAT[p-1]]
            node_idx = name_to_idx[veh.visitedSAT[p]]
            next_idx = name_to_idx[veh.visitedSAT[p+1]]
            saving = min_dv_tab[prev_idx, node_idx] + min_dv_tab[node_idx, next_idx] -
                     min_dv_tab[prev_idx, next_idx]
            push!(scored, (saving, v, p))
        end
    end
    sort!(scored, by = x -> x[1], rev = true)

    total_demands = count(x -> x > 0, (uid for veh in schedule for uid in veh.visitedUID))
    k = max(1, round(Int, total_demands * frac))
    k = min(k, length(scored))

    # Sample from top-2k candidates (not always the exact worst k) for diversity
    pool = scored[1:min(2k, length(scored))]
    chosen = sort(randperm(length(pool))[1:k])   # random k from pool, sorted desc for safe deletion

    removed_uids = Int[]
    to_del = Dict{Int, Vector{Int}}()
    for ci in chosen
        _, v, p = pool[ci]
        push!(removed_uids, schedule[v].visitedUID[p])
        push!(get!(to_del, v, Int[]), p)
    end
    for (v, positions) in to_del
        for pos in sort(positions, rev=true)
            deleteat!(schedule[v].visitedUID, pos)
            deleteat!(schedule[v].visitedSAT, pos)
            deleteat!(schedule[v].arrivals,   pos)
            deleteat!(schedule[v].departures, pos)
            deleteat!(schedule[v].costs,      pos)
        end
    end

    # ── Regret-2 insertion ────────────────────────────────────────────────────
    out_unassigned = Int[]
    _regret2_insert!(schedule, removed_uids, demands, CostTable, MinTOFTable,
                     name_to_idx, out_unassigned)

    unassigned = _merge_unassigned(unassigned_in, out_unassigned, demands)
    return schedule, unassigned
end


# ── Shaw orbital-relatedness operator ─────────────────────────────────────────
function shaw_removal_repair(schedule, demands, CostTable, MinTOFTable, min_dv_tab, sim,
                              unassigned_in=nothing;
                              destroy_frac=nothing, shaw_α=1.0, shaw_β=0.3, shaw_p=3.0)

    schedule = copy_schedule(schedule)
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids   = demands["sat_identifiers"]
    deadlines = demands["demand_deadlines"]

    # ── Shaw removal ──────────────────────────────────────────────────────────
    all_positions = [(v, p) for (v, veh) in enumerate(schedule)
                             for p in eachindex(veh.visitedUID)
                             if veh.visitedUID[p] > 0]
    isempty(all_positions) && return schedule, unassigned_in

    # Randomise destruction intensity each call: uniform in [0.25, 0.50]
    frac = isnothing(destroy_frac) ? (0.25 + rand() * 0.25) : destroy_frac
    total_demands = length(all_positions)
    k = max(1, round(Int, total_demands * frac))

    seed_v, seed_p = rand(all_positions)
    seed_name = schedule[seed_v].visitedSAT[seed_p]
    seed_idx  = name_to_idx[seed_name]
    seed_uid  = schedule[seed_v].visitedUID[seed_p]
    seed_dl   = deadlines[seed_uid]

    removed_uids = Int[]
    for _ in 1:k
        cands = Tuple{Float64, Int, Int}[]
        for (v, veh) in enumerate(schedule)
            for p in eachindex(veh.visitedUID)
                veh.visitedUID[p] > 0 || continue
                (v == seed_v && p == seed_p) && continue
                c_idx = name_to_idx[veh.visitedSAT[p]]
                c_dl  = deadlines[veh.visitedUID[p]]
                rel   = shaw_α * min_dv_tab[seed_idx, c_idx] +
                        shaw_β * abs(seed_dl - c_dl)
                push!(cands, (rel, v, p))
            end
        end
        isempty(cands) && break
        sort!(cands, by = x -> x[1])
        idx = clamp(floor(Int, rand()^shaw_p * length(cands)) + 1, 1, length(cands))
        _, pv, pp = cands[idx]
        push!(removed_uids, schedule[pv].visitedUID[pp])
        deleteat!(schedule[pv].visitedUID, pp)
        deleteat!(schedule[pv].visitedSAT, pp)
        deleteat!(schedule[pv].arrivals,   pp)
        deleteat!(schedule[pv].departures, pp)
        deleteat!(schedule[pv].costs,      pp)
        seed_idx = name_to_idx[sat_ids[removed_uids[end]]]
        seed_dl  = deadlines[removed_uids[end]]
    end

    # ── Regret-2 insertion ────────────────────────────────────────────────────
    out_unassigned = Int[]
    _regret2_insert!(schedule, removed_uids, demands, CostTable, MinTOFTable,
                     name_to_idx, out_unassigned)

    unassigned = _merge_unassigned(unassigned_in, out_unassigned, demands)
    return schedule, unassigned
end


# ── Helper: merge newly unassigned UIDs into the unassigned dict ──────────────
function _merge_unassigned(unassigned_in, new_uids, demands)
    isempty(new_uids) && return unassigned_in
    if unassigned_in === nothing
        return subset_unassigned(new_uids, demands)
    else
        return subset_unassigned(vcat(unassigned_in["UIDs"], new_uids), demands)
    end
end


function remove_from_archive!(A, idx)
    deleteat!(A.solutions, idx)
    deleteat!(A.unassigned_sets, idx)
    deleteat!(A.total_deltaV, idx)
    deleteat!(A.total_serv_time_unassigned, idx)
    deleteat!(A.total_vehicles_used, idx)
    # Tree is not updated here — caller must rebuild after batch removals
end


function dominates(A, ai, B, bi)
    A.total_deltaV[ai] <= B.total_deltaV[bi] &&
    A.total_serv_time_unassigned[ai] <= B.total_serv_time_unassigned[bi] &&
    A.total_vehicles_used[ai] <= B.total_vehicles_used[bi] &&
    (A.total_deltaV[ai] < B.total_deltaV[bi] ||
     A.total_serv_time_unassigned[ai] < B.total_serv_time_unassigned[bi] ||
     A.total_vehicles_used[ai] < B.total_vehicles_used[bi])
end

function merge_archive!(F, G)
    isempty(G.solutions) && return
    isempty(F.solutions) && begin
        append!(F.solutions, G.solutions)
        append!(F.unassigned_sets, G.unassigned_sets)
        append!(F.total_deltaV, G.total_deltaV)
        append!(F.total_serv_time_unassigned, G.total_serv_time_unassigned)
        append!(F.total_vehicles_used, G.total_vehicles_used)
        rebuild_octree!(F)
        empty!(G.solutions)
        empty!(G.unassigned_sets)
        empty!(G.total_deltaV)
        empty!(G.total_serv_time_unassigned)
        empty!(G.total_vehicles_used)
        G.tree = nothing
        return
    end

    # ── Filter G: drop anything F already dominates (octree, k log N) ──
    g_survive = trues(length(G.solutions))
    for j in eachindex(G.solutions)
        if F.tree !== nothing && !isempty(F.solutions)
            if is_dominated(F.tree,
                            G.total_deltaV[j],
                            G.total_serv_time_unassigned[j],
                            G.total_vehicles_used[j], F)
                g_survive[j] = false
            end
        else
            for i in eachindex(F.solutions)
                if dominates(F, i, G, j)
                    g_survive[j] = false
                    break
                end
            end
        end
    end

    # ── Remove from F anything dominated by surviving G ──
    surviving_g = findall(g_survive)
    f_dominated = zeros(Bool, length(F.solutions))
    for i in eachindex(F.solutions)
        for j in surviving_g
            if dominates(G, j, F, i)
                f_dominated[i] = true
                break
            end
        end
    end

    # ── Phase B: batch execute ──
    for idx in findall(f_dominated) |> sort |> reverse
        remove_from_archive!(F, idx)
    end

    append!(F.solutions, G.solutions[surviving_g])
    append!(F.unassigned_sets, G.unassigned_sets[surviving_g])
    append!(F.total_deltaV, G.total_deltaV[surviving_g])
    append!(F.total_serv_time_unassigned, G.total_serv_time_unassigned[surviving_g])
    append!(F.total_vehicles_used, G.total_vehicles_used[surviving_g])

    rebuild_octree!(F)

    # Clear G (already consumed)
    empty!(G.solutions)
    empty!(G.unassigned_sets)
    empty!(G.total_deltaV)
    empty!(G.total_serv_time_unassigned)
    empty!(G.total_vehicles_used)
    G.tree = nothing
end



# ── RAAN-walk resequencing ────────────────────────────────────────────────────
# Selects the highest-ΔV vehicle and reorders its demands by greedy nearest-RAAN
# neighbour at the time of each transfer.  Low RAAN delta → low plane-change ΔV.
function raan_walk_resequence(schedule, demands, CostTable, MinTOFTable, sim)
    isempty(schedule) && return schedule
    schedule = copy_schedule(schedule)
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids   = demands["sat_identifiers"]
    deadlines = demands["demand_deadlines"]
    svc_times = demands["service_times"]

    # J2 RAAN drift rate [rad/day] for satellite index i
    function raan_drift(idx)
        a   = sim.orbital_elements[1, idx]   # semi-major axis [km]
        inc = sim.orbital_elements[2, idx]   # inclination [rad]
        n   = sqrt(MU_MDLS / a^3) * 86400.0 # mean motion [rad/day]
        return -1.5 * n * J2_MDLS * (RE_MDLS / a)^2 * cos(inc)
    end

    # RAAN at epoch t [days] for satellite index i
    function raan_at(idx, t)
        Ω0 = sim.orbital_elements[3, idx]
        return Ω0 + raan_drift(idx) * t
    end

    # angular difference in [-π, π]
    ang_diff(a, b) = mod(a - b + π, 2π) - π

    # Select vehicle with highest total ΔV cost (excluding first depot entry)
    best_v = argmax([sum(veh.costs) for veh in schedule])
    veh    = schedule[best_v]

    # Collect real demand positions (uid > 0)
    demand_positions = [p for p in eachindex(veh.visitedUID) if veh.visitedUID[p] > 0]
    length(demand_positions) < 2 && return schedule

    demand_uids = [veh.visitedUID[p] for p in demand_positions]

    # Greedy nearest-RAAN-neighbour resequencing
    # Start from the depot (position 1)
    depot_name = veh.visitedSAT[1]
    depot_idx  = name_to_idx[depot_name]
    depot_dep  = veh.departures[1]

    remaining  = collect(demand_uids)
    new_order  = Int[]
    prev_idx   = depot_idx
    prev_dep   = depot_dep

    while !isempty(remaining)
        best_uid   = 0
        best_dRaan = Inf
        best_arr   = 0.0

        for uid in remaining
            r_idx    = name_to_idx[sat_ids[uid]]
            tof_info = get(MinTOFTable, (prev_idx, r_idx), nothing)
            tof_info === nothing && continue
            arr      = prev_dep + tof_info[1]
            arr + svc_times[uid] > deadlines[uid] && continue

            dΩ = abs(ang_diff(raan_at(r_idx, arr), raan_at(prev_idx, prev_dep)))
            if dΩ < best_dRaan
                best_dRaan = dΩ
                best_uid   = uid
                best_arr   = arr
            end
        end

        best_uid == 0 && break   # no feasible next demand
        push!(new_order, best_uid)
        filter!(u -> u != best_uid, remaining)
        prev_idx = name_to_idx[sat_ids[best_uid]]
        prev_dep = best_arr + svc_times[best_uid]
    end

    length(new_order) != length(demand_uids) && return schedule  # couldn't resequence all

    # Rebuild vehicle with new order
    new_uids  = Int[veh.visitedUID[1]]        # depot
    new_sats  = String[veh.visitedSAT[1]]
    new_arrs  = Float64[veh.arrivals[1]]
    new_deps  = Float64[veh.departures[1]]
    new_costs = Float64[veh.costs[1]]

    p_prev = 1
    for uid in new_order
        r_idx    = name_to_idx[sat_ids[uid]]
        prev_idx2 = name_to_idx[new_sats[end]]
        tof_info = get(MinTOFTable, (prev_idx2, r_idx), nothing)
        tof_info === nothing && return schedule
        arr = new_deps[end] + tof_info[1]
        arr + svc_times[uid] > deadlines[uid] && return schedule
        dep = arr + svc_times[uid]
        dv  = snap_cost(CostTable, prev_idx2, r_idx, new_deps[end], arr)
        push!(new_uids, uid);     push!(new_sats, sat_ids[uid])
        push!(new_arrs, arr);     push!(new_deps, dep)
        push!(new_costs, dv)
    end

    # Also carry over any depot-return entries from original vehicle
    for p in eachindex(veh.visitedUID)
        veh.visitedUID[p] < 0 && p > 1 && (
            push!(new_uids, veh.visitedUID[p]); push!(new_sats, veh.visitedSAT[p]);
            push!(new_arrs, veh.arrivals[p]);   push!(new_deps, veh.departures[p]);
            push!(new_costs, veh.costs[p]))
    end

    new_dv  = sum(new_costs)
    old_dv  = sum(veh.costs)
    new_dv >= old_dv && return schedule   # no improvement

    schedule[best_v] = vehicle(new_uids, new_sats, new_arrs, new_deps, new_costs)
    return schedule
end

# ── RAAN-phasing timing search ────────────────────────────────────────────────
# For the top-k highest-cost legs, computes when the two satellites' RAANs will
# next be aligned (using J2 drift), then searches the cost table near that epoch
# for a cheaper departure — a global timing search vs opt_times' local ±1 cell.
function raan_phasing_timing(schedule, demands, CostTable, MinTOFTable, sim;
                              k::Int=3, ag=nothing)
    isempty(schedule) && return schedule
    schedule    = copy_schedule(schedule)
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    ct          = ag !== nothing ? ag : CostTable

    # J2 drift rate [rad/day]
    function raan_drift(idx)
        a   = sim.orbital_elements[1, idx]
        inc = sim.orbital_elements[2, idx]
        n   = sqrt(MU_MDLS / a^3) * 86400.0
        return -1.5 * n * J2_MDLS * (RE_MDLS / a)^2 * cos(inc)
    end

    # Find the epoch (relative to t0, in days) when ΔΩ = 0 mod 2π nearest to t_now
    function raan_alignment_epoch(idx_f, idx_t, t_now)
        Ω_f  = sim.orbital_elements[3, idx_f] + raan_drift(idx_f) * t_now
        Ω_t  = sim.orbital_elements[3, idx_t] + raan_drift(idx_t) * t_now
        dω   = raan_drift(idx_t) - raan_drift(idx_f)
        abs(dω) < 1e-12 && return t_now  # same drift rate → always aligned (or never)
        dΩ0  = mod(Ω_t - Ω_f, 2π)
        # time to alignment: dΩ0 + dω*(t - t_now) = 0  →  t = t_now - dΩ0/dω
        dt   = -dΩ0 / dω
        # get nearest future alignment
        period = abs(2π / dω)
        while dt < 0; dt += period; end
        return t_now + dt
    end

    # Collect all legs with their costs
    scored = Tuple{Float64, Int, Int}[]   # (cost, veh_idx, pos)
    for (vi, veh) in enumerate(schedule)
        for p in 2:length(veh.visitedUID)
            veh.visitedUID[p] > 0 || continue  # skip depot entries
            push!(scored, (veh.costs[p], vi, p))
        end
    end
    sort!(scored, by=x->x[1], rev=true)
    top = scored[1:min(k, length(scored))]

    # Try to shift each leg to its RAAN alignment epoch
    for (_, vi, p) in top
        veh      = schedule[vi]
        from_idx = name_to_idx[veh.visitedSAT[p-1]]
        to_idx   = name_to_idx[veh.visitedSAT[p]]
        uid      = veh.visitedUID[p]
        uid > 0 || continue

        uid_pos  = findfirst(==(uid), demands["UIDs"])
        uid_pos === nothing && continue
        deadline = demands["demand_deadlines"][uid_pos]
        svc_time = demands["service_times"][uid_pos]

        t_align = raan_alignment_epoch(from_idx, to_idx, veh.departures[p-1])
        tof_info = get(MinTOFTable, (from_idx, to_idx), nothing)
        tof_info === nothing && continue
        min_tof  = tof_info[1]

        # Search ±30 days around alignment epoch in cost table (snap to 15-day grid)
        best_dep  = veh.departures[p-1]
        best_cost = veh.costs[p]

        for dep_shift in -30.0:15.0:30.0
            dep_try = t_align + dep_shift
            dep_try < veh.departures[p-1] && continue   # can't go back in time
            arr_try = dep_try + min_tof
            arr_try + svc_time > deadline && continue
            c = snap_cost(ct, from_idx, to_idx, dep_try, arr_try)
            isfinite(c) && c < best_cost && (best_dep = dep_try; best_cost = c)
        end

        best_dep == veh.departures[p-1] && continue  # no improvement found

        Δ = best_dep - veh.departures[p-1]
        Δ <= 0 && continue

        # Minimum departure from p-1: service must be complete (dwell is free)
        uid_prev = veh.visitedUID[p-1]
        svc_prev = uid_prev < 0 ? REFUEL_TIME :
                   demands["service_times"][findfirst(==(uid_prev), demands["UIDs"])]
        best_dep < veh.arrivals[p-1] + svc_prev && continue

        arr_new = best_dep + min_tof
        arr_new + svc_time > deadline && continue

        # Check all downstream deadline feasibility
        feasible_phase = true
        for j in p+1:length(veh.visitedUID)
            uid_j = veh.visitedUID[j]
            if uid_j > 0
                pos_j = findfirst(==(uid_j), demands["UIDs"])
                if pos_j !== nothing &&
                   veh.arrivals[j] + Δ + demands["service_times"][pos_j] > demands["demand_deadlines"][pos_j]
                    feasible_phase = false; break
                end
            end
        end
        feasible_phase || continue

        # Apply: set dwell at p-1, propagate Δ to p and all downstream
        old_total = sum(veh.costs)
        veh.departures[p-1] = best_dep
        veh.arrivals[p]     = arr_new
        veh.departures[p]   = arr_new + svc_time
        veh.costs[p]        = snap_cost(ct, from_idx, to_idx, best_dep, arr_new)
        for j in p+1:length(veh.visitedUID)
            veh.arrivals[j]   += Δ
            veh.departures[j] += Δ
        end
        for j in p+1:length(veh.visitedUID)
            fi = name_to_idx[veh.visitedSAT[j-1]]
            ti = name_to_idx[veh.visitedSAT[j]]
            veh.costs[j] = snap_cost(ct, fi, ti, veh.departures[j-1], veh.arrivals[j])
        end
        # Revert if no net improvement
        if sum(veh.costs) >= old_total
            veh.departures[p-1] = best_dep - Δ
            veh.arrivals[p]     = arr_new - Δ
            veh.departures[p]   = (arr_new - Δ) + svc_time
            veh.costs[p]        = snap_cost(ct, from_idx, to_idx, veh.departures[p-1], veh.arrivals[p])
            for j in p+1:length(veh.visitedUID)
                veh.arrivals[j]   -= Δ
                veh.departures[j] -= Δ
            end
            for j in p+1:length(veh.visitedUID)
                fi = name_to_idx[veh.visitedSAT[j-1]]
                ti = name_to_idx[veh.visitedSAT[j]]
                veh.costs[j] = snap_cost(ct, fi, ti, veh.departures[j-1], veh.arrivals[j])
            end
        end
    end

    return schedule
end

function MDLS(maxiter, demands, simulation, cost_table, mintof_table, min_dv_tab;
              nvehicles=20, time_limit=Inf, init_sol=nothing, init_unassigned=nothing,
              dv_budget=5000.0, opt_times_shift=15.0, opt_times_top_pct=0.5)

    t_start = time()
    if isnothing(init_sol)
        init_sol, init_unassigned = make_init_schedule(demands, simulation, nvehicles=nvehicles)
    end
    sim_obj = load_sim()

    dv0 = isempty(init_sol) ? 0.0 : sum(sum(veh.costs) for veh in init_sol)
    us0 = init_unassigned === nothing ? 0.0 :
          sum(get(init_unassigned, "asset_values", init_unassigned["service_times"]))
    F = Archive([init_sol], [init_unassigned],
                [dv0], [us0], [length(init_sol)],
                _init_octree(dv0, us0, length(init_sol), 1))

    # ── ΔV operator pool (3 operators, one sampled randomly each iteration) ────
    # LNS operators (destroy_and_repair, shaw_removal_repair) excluded for fair comparison
    # with population-based GAs — uncomment to restore full MDLS+LNS:
    # dv_op_names = ["opt_times", "destroy_repair", "shaw"]
    # dv_operators = [
    #     (s, u) -> (opt_times_combined(s, demands, cost_table, mintof_table, sim_obj), u),
    #     (s, u) -> destroy_and_repair(s, demands, cost_table, mintof_table, min_dv_tab, sim_obj, u),
    #     (s, u) -> shaw_removal_repair(s, demands, cost_table, mintof_table, min_dv_tab, sim_obj, u),
    # ]
    dv_op_names = ["opt_times"]
    dv_operators = [
        (s, u) -> (opt_times_combined(s, demands, cost_table, mintof_table, sim_obj; shift=opt_times_shift, top_pct=opt_times_top_pct), u),
    ]

    op_times     = zeros(3)   # [consolidate, create_veh, dv_op]
    dv_op_counts = zeros(Int, length(dv_operators))

    for iter in 1:maxiter
        time() - t_start > time_limit && break
        idx = rand(1:length(F.solutions))
        x   = copy_schedule(F.solutions[idx])
        u_x = F.unassigned_sets[idx]

        G = Archive(Vector{Vector{vehicle}}(), Vector{Union{Nothing,Dict}}(), Float64[], Float64[], Int[], nothing)

        # ── Vehicles objective: bulk remove then single regret-2 reinsert ───────
        n_veh     = length(x)
        k_remove  = max(1, round(Int, rand() * 0.7 * n_veh + 0.2 * n_veh))  # Uniform[20%,90%] of vehicles
        cons_task = Threads.@spawn bulk_remove_vehicles(x, demands, cost_table, mintof_table,
                                                        sim_obj, u_x; k_remove=k_remove)

        # ── Unserved objective: create_vehicle only if unassigned demands exist
        has_unassigned = u_x !== nothing && !isempty(get(u_x, "UIDs", []))
        create_task = if has_unassigned
            k_add = max(1, min(
                round(Int, rand() * 0.9 * n_veh + 0.1 * n_veh),  # Uniform[10%,100%] of vehicles
                nvehicles - n_veh                                   # respect max vehicle limit
            ))
            Threads.@spawn begin
                s, u = x, u_x
                for _ in 1:k_add
                    s, u = create_vehicle(s, demands, u, cost_table, mintof_table, sim_obj; dv_budget=dv_budget)
                end
                (s, u)
            end
        else
            nothing
        end

        # ── ΔV objective: randomly sampled operator ───────────────────────────
        dv_idx  = rand(1:length(dv_operators))
        dv_op   = dv_operators[dv_idx]
        dv_op_counts[dv_idx] += 1
        dv_task = Threads.@spawn dv_op(x, u_x)

        # ── Collect results into G ────────────────────────────────────────────
        for (k, t) in enumerate([cons_task, create_task, dv_task])
            t === nothing && continue
            op_times[k] += @elapsed begin
                new_sol, new_u = fetch(t)
                dv = isempty(new_sol) ? 0.0 : sum(sum(veh.costs) for veh in new_sol)
                us = new_u === nothing ? 0.0 :
                     sum(get(new_u, "asset_values", new_u["service_times"]))
                push!(G.solutions, new_sol)
                push!(G.unassigned_sets, new_u)
                push!(G.total_deltaV, dv)
                push!(G.total_serv_time_unassigned, us)
                push!(G.total_vehicles_used, length(new_sol))
            end
        end

        merge_archive!(F, G)
        isempty(F.solutions) && break
    end

    @info "MDLS operator timing:" *
          "\n  consolidate = $(round(op_times[1]; digits=2))s" *
          "\n  create_veh  = $(round(op_times[2]; digits=2))s" *
          "\n  dv_ops      = $(round(op_times[3]; digits=2))s"
    @info "ΔV operator usage:" *
          join(["\n  $(dv_op_names[k]) = $(dv_op_counts[k])" for k in eachindex(dv_op_names)])

    # Final dominance cleanup using the octree (O(N log N))
    rebuild_octree!(F)
    dominated = zeros(Bool, length(F.solutions))
    for i in eachindex(F.solutions)
        if F.tree !== nothing
            dominated[i] = is_dominated(F.tree,
                                        F.total_deltaV[i],
                                        F.total_serv_time_unassigned[i],
                                        F.total_vehicles_used[i],
                                        F, i)
        end
    end
    for idx in findall(dominated) |> sort |> reverse
        remove_from_archive!(F, idx)
    end
    rebuild_octree!(F)

    return F
end












if !@isdefined(GATESTS_INCLUDE) || !GATESTS_INCLUDE
    @info "Running MDLS …"
    archive = MDLS(1000, demands, load_sim(), cost_table, mintof_table, min_dv_tab)
    @info "MDLS complete" n_solutions=length(archive.solutions)

    @save joinpath(@__DIR__, "outputs", "mdls_archive.jld2") archive

    fig = plot_pareto(archive)
    save(joinpath(@__DIR__, "outputs", "pareto_front.png"), fig)
end










