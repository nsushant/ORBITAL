include("sim/propagator.jl")
include("demands/generate_demands.jl")
include("sol_utils.jl")
include("plots/gantt.jl")
include("plots/visualise.jl")

demands = load_demands()
cost_table = load_cost_table()
mintof_table = build_min_tof_table()
schedule, unassigned = make_init_schedule(demands, load_sim(),nvehicles=40)

schedule_copy = copy(schedule)


if unassigned !== nothing
    @warn "Unassigned demands" n_unassigned=length(unassigned["UIDs"])
else
    @info "All demands assigned to vehicles"
end

for (v, veh) in enumerate(schedule)
    for i in 2:length(veh.visitedUID)
        gap = veh.arrivals[i] - veh.departures[i-1]
        @info "V$v step $i: $(veh.visitedSAT[i-1]) → $(veh.visitedSAT[i]), gap=$gap days"
    end
end

for (v, veh) in enumerate(schedule)
    for i in eachindex(veh.visitedUID)
        dt = veh.departures[i] - veh.arrivals[i]
        if dt > 10
            @warn "Large block" v i sat=veh.visitedSAT[i] uid=veh.visitedUID[i] arr=veh.arrivals[i] dep=veh.departures[i] dt
        end
    end
end

plot_schedule_gantt(schedule)



# operators to reduce the delta V of a schedule 

function opt_times(schedule, demands, CostTable, MinTOFTable, sim; top_pct=0.20)

    schedule = copy_schedule(schedule)

    # shift arrival and departure times independently.  


    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids   = demands["sat_identifiers"]
    svc_times = demands["service_times"]
    deadlines = demands["demand_deadlines"]

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
        base_shift = 15.0

        best_cost = legs[n][1]
        applied_dir = nothing

        # ── Later: shift arrivals[i:end] right ──
        new_arrs = veh.arrivals[i:end] .+ base_shift
        new_deps = veh.departures[i:end]

        feasible = true
        for j in eachindex(new_arrs)
            uid = veh.visitedUID[i + j - 1]
            new_arrs[j] > new_deps[j] && (feasible = false; break)
            uid > 0 && new_arrs[j] + svc_times[uid] > deadlines[uid] && (feasible = false; break)
        end
        new_arrs[1] - veh.departures[i-1] < min_tof && (feasible = false)

        if feasible
            if length(new_arrs) >= 2
                new_costs = Float64[snap_cost(CostTable,
                    name_to_idx[veh.visitedSAT[j-1]],
                    name_to_idx[veh.visitedSAT[j]],
                    new_deps[j-1], new_arrs[j])
                    for j in 2:length(new_arrs)]
                total_new = sum(new_costs)
            else
                total_new = snap_cost(CostTable, from_idx, to_idx, veh.departures[i-1], new_arrs[1])
                new_costs = [total_new]
            end
            if total_new < best_cost
                best_cost = total_new
                applied_dir = :later
            end
        end

        # ── Earlier: shift departures[1:i-1] left ──
        new_deps_up = veh.departures[1:i-1] .- base_shift
        new_arrs_up = veh.arrivals[1:i-1]

        feasible = true
        for j in 1:i-1
            uid = veh.visitedUID[j]
            veh.arrivals[j] > new_deps_up[j] && (feasible = false; break)
        end
        new_arrs_up[1] < 0.0 && (feasible = false)
        for j in 2:i-1
            tof_check = new_arrs_up[j] - new_deps_up[j-1]
            min_tof_j = get(MinTOFTable,
                (name_to_idx[veh.visitedSAT[j-1]], name_to_idx[veh.visitedSAT[j]]),
                (0.0, 0.0))[1]
            tof_check < min_tof_j && (feasible = false; break)
        end

        if feasible
            new_tof = veh.arrivals[i] - new_deps_up[end]
            new_tof >= min_tof || (feasible = false)
        end

        if feasible
            new_costs_up = Float64[snap_cost(CostTable,
                name_to_idx[veh.visitedSAT[j-1]],
                name_to_idx[veh.visitedSAT[j]],
                new_deps_up[j-1], new_arrs_up[j])
                for j in 2:i-1]
            dv_target = snap_cost(CostTable, from_idx, to_idx, new_deps_up[end], veh.arrivals[i])
            total_new = sum(new_costs_up) + dv_target
            if total_new < best_cost
                best_cost = total_new
                applied_dir = :earlier
            end
        end

        applied_dir === nothing && continue
        if applied_dir == :later
            veh.arrivals[i:end]   = new_arrs
            veh.departures[i:end] = new_deps
            veh.costs[i:end]      = new_costs
        else
            veh.departures[1:i-1] = new_deps_up
            veh.costs[2:i]        = vcat(new_costs_up, [dv_target])
        end
    end
    return schedule
end

function opt_times2(schedule, demands, CostTable, MinTOFTable, sim; top_pct=0.20)

    schedule = copy_schedule(schedule)

    #shifts arrival and departure times simultaneously

    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    sat_ids   = demands["sat_identifiers"]
    svc_times = demands["service_times"]
    deadlines = demands["demand_deadlines"]

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
        base_shift = 15.0

        best_cost = legs[n][1]
        applied_dir = nothing

        # ── Later: shift block i:end right ──
        new_arrs = veh.arrivals[i:end] .+ base_shift
        new_deps = veh.departures[i:end] .+ base_shift

        feasible = true
        for j in eachindex(new_arrs)
            uid = veh.visitedUID[i + j - 1]
            uid > 0 && new_arrs[j] + svc_times[uid] > deadlines[uid] && (feasible = false; break)
        end
        new_arrs[1] - veh.departures[i-1] < min_tof && (feasible = false)

        if feasible
            new_dv = snap_cost(CostTable, from_idx, to_idx, veh.departures[i-1], new_arrs[1])
            if new_dv < best_cost
                best_cost = new_dv
                applied_dir = :later
            end
        end

        # ── Earlier: shift block 1:i-1 left ──
        new_arrs_up = veh.arrivals[1:i-1] .- base_shift
        new_deps_up = veh.departures[1:i-1] .- base_shift

        feasible = true
        for j in 1:i-1
            new_arrs_up[j] < 0.0 && (feasible = false; break)
        end

        if feasible
            new_dv = snap_cost(CostTable, from_idx, to_idx, new_deps_up[end], veh.arrivals[i])
            if new_dv < best_cost
                best_cost = new_dv
                applied_dir = :earlier
            end
        end

        applied_dir === nothing && continue
        if applied_dir == :later
            veh.arrivals[i:end]   = new_arrs
            veh.departures[i:end] = new_deps
            veh.costs[i]          = best_cost
        else
            veh.arrivals[1:i-1]   = new_arrs_up
            veh.departures[1:i-1] = new_deps_up
            veh.costs[i]          = best_cost
        end
    end
    return schedule
end





function polish(schedule, demands, CostTable, MinTOFTable, sim; n_iters=5)
    for _ in 1:n_iters
        schedule = opt_times(schedule, demands, CostTable, MinTOFTable, sim)
        schedule = swap_intratour(schedule, demands, CostTable, MinTOFTable, sim)
        schedule = swap_cross_vehicle(schedule, demands, CostTable, MinTOFTable, sim)
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

                arr_cand = veh.departures[p] + tof_prev_cand
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

                arr_cand2 = dep_depot + tof_depot_cand
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

                arr_cand3 = veh.departures[n] + tof_depot_cand
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
        unassigned = Dict{String, Any}(
            "sat_identifiers" => [sat_ids[u] for u in unassigned_uids],
            "demand_deadlines" => [deadlines[u] for u in unassigned_uids],
            "service_times"    => [svc_times[u] for u in unassigned_uids],
            "UIDs"             => unassigned_uids,
        )
    else
        all_uids = vcat(unassigned_in["UIDs"], unassigned_uids)
        unassigned = Dict{String, Any}(
            "sat_identifiers" => [sat_ids[u] for u in all_uids],
            "demand_deadlines" => [deadlines[u] for u in all_uids],
            "service_times"    => [svc_times[u] for u in all_uids],
            "UIDs"             => all_uids,
        )
    end

    schedule = polish(schedule, demands, CostTable, MinTOFTable, sim)
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

    unrouted = Set{Int}(unassigned["UIDs"])
    sim_idx_for_uid = [name_to_idx[sat_ids[uid]] for uid in demands["UIDs"]]

    depot_idxs  = findall(n -> startswith(n, "depot"), sim.names)
    dep_idx     = depot_idxs[(length(schedule) % length(depot_idxs)) + 1]
    depot_name  = sim.names[dep_idx]

    all_depot  = [uid for veh in schedule for uid in veh.visitedUID if uid < 0]
    depot_uid  = isempty(all_depot) ? -1 : minimum(all_depot) - 1

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
                                          MinTOFTable, sim_idx_for_uid)
        best_uid === nothing && break

        cand_idx = sim_idx_for_uid[best_uid]
        tof, _   = MinTOFTable[(current_sim_idx, cand_idx)]
        svc_time = svc_times[best_uid]
        arr_time = current_time + tof

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

    schedule = polish(schedule, demands, CostTable, MinTOFTable, sim)

    if isempty(unrouted)
        return schedule, unassigned
    else
        remaining_uids = collect(unrouted)
        unassigned = Dict{String, Any}(
            "sat_identifiers" => [sat_ids[u] for u in remaining_uids],
            "demand_deadlines" => [deadlines[u] for u in remaining_uids],
            "service_times"    => [svc_times[u] for u in remaining_uids],
            "UIDs"             => remaining_uids,
        )
        return schedule, unassigned
    end
end











function remove_from_archive!(A, idx)
    deleteat!(A.solutions, idx)
    deleteat!(A.unassigned_sets, idx)
    deleteat!(A.total_deltaV, idx)
    deleteat!(A.total_serv_time_unassigned, idx)
    deleteat!(A.total_vehicles_used, idx)
end


function dominates(A, ai, B, bi)
    A.total_deltaV[ai] <= B.total_deltaV[bi] &&
    A.total_serv_time_unassigned[ai] <= B.total_serv_time_unassigned[bi] &&
    A.total_vehicles_used[ai] <= B.total_vehicles_used[bi] &&
    (A.total_deltaV[ai] < B.total_deltaV[bi] ||
     A.total_serv_time_unassigned[ai] < B.total_serv_time_unassigned[bi] ||
     A.total_vehicles_used[ai] < B.total_vehicles_used[bi])
end

function update_archive!(F, G)
    to_rm_f = Int[]
    for i in eachindex(F.solutions)
        for j in eachindex(G.solutions)
            if dominates(G, j, F, i)
                push!(to_rm_f, i)
                break
            end
        end
    end
    for idx in sort(to_rm_f, rev=true)
        remove_from_archive!(F, idx)
    end

    to_rm_g = Int[]
    for j in eachindex(G.solutions)
        for i in eachindex(F.solutions)
            if dominates(F, i, G, j)
                push!(to_rm_g, j)
                break
            end
        end
    end
    for idx in sort(to_rm_g, rev=true)
        remove_from_archive!(G, idx)
    end
end

_new_archive() = Archive(Vector{Vector{vehicle}}(), Vector{Union{Nothing,Dict}}(), Float64[], Float64[], Int[])

function prune_archive!(A)
    n = length(A.solutions)
    dominated = zeros(Bool, n)
    Threads.@threads for i in 1:n
        for j in 1:n
            i == j && continue
            if dominates(A, j, A, i)
                dominated[i] = true
                break
            end
        end
    end
    for idx in findall(dominated) |> reverse
        remove_from_archive!(A, idx)
    end
end

function _add_to_archive!(A, sol, un)
    dv = isempty(sol) ? 0.0 : sum(sum(veh.costs) for veh in sol)
    us = un === nothing ? 0.0 : sum(un["service_times"])
    push!(A.solutions, sol)
    push!(A.unassigned_sets, un)
    push!(A.total_deltaV, dv)
    push!(A.total_serv_time_unassigned, us)
    push!(A.total_vehicles_used, length(sol))
end

Base.@noinline function _seed_archive(sol, un)
    A = Archive(Vector{Vector{vehicle}}(), Vector{Union{Nothing,Dict}}(), Float64[], Float64[], Int[])
    _add_to_archive!(A, sol, un)
    return A
end

function MDLS(maxiter, demands, simulation, cost_table, mintof_table; nvehicles=25)

    init_sol, unassigned = make_init_schedule(demands, simulation, nvehicles=nvehicles)
    sim_obj = load_sim()

    F = _seed_archive(init_sol, unassigned)

    op_names = ["opt_times", "opt_times2", "consolidate", "swap_cross", "swap_intra", "create_veh", "prune"]
    op_times = zeros(7)

    operators = [
        (s, u) -> (opt_times(s, demands, cost_table, mintof_table, sim_obj), u),
        (s, u) -> (opt_times2(s, demands, cost_table, mintof_table, sim_obj), u),
        (s, u) -> consolidate_demands(s, demands, cost_table, mintof_table, sim_obj, u),
        (s, u) -> (swap_cross_vehicle(s, demands, cost_table, mintof_table, sim_obj), u),
        (s, u) -> (swap_intratour(s, demands, cost_table, mintof_table, sim_obj), u),
        (s, u) -> create_vehicle(s, demands, u, cost_table, mintof_table, sim_obj),
    ]

    for iter in 1:maxiter
        idx = rand(1:length(F.solutions))
        x = copy_schedule(F.solutions[idx])
        u_x = F.unassigned_sets[idx]

        G = _new_archive()

        tasks = [Threads.@spawn op(x, u_x) for op in operators]
        for (k, t) in enumerate(tasks)
            op_times[k] += @elapsed begin
                new_sol, new_u = fetch(t)
                _add_to_archive!(G, new_sol, new_u)
            end
        end

        update_archive!(F, G)
        for j in eachindex(G.solutions)
            _add_to_archive!(F, G.solutions[j], G.unassigned_sets[j])
        end
        op_times[7] += @elapsed prune_archive!(F)
        isempty(F.solutions) && break
    end

    @info "MDLS operator timing (total seconds across all iterations):" *
          join(["\n  $(op_names[k]) = $(round(op_times[k]; digits=2))s" for k in 1:7])

    return F
end












@info "Running MDLS …"
archive = MDLS(3000, demands, load_sim(), cost_table, mintof_table)
@info "MDLS complete" n_solutions=length(archive.solutions)

@save joinpath(@__DIR__, "outputs", "mdls_archive.jld2") archive

fig = plot_pareto(archive)
save(joinpath(@__DIR__, "outputs", "pareto_front.png"), fig)










