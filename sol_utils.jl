
include(joinpath(@__DIR__, "fuel_cost_calc", "gen_cost_table.jl"))

mutable struct vehicle
    visitedUID::Vector{Int}
    visitedSAT::Vector{String}
    arrivals::Vector{Float64}
    departures::Vector{Float64}
    costs::Vector{Float64}
end

function copy_schedule(schedule)
    [vehicle(copy(v.visitedUID), copy(v.visitedSAT), copy(v.arrivals),
             copy(v.departures), copy(v.costs)) for v in schedule]
end

mutable struct Archive
    solutions::Vector{Vector{vehicle}}
    unassigned_sets::Vector{Union{Nothing, Dict}}
    total_deltaV::Vector{Float64}
    total_serv_time_unassigned::Vector{Float64}
    total_vehicles_used::Vector{Int}
end

function snap_cost(CostTable, from_idx, to_idx, dep_epoch, arr_epoch)
    dep_snap = clamp(round(dep_epoch / 15.0) * 15.0, 0.0, 400.0)
    arr_snap = clamp(round(arr_epoch / 15.0) * 15.0, 15.0, 400.0)
    get(CostTable, (from_idx, to_idx, dep_snap, arr_snap), Inf)
end

function get_best_next(current_sim_idx, current_time, unrouted_set,
                       sat_ids, svc_times, deadlines,
                       MinTOFTable, sim_idx_for_uid)
    best_uid = nothing
    best_dv  = Inf
    for uid in unrouted_set
        cand_idx = sim_idx_for_uid[uid]
        key      = (current_sim_idx, cand_idx)
        haskey(MinTOFTable, key) || continue
        tof, dv = MinTOFTable[key]
        dv >= best_dv && continue
        current_time + tof + svc_times[uid] > deadlines[uid] && continue
        best_dv  = dv
        best_uid = uid
    end
    return best_uid, best_dv
end

function make_init_schedule(demands, sim; nvehicles=10, dv_budget=5000.0, start_time=0.0, refuel_time=0.5)
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    MinTOFTable = build_min_tof_table()
    depot_idxs  = findall(n -> startswith(n, "depot"), sim.names)

    sat_ids      = demands["sat_identifiers"]
    svc_times    = demands["service_times"]
    deadlines    = demands["demand_deadlines"]
    sim_idx_for_uid = [name_to_idx[sat_ids[uid]] for uid in demands["UIDs"]]

    unrouted   = Set{Int}(demands["UIDs"])
    depot_uid  = 0
    schedule   = vehicle[]

    for v in 1:nvehicles
        isempty(unrouted) && break

        dep_idx      = depot_idxs[(v - 1) % length(depot_idxs) + 1]
        depot_name   = sim.names[dep_idx]

        depot_uid -= 1
        visited_uids = [depot_uid]
        visited_sats = [depot_name]
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
    end

    unassigned = isempty(unrouted) ? nothing : Dict{String, Any}(
        "sat_identifiers" => [sat_ids[u] for u in unrouted],
        "demand_deadlines" => [deadlines[u] for u in unrouted],
        "service_times"    => [svc_times[u] for u in unrouted],
        "UIDs"             => collect(unrouted),
    )

    return schedule, unassigned
end
