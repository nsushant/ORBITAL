using JSON3

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

function save_schedule_json(schedule, unassigned, path::String)
    json_sol = [
        Dict(
            "visitedUID" => veh.visitedUID,
            "visitedSAT" => veh.visitedSAT,
            "arrivals"   => veh.arrivals,
            "departures" => veh.departures,
            "costs"      => veh.costs,
        ) for veh in schedule
    ]
    open(path, "w") do f
        JSON3.write(f, Dict("schedule" => json_sol, "unassigned" => unassigned))
    end
end

mutable struct OctNode
    dv_lo::Float64; dv_hi::Float64
    us_lo::Float64; us_hi::Float64
    veh_lo::Float64; veh_hi::Float64
    ideal_dv::Float64; ideal_us::Float64; ideal_veh::Float64
    idx::Union{Nothing, Int}
    children::Union{Nothing, Vector{OctNode}}
end

mutable struct Archive
    solutions::Vector{Vector{vehicle}}
    unassigned_sets::Vector{Union{Nothing, Dict}}
    total_deltaV::Vector{Float64}
    total_serv_time_unassigned::Vector{Float64}
    total_vehicles_used::Vector{Int}
    tree::Union{Nothing, OctNode}
end

# ── Octree helpers ─────────────────────────────────────────────────────────────

const OCTREE_MAX_DEPTH = 30

function _octant(node, dv, us, veh)
    mid_dv = (node.dv_lo + node.dv_hi) / 2
    mid_us = (node.us_lo + node.us_hi) / 2
    mid_veh = (node.veh_lo + node.veh_hi) / 2
    dv_bit = dv >= mid_dv ? 1 : 0
    us_bit = us >= mid_us ? 1 : 0
    veh_bit = veh >= mid_veh ? 1 : 0
    1 + dv_bit + us_bit * 2 + veh_bit * 4
end

function _init_octree(dv, us, veh, idx)
    pad_dv = max(abs(dv) * 0.01, 1.0)
    pad_us = max(abs(us) * 0.01, 1.0)
    pad_veh = max(abs(veh) * 0.01, 1.0)
    OctNode(dv - pad_dv, dv + pad_dv,
            us - pad_us, us + pad_us,
            veh - pad_veh, veh + pad_veh,
            dv, us, veh, idx, nothing)
end

function _split_leaf!(node)
    mid_dv = (node.dv_lo + node.dv_hi) / 2
    mid_us = (node.us_lo + node.us_hi) / 2
    mid_veh = (node.veh_lo + node.veh_hi) / 2
    children = OctNode[]
    for veh_bit in 0:1, us_bit in 0:1, dv_bit in 0:1
        push!(children, OctNode(
            dv_bit == 0 ? node.dv_lo : mid_dv,
            dv_bit == 0 ? mid_dv : node.dv_hi,
            us_bit == 0 ? node.us_lo : mid_us,
            us_bit == 0 ? mid_us : node.us_hi,
            veh_bit == 0 ? node.veh_lo : mid_veh,
            veh_bit == 0 ? mid_veh : node.veh_hi,
            Inf, Inf, Inf, nothing, nothing))
    end
    node.children = children
    node.idx = nothing
end

function _update_ideal!(node)
    node.ideal_dv = minimum(c.ideal_dv for c in node.children)
    node.ideal_us = minimum(c.ideal_us for c in node.children)
    node.ideal_veh = minimum(c.ideal_veh for c in node.children)
end

function _expand_to_fit!(node, dv, us, veh)
    changed = false
    if dv < node.dv_lo
        node.dv_lo = dv * 0.99
        changed = true
    end
    if dv > node.dv_hi
        node.dv_hi = dv * 1.01
        changed = true
    end
    if us < node.us_lo
        node.us_lo = us * 0.99
        changed = true
    end
    if us > node.us_hi
        node.us_hi = us * 1.01
        changed = true
    end
    if veh < node.veh_lo
        node.veh_lo = veh - 1
        changed = true
    end
    if veh > node.veh_hi
        node.veh_hi = veh + 1
        changed = true
    end
    changed
end

function _dominates_point(A, ai, dv, us, veh)
    ad = A.total_deltaV[ai]; au = A.total_serv_time_unassigned[ai]; av = A.total_vehicles_used[ai]
    ad <= dv && au <= us && av <= veh && (ad < dv || au < us || av < veh)
end

function is_dominated(node, dv, us, veh, A, exclude_idx=nothing)
    if dv < node.ideal_dv || us < node.ideal_us || veh < node.ideal_veh
        return false
    end
    if node.children === nothing
        if node.idx !== nothing && (exclude_idx === nothing || node.idx != exclude_idx)
            if _dominates_point(A, node.idx, dv, us, veh)
                return true
            end
        end
        return false
    else
        for child in node.children
            if is_dominated(child, dv, us, veh, A, exclude_idx)
                return true
            end
        end
        return false
    end
end

function octree_insert!(node, idx, A, depth=0)
    dv = A.total_deltaV[idx]
    us = A.total_serv_time_unassigned[idx]
    veh = A.total_vehicles_used[idx]

    if node.children === nothing
        if node.idx === nothing
            node.idx = idx
            node.ideal_dv = dv
            node.ideal_us = us
            node.ideal_veh = veh
            return true
        else
            old_idx = node.idx
            old_dv = A.total_deltaV[old_idx]
            old_us = A.total_serv_time_unassigned[old_idx]
            old_veh = A.total_vehicles_used[old_idx]

            old_le = old_dv <= dv && old_us <= us && old_veh <= veh
            new_le = dv <= old_dv && us <= old_us && veh <= old_veh
            old_lt = old_dv < dv || old_us < us || old_veh < veh
            new_lt = dv < old_dv || us < old_us || veh < old_veh

            if old_le && old_lt
                return false
            elseif new_le && new_lt
                node.idx = idx
                node.ideal_dv = dv
                node.ideal_us = us
                node.ideal_veh = veh
                return true
            else
                dv == old_dv && us == old_us && veh == old_veh && return false
                depth >= OCTREE_MAX_DEPTH && return false
                _split_leaf!(node)
                o1 = _octant(node, old_dv, old_us, old_veh)
                o2 = _octant(node, dv, us, veh)
                octree_insert!(node.children[o1], old_idx, A, depth + 1)
                octree_insert!(node.children[o2], idx, A, depth + 1)
                _update_ideal!(node)
                return true
            end
        end
    else
        o = _octant(node, dv, us, veh)
        changed = octree_insert!(node.children[o], idx, A, depth + 1)
        if changed
            _update_ideal!(node)
        end
        return changed
    end
end

function rebuild_octree!(A)
    n = length(A.solutions)
    if n == 0
        A.tree = nothing
        return
    end
    A.tree = _init_octree(A.total_deltaV[1], A.total_serv_time_unassigned[1], A.total_vehicles_used[1], 1)
    for i in 2:n
        _expand_to_fit!(A.tree, A.total_deltaV[i], A.total_serv_time_unassigned[i], A.total_vehicles_used[i])
        octree_insert!(A.tree, i, A)
    end
end

function build_min_dv_table(cost_table, n_nodes)
    tab = fill(Inf, n_nodes, n_nodes)
    for ((i, j, _, _), dv) in cost_table
        dv < tab[i, j] && (tab[i, j] = dv)
    end
    return tab
end

# ── Evaluation instrumentation (opt-in) ─────────────────────────────────────
# Set OOS_COUNT_EVALS=1 to count work units. COUNT_EVALS is a const Bool, so
# when unset the branches below constant-fold away and cost nothing.
#   SNAP_COST_CALLS — transfer-cost table lookups (the atomic work unit shared
#                     with the Python stack, where the same quantity is counted
#                     by wrapping the cost-table dict)
#   FULL_EVALS      — complete objective-vector evaluations of a whole solution,
#                     the unit pymoo terminates on ("n_eval")
const COUNT_EVALS     = get(ENV, "OOS_COUNT_EVALS", "0") == "1"
const SNAP_COST_CALLS = Threads.Atomic{Int}(0)
const FULL_EVALS      = Threads.Atomic{Int}(0)
reset_eval_counters!() = (SNAP_COST_CALLS[] = 0; FULL_EVALS[] = 0)
eval_counters() = (snap_cost_calls = SNAP_COST_CALLS[], full_evals = FULL_EVALS[])

const INFEASIBLE_LEG_COST = 1.0e6
# Edelbaum tables cover ~400 days (J2 precession period). Override with
# COST_TABLE_PERIOD if using a longer Lu table (e.g. 1825).
const COST_TABLE_PERIOD = parse(Float64, get(ENV, "COST_TABLE_PERIOD", "400.0"))

# For missions longer than COST_TABLE_PERIOD days, wrap departure epoch onto the cost table's
# time range using modular arithmetic (J2 precession is approximately periodic over ~400 days).
function snap_cost(CostTable, from_idx, to_idx, dep_epoch, arr_epoch)
    COUNT_EVALS && Threads.atomic_add!(SNAP_COST_CALLS, 1)
    tof     = arr_epoch - dep_epoch
    dep_w   = mod(dep_epoch, COST_TABLE_PERIOD)
    arr_w   = dep_w + tof
    dep_snap = clamp(round(dep_w / 15.0) * 15.0,  0.0, COST_TABLE_PERIOD)
    arr_snap = clamp(round(arr_w / 15.0) * 15.0, 15.0, COST_TABLE_PERIOD)
    get(CostTable, (from_idx, to_idx, dep_snap, arr_snap), INFEASIBLE_LEG_COST)
end

@inline function demand_ready(demands, uid::Int)
    av = get(demands, "available_times", nothing)
    av === nothing && return 0.0
    return Float64(av[uid])
end

function demand_ready_vec(demands)
    n = length(demands["demand_deadlines"])
    av = get(demands, "available_times", nothing)
    av === nothing && return zeros(Float64, n)
    return av
end

function subset_unassigned(uids, demands)
    sat_ids    = demands["sat_identifiers"]
    deadlines  = demands["demand_deadlines"]
    svc_times  = demands["service_times"]
    available  = demand_ready_vec(demands)
    asset_vals = get(demands, "asset_values", nothing)
    d = Dict{String, Any}(
        "sat_identifiers"  => [sat_ids[u] for u in uids],
        "demand_deadlines" => [deadlines[u] for u in uids],
        "available_times"  => [available[u] for u in uids],
        "service_times"    => [svc_times[u] for u in uids],
        "UIDs"             => collect(uids),
    )
    asset_vals !== nothing && (d["asset_values"] = [asset_vals[u] for u in uids])
    return d
end

# ── Weibull depreciation ───────────────────────────────────────────────────────
# Models satellite asset value decay with age using a Weibull survival function.
# k=1.5 (mild wear-out), λ=5 years characteristic life.
const WEIBULL_LAMBDA = 5.0   # years
const WEIBULL_K      = 1.5   # shape (k>1 → accelerating depreciation)

function weibull_depreciate(base_value::Float64, age_years::Float64) :: Float64
    age_years <= 0.0 && return base_value
    return base_value * exp(-(age_years / WEIBULL_LAMBDA)^WEIBULL_K)
end

function get_best_next(current_sim_idx, current_time, unrouted_set,
                       sat_ids, svc_times, deadlines,
                       MinTOFTable, sim_idx_for_uid, available)
    # Prefer demands that are already ready by arrival (no extra wait). If none,
    # allow waiting until available_times so later-year jobs can still be served.
    for allow_wait in (false, true)
        best_uid = nothing
        best_dv  = Inf
        for uid in unrouted_set
            cand_idx = sim_idx_for_uid[uid]
            key      = (current_sim_idx, cand_idx)
            haskey(MinTOFTable, key) || continue
            tof, dv = MinTOFTable[key]
            dv >= best_dv && continue
            arr = max(current_time + tof, available[uid])
            arr + svc_times[uid] > deadlines[uid] && continue
            !allow_wait && available[uid] > current_time + tof && continue
            best_dv  = dv
            best_uid = uid
        end
        best_uid !== nothing && return best_uid, best_dv
    end
    return nothing, Inf
end

function make_init_schedule(demands, sim; nvehicles=10, dv_budget=5000.0, start_time=0.0, refuel_time=0.5, min_tof_table=nothing)
    name_to_idx = Dict(sim.names[i] => i for i in eachindex(sim.names))
    MinTOFTable = min_tof_table !== nothing ? min_tof_table : build_min_tof_table()
    depot_idxs  = findall(n -> startswith(n, "depot"), sim.names)

    sat_ids      = demands["sat_identifiers"]
    svc_times    = demands["service_times"]
    deadlines    = demands["demand_deadlines"]
    available    = demand_ready_vec(demands)
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

    if isempty(unrouted)
        unassigned = nothing
    else
        uids_out   = collect(unrouted)
        unassigned = subset_unassigned(uids_out, demands)
    end

    return schedule, unassigned
end
