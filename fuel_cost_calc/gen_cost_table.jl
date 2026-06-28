
using Base.Threads
using JLD2
using ProgressMeter

include(joinpath(@__DIR__, "..", "cost", "cost_functions.jl"))


struct table_pairs
    satinit::Int
    satarr::Int
    dep::Float64
    arr::Float64
end




function build_cost_table(sim, keys; type="LT", prog=nothing)

    vals = Vector{Float64}(undef, length(keys))

    @threads for i in eachindex(keys)

      if type=="LT"
        vals[i] = LT_cost_calculation(sim, keys[i]...)
      elseif type=="HT"
        vals[i] = HT_cost_calculation(sim, keys[i]...)
      else
        vals[i] = min(LT_cost_calculation(sim, keys[i]...), HT_cost_calculation(sim, keys[i]...))
      end

      prog !== nothing && next!(prog)

    end

    return Dict(keys .=> vals)
end


# ── Plane grouping ────────────────────────────────────────────────────────────
"""
    get_plane_groups(sim, sats; raan_bin_deg=5.0) → Dict{Int,Vector{Int}}

Group satellite indices by orbital plane using RAAN at t=0.
Returns plane_bin_id → [sat_idx, ...].
"""
function get_plane_groups(sim, sats; raan_bin_deg=5.0)
    raan_bin = deg2rad(raan_bin_deg)
    groups = Dict{Int, Vector{Int}}()
    h5open(sim.traj_file, "r") do f
        for idx in sats
            name = sim.names[idx]
            oe   = f["$name/orbital_elements"][:, 1]   # [a, i, raan, nu] at t=0
            raan = oe[3]
            bin_id = floor(Int, raan / raan_bin)
            push!(get!(groups, bin_id, Int[]), idx)
        end
    end
    return groups
end


# ── LT spiral only (no phasing, plane-level) ──────────────────────────────────
function LT_spiral_only(sim, rep_i::Int, rep_j::Int,
                        dep_days::Float64, arr_days::Float64) :: Float64
    tof_days = arr_days - dep_days
    tof_days <= 0.0 && return 1e8

    k_dep = nearest_idx(sim.times, dep_days * 86400.0)
    k_arr = nearest_idx(sim.times, arr_days * 86400.0)

    name_i = sim.names[rep_i]
    name_j = sim.names[rep_j]

    a_i = inc_i = raan_i = 0.0
    a_j = inc_j = raan_j = 0.0

    h5open(sim.traj_file, "r") do f
        oe_i = f["$name_i/orbital_elements"][:, k_dep]
        oe_j = f["$name_j/orbital_elements"][:, k_arr]
        a_i, inc_i, raan_i = oe_i[1], oe_i[2], oe_i[3]
        a_j, inc_j, raan_j = oe_j[1], oe_j[2], oe_j[3]
    end

    same_plane = abs(inc_i - inc_j) < 0.01 && abs(raan_i - raan_j) < 0.01
    same_plane && return 0.0  # phasing-only case, handled separately

    result = try
        calculate_transfer_cost(a_i, inc_i, raan_i, a_j, inc_j, raan_j, tof_days)
    catch
        return 1e8
    end

    return result["deltaV_total"]
end


# ── LT phasing only (per-satellite, analytical) ───────────────────────────────
function LT_phasing_only(sim, sat_i::Int, sat_j::Int,
                         dep_days::Float64, arr_days::Float64) :: Float64
    tof_days = arr_days - dep_days
    tof_days <= 0.0 && return 1e8

    k_dep = nearest_idx(sim.times, dep_days * 86400.0)
    k_arr = nearest_idx(sim.times, arr_days * 86400.0)

    name_i = sim.names[sat_i]
    name_j = sim.names[sat_j]

    a_i = nu_i = 0.0
    a_j = nu_j = 0.0

    h5open(sim.traj_file, "r") do f
        oe_i = f["$name_i/orbital_elements"][:, k_dep]
        oe_j = f["$name_j/orbital_elements"][:, k_arr]
        a_i, nu_i = oe_i[1], oe_i[4]
        a_j, nu_j = oe_j[1], oe_j[4]
    end

    n_i         = sqrt(MU_LT / a_i^3)
    nu_i_at_arr = mod(nu_i + n_i * tof_days * 86400.0, 2π)
    dv_phase    = phasing(a_j, nu_i_at_arr, nu_j, 1) * 1000.0  # km/s → m/s

    return isfinite(dv_phase) ? dv_phase : 1e8
end


# ── Fast sat-to-sat table: plane-level spiral + per-sat phasing ───────────────
"""
    build_cost_table_fast(sim, sat_to_sat_keys, plane_groups; prog=nothing)

Speedup vs build_cost_table: compute the expensive NLsolve spiral ΔV once per
(plane_i, plane_j, dep, arr), then add cheap analytical phasing per satellite pair.
"""
function build_cost_table_fast(sim, sat_to_sat_keys, plane_groups; prog=nothing)

    # sat_idx → plane_bin_id; plane_bin_id → representative sat
    sat_to_plane = Dict{Int, Int}()
    rep_sat      = Dict{Int, Int}()
    for (pid, ss) in plane_groups
        rep_sat[pid] = ss[1]
        for s in ss
            sat_to_plane[s] = pid
        end
    end

    plane_ids = sort(collect(keys(plane_groups)))

    # Unique (dep, arr) pairs from the key list
    dep_arr_set = Set{Tuple{Float64,Float64}}()
    for k in sat_to_sat_keys
        push!(dep_arr_set, (k[3], k[4]))
    end
    dep_arr_pairs = collect(dep_arr_set)

    # Phase 1: spiral ΔV per cross-plane (plane_i, plane_j, dep, arr) — threaded
    cross_pairs = Tuple{Int,Int,Float64,Float64}[]
    for pi in plane_ids, pj in plane_ids, (dep, arr) in dep_arr_pairs
        pi != pj && push!(cross_pairs, (pi, pj, dep, arr))
    end

    spiral_vals = Vector{Float64}(undef, length(cross_pairs))
    @threads for idx in eachindex(cross_pairs)
        pi, pj, dep, arr = cross_pairs[idx]
        spiral_vals[idx] = LT_spiral_only(sim, rep_sat[pi], rep_sat[pj], dep, arr)
        prog !== nothing && next!(prog)
    end
    spiral_table = Dict(cross_pairs .=> spiral_vals)

    # Phase 2: per-satellite phasing + spiral lookup
    vals = Vector{Float64}(undef, length(sat_to_sat_keys))
    @threads for i in eachindex(sat_to_sat_keys)
        sat_i, sat_j, dep, arr = sat_to_sat_keys[i]
        pi = sat_to_plane[sat_i]
        pj = sat_to_plane[sat_j]

        if pi == pj
            vals[i] = LT_phasing_only(sim, sat_i, sat_j, dep, arr)
        else
            dv_spiral = get(spiral_table, (pi, pj, dep, arr), 1e8)
            dv_phase  = LT_phasing_only(sim, sat_i, sat_j, dep, arr)
            vals[i]   = (dv_spiral >= 1e8 || dv_phase >= 1e8) ? 1e8 : dv_spiral + dv_phase
        end

        prog !== nothing && next!(prog)
    end

    return Dict(sat_to_sat_keys .=> vals)
end




"""
    load_cost_table() → Dict

Load the saved cost table from `outputs/cost_table.jld2`.
"""
function load_cost_table()
    path = "outputs/cost_table.jld2"
    isfile(path) || error("No cost table found at $path — run gen_cost_table() first.")
    local CostTable
    @load path CostTable
    return CostTable
end

function gen_cost_table(sim; tof_step = 30.0, t_end = 400.0)

        if isfile("outputs/cost_table.jld2")

        println("Cost table file found. Load existing (y) or rebuild (n)?")
        choice = strip(readline())

          if choice == "y"

            @load "outputs/cost_table.jld2" CostTable
              return CostTable

          end

        end

        arrivals   = collect(tof_step:tof_step:t_end)
        departures = collect(tof_step:tof_step:t_end)

        depots  = findall(startswith("depot"),  sim.names)
        sats    = findall(startswith("sat"),    sim.names)
        debris  = findall(startswith("debris"), sim.names)

        plane_groups = get_plane_groups(sim, sats)
        @info "Plane groups" n_planes=length(plane_groups)

        has_debris = !isempty(debris)

        # ── sat-sat (full 4D — required for multi-shell constellations) ───────
        sat_to_sat_keys = [(satinit, satarr, dep, dep + arr)
                            for arr in arrivals
                            for dep in departures
                            for satinit in sats
                            for satarr in sats
                            if satinit != satarr]

        # ── sat/depot cross terms ─────────────────────────────────────────────
        sat_to_depot_keys = [(sat, depot, dep, dep + arr)
                            for arr in arrivals
                            for dep in departures
                            for sat in sats
                            for depot in depots]

        depot_to_sat_keys = [(depot, sat, dep, dep + arr)
                            for arr in arrivals
                            for dep in departures
                            for sat in sats
                            for depot in depots]

        # ── debris cross terms (only built if debris nodes exist) ─────────────
        if has_debris
            sat_to_debris_keys = [(sat, deb, dep, dep + arr)
                                    for arr in arrivals
                                    for dep in departures
                                    for sat in sats
                                    for deb in debris]

            debris_to_sat_keys = [(deb, sat, dep, dep + arr)
                                    for arr in arrivals
                                    for dep in departures
                                    for sat in sats
                                    for deb in debris]

            depot_to_debris_keys = [(depot, deb, dep, dep + arr)
                                    for arr in arrivals
                                    for dep in departures
                                    for depot in depots
                                    for deb in debris]

            debris_to_depot_keys = [(deb, depot, dep, dep + arr)
                                    for arr in arrivals
                                    for dep in departures
                                    for depot in depots
                                    for deb in debris]
        end

        # ── pre-compute cross-plane spiral count for progress bar ────────────
        dep_arr_count = length(departures) * length(arrivals)
        n_planes      = length(plane_groups)
        n_cross_pairs = n_planes * (n_planes - 1) * dep_arr_count

        @info "Building cost tables" n_sat_pairs=length(sat_to_sat_keys) n_depot_sat_pairs=length(depot_to_sat_keys) n_planes=n_planes n_spiral_pairs=n_cross_pairs has_debris=has_debris

        # ── single shared progress bar ────────────────────────────────────────
        total_pairs = n_cross_pairs + length(sat_to_sat_keys) + length(depot_to_sat_keys) + length(sat_to_depot_keys)
        has_debris && (total_pairs += length(sat_to_debris_keys) + length(debris_to_sat_keys) +
                                      length(depot_to_debris_keys) + length(debris_to_depot_keys))
        prog = Progress(total_pairs; desc="Cost tables: ", barlen=40, showspeed=true)

        # ── spawn all tasks ───────────────────────────────────────────────────
        task_sat_to_sat   = @spawn build_cost_table_fast(sim, sat_to_sat_keys, plane_groups; prog=prog)
        task_depot_to_sat = @spawn build_cost_table(sim, depot_to_sat_keys; type="LT", prog=prog)
        task_sat_to_depot = @spawn build_cost_table(sim, sat_to_depot_keys; type="LT", prog=prog)

        if has_debris
            task_sat_to_debris   = @spawn build_cost_table(sim, sat_to_debris_keys;   prog=prog)
            task_debris_to_sat   = @spawn build_cost_table(sim, debris_to_sat_keys;   prog=prog)
            task_depot_to_debris = @spawn build_cost_table(sim, depot_to_debris_keys; prog=prog)
            task_debris_to_depot = @spawn build_cost_table(sim, debris_to_depot_keys; prog=prog)
        end

        # ── fetch results ─────────────────────────────────────────────────────
        CostTable_sat_to_sat   = fetch(task_sat_to_sat)
        CostTable_depot_to_sat = fetch(task_depot_to_sat)
        CostTable_sat_to_depot = fetch(task_sat_to_depot)

        if has_debris
            CostTable_sat_to_debris   = fetch(task_sat_to_debris)
            CostTable_debris_to_sat   = fetch(task_debris_to_sat)
            CostTable_depot_to_debris = fetch(task_depot_to_debris)
            CostTable_debris_to_depot = fetch(task_debris_to_depot)
        end
        finish!(prog)

        # ── merge ─────────────────────────────────────────────────────────────
        @info "Merging cost tables ..."
        CostTable = merge(CostTable_depot_to_sat, CostTable_sat_to_depot, CostTable_sat_to_sat)

        if has_debris
            CostTable = merge(CostTable,
                              CostTable_sat_to_debris,
                              CostTable_debris_to_sat,
                              CostTable_depot_to_debris,
                              CostTable_debris_to_depot)
        end

        @info "Saving cost table ..."
        @save "outputs/cost_table.jld2" CostTable
        @info "Cost tables complete, saved to outputs/cost_table.jld2"

        return CostTable

end

function build_min_tof_table()
    path = "outputs/min_tof_table.jld2"
    if isfile(path)
        local MinTOFTable
        @load path MinTOFTable
        return MinTOFTable
    end

    CostTable = load_cost_table()
    MinTOFTable = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()

    for ((from, to, dep, arr), dv) in CostTable
        dv >= 1e7 && continue
        tof = arr - dep
        key = (from, to)
        if !haskey(MinTOFTable, key) || tof < MinTOFTable[key][1]
            MinTOFTable[key] = (tof, dv)
        end
    end

    @save path MinTOFTable
    return MinTOFTable
end


