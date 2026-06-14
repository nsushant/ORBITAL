
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




function build_cost_table(sim, keys; type="hybrid", prog=nothing)

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

function gen_cost_table(sim; tof_steps = 15)

        if isfile("outputs/cost_table.jld2")

        println("Cost table file found. Load existing (y) or rebuild (n)?")
        choice = strip(readline())

          if choice == "y"

            @load "outputs/cost_table.jld2" CostTable
              return CostTable

          end

        end

        arrivals   = collect(15.0:15.0:400.0)
        departures = collect(15.0:15.0:400.0)

        depots  = findall(startswith("depot"),  sim.names)
        sats    = findall(startswith("sat"),    sim.names)
        debris  = findall(startswith("debris"), sim.names)

        has_debris = !isempty(debris)

        # ── sat-sat (epoch-independent: single departure at t=0) ──────────────
        sat_to_sat_keys = [(satinit, satarr, 0.0, arr)
                            for arr in arrivals
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

        @info "Building cost tables" n_sat_pairs=length(sat_to_sat_keys) n_depot_sat_pairs=length(depot_to_sat_keys) has_debris=has_debris

        # ── single shared progress bar ────────────────────────────────────────
        total_pairs = length(sat_to_sat_keys) + length(depot_to_sat_keys) + length(sat_to_depot_keys)
        has_debris && (total_pairs += length(sat_to_debris_keys) + length(debris_to_sat_keys) +
                                      length(depot_to_debris_keys) + length(debris_to_depot_keys))
        prog = Progress(total_pairs; desc="Cost tables: ", barlen=40, showspeed=true)

        # ── spawn all tasks ───────────────────────────────────────────────────
        task_sat_to_sat   = @spawn build_cost_table(sim, sat_to_sat_keys;   prog=prog)
        task_depot_to_sat = @spawn build_cost_table(sim, depot_to_sat_keys; prog=prog)
        task_sat_to_depot = @spawn build_cost_table(sim, sat_to_depot_keys; prog=prog)

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
        # ── broadcast sat-sat to 4D (dep, arr epochs) — parallelised ────────────
        @info "Broadcasting sat→sat to 4D ..."
        bc_inputs = [(satinit, satarr, dep, arr)
                     for arr in arrivals
                     for dep in departures
                     for satinit in sats
                     for satarr in sats
                     if satinit != satarr && arr > dep]

        bc_keys = Vector{Tuple{Int,Int,Float64,Float64}}(undef, length(bc_inputs))
        bc_vals = Vector{Float64}(undef, length(bc_inputs))

        @threads for i in eachindex(bc_inputs)
            satinit, satarr, dep, arr = bc_inputs[i]
            bc_keys[i] = (satinit, satarr, dep, dep + arr)
            bc_vals[i] = CostTable_sat_to_sat[(satinit, satarr, 0.0, arr)]
        end

        CostTable_sat_to_sat4D = Dict(bc_keys .=> bc_vals)
        @info "4D broadcast done" n_entries=length(CostTable_sat_to_sat4D)

        # ── merge ─────────────────────────────────────────────────────────────
        @info "Merging cost tables ..."
        CostTable = merge(CostTable_depot_to_sat, CostTable_sat_to_depot, CostTable_sat_to_sat4D)

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


