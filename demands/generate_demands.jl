# demands/generate_demands.jl
# generate_demands: produces (sat_identifiers, demand_deadlines, service_times)
# from a Simulation object and a demand_params Dict.

using JLD2
using Random
using Distributions
using StatsBase: Weights, sample
using Dates

# ── Distribution sampler ───────────────────────────────────────────────────────
"""
    sample_from_dist(disttype, lo, hi, n, rng) → Vector{Float64}

Draw `n` samples in [lo, hi] using the specified distribution type.
Supported: "normal", "uniform"
"""
function sample_from_dist(disttype::String, lo::Float64, hi::Float64,
                          n::Int, rng::AbstractRNG) :: Vector{Float64}
    if disttype == "normal"
        μ = (lo + hi) / 2.0
        σ = (hi - lo) / 6.0          # ±3σ covers ~99.7% of [lo,hi]
        d = Normal(μ, σ)
        return clamp.(rand(rng, d, n), lo, hi)
    elseif disttype == "uniform"
        return lo .+ (hi - lo) .* rand(rng, n)
    else
        error("Unknown disttype \"$disttype\". Supported: \"normal\", \"uniform\"")
    end
end

# ── Main entry point ───────────────────────────────────────────────────────────
"""
    generate_demands(simulation, demand_params)
        → (sat_identifiers, demand_deadlines, service_times)

Generate service demands for satellites within a ΔV range from the depot.

# Required keys in `demand_params`
| Key             | Type              | Description |
|-----------------|-------------------|-------------|
| `"num_demands"` | Int               | Total demands to generate |
| `"deltaV_dist"` | Float64           | Max ΔV from depot [m/s] — filters candidate sats |
| `"time_dist"`   | [t_min, t_max]    | Deadline range [days] |
| `"service_times"` | [s_min, s_max]  | Service time range [days] |
| `"disttype"`    | String            | "normal" or "uniform" |
| `"seed"`        | Int               | RNG seed for reproducibility |

# Optional keys
| Key               | Type | Description |
|-------------------|------|-------------|
| `"num_satellites"` | Int | Fix satellite pool size (num_demands >= num_satellites) |
| `"type"`           | String | "random" (default); "physical" reserved for future |
"""
function generate_demands(simulation, demand_params::Dict;
                          cost_table=nothing) :: Dict{String, Any}

    # ── Parse params ──────────────────────────────────────────────────────────
    num_demands  = Int(demand_params["num_demands"])
    dv_limit     = Float64(demand_params["deltaV_dist"])
    time_dist    = Float64.(demand_params["time_dist"])        # [t_min, t_max] days
    svc_range    = Float64.(demand_params["service_times"])    # [s_min, s_max] days
    disttype     = String(demand_params["disttype"])
    seed         = Int(demand_params["seed"])
    num_sats_opt = get(demand_params, "num_satellites", nothing)
    dem_type     = get(demand_params, "type", "random")

    dem_type != "random" &&
        @warn "demand type \"$dem_type\" not yet implemented — falling back to \"random\""

    num_sats_opt !== nothing && num_demands < Int(num_sats_opt) &&
        error("num_demands ($num_demands) must be >= num_satellites ($(Int(num_sats_opt)))")

    rng = MersenneTwister(seed)

    # ── Load cost table (or use pre-loaded) ───────────────────────────────────
    CostTable = if cost_table !== nothing
        cost_table
    else
        cost_table_path = "outputs/cost_table.jld2"
        isfile(cost_table_path) ||
            error("Cost table not found at $cost_table_path — run gen_cost_table(simulation) first.")
        @info "Loading cost table from disk ..."
        local CostTable
        @load cost_table_path CostTable
        @info "Cost table loaded" n_entries=length(CostTable)
        CostTable
    end

    # ── Find depot indices ────────────────────────────────────────────────────
    depot_idxs   = findall(n -> startswith(n, "depot"), simulation.names)
    isempty(depot_idxs) && error("No depot nodes found in simulation.")
    depot_idx_set = Set(depot_idxs)

    sat_idxs     = findall(n -> startswith(n, "sat"), simulation.names)
    sat_idx_set  = Set(sat_idxs)

    # ── Single pass over cost table: build per-sat min ΔV and max TOF ─────────
    @info "Scanning cost table for candidate satellites ..."
    sat_min_dv  = Dict{Int, Float64}()   # sat_idx → min ΔV from any depot
    sat_max_tof = Dict{Int, Float64}()   # sat_idx → max feasible TOF [days]

    for (key, val) in CostTable
        k_from, k_to, dep, arr = key
        k_from in depot_idx_set || continue
        k_to   in sat_idx_set   || continue
        val >= 1e7              && continue   # mask invalid entries

        tof = arr - dep
        cur_dv  = get(sat_min_dv,  k_to, Inf)
        cur_tof = get(sat_max_tof, k_to, 0.0)
        val < cur_dv  && (sat_min_dv[k_to]  = val)
        tof > cur_tof && (sat_max_tof[k_to] = tof)
    end

    # ── Filter to candidates within ΔV limit ──────────────────────────────────
    candidate_sats  = String[]
    max_tof_for_sat = Dict{String, Float64}()

    for si in sat_idxs
        get(sat_min_dv, si, Inf) > dv_limit && continue
        sat_name = simulation.names[si]
        push!(candidate_sats, sat_name)
        max_tof_for_sat[sat_name] = get(sat_max_tof, si, time_dist[2])
    end

    isempty(candidate_sats) &&
        error("No satellites found within deltaV_dist=$dv_limit m/s from any depot.")

    @info "generate_demands" candidate_sats=length(candidate_sats) num_demands=num_demands disttype=disttype

    # ── Build satellite pool ──────────────────────────────────────────────────
    pool = if num_sats_opt !== nothing
        n_pool = Int(num_sats_opt)
        n_pool > length(candidate_sats) &&
            @warn "num_satellites=$n_pool exceeds candidate count=$(length(candidate_sats)); using all candidates"
        n_take = min(n_pool, length(candidate_sats))
        # Sample unique satellites using disttype (index-based for normal: centre-biased)
        if disttype == "normal"
            weights = [exp(-0.5 * ((i - (length(candidate_sats)+1)/2) /
                                   (length(candidate_sats)/6))^2)
                       for i in 1:length(candidate_sats)]
            weights ./= sum(weights)
            chosen_idx = Set{Int}()
            while length(chosen_idx) < n_take
                push!(chosen_idx, sample(rng, 1:length(candidate_sats),
                                         Weights(weights)))
            end
            candidate_sats[collect(chosen_idx)]
        else
            candidate_sats[randperm(rng, length(candidate_sats))[1:n_take]]
        end
    else
        candidate_sats
    end

    # ── Generate demands ───────────────────────────────────────────────────────
    sat_identifiers = Vector{String}(undef, num_demands)
    demand_deadlines = Vector{Float64}(undef, num_demands)
    service_times_out = Vector{Float64}(undef, num_demands)

    for k in 1:num_demands
        # Sample satellite from pool uniformly
        sat_name = pool[rand(rng, 1:length(pool))]

        # Deadline upper bound for this sat
        dl_hi = min(Float64(time_dist[2]), max_tof_for_sat[sat_name])
        dl_lo = Float64(time_dist[1])
        dl_hi < dl_lo && (dl_hi = dl_lo + 1.0)   # safety floor

        deadline     = sample_from_dist(disttype, dl_lo, dl_hi, 1, rng)[1]
        service_time = sample_from_dist(disttype, svc_range[1], svc_range[2], 1, rng)[1]

        sat_identifiers[k]    = sat_name
        demand_deadlines[k]   = deadline
        service_times_out[k]  = service_time
    end

    # Asset values: look up per-satellite value if provided, else default to V1 value
    sat_values_param = get(demand_params, "sat_values", Dict{String,Float64}())
    default_asset_val = get(demand_params, "default_asset_value", 1_251_000.0)
    asset_values_out = [get(sat_values_param, sat_identifiers[k], default_asset_val)
                        for k in 1:num_demands]

    demands = Dict{String, Any}(
        "sat_identifiers" => sat_identifiers,
        "demand_deadlines" => demand_deadlines,
        "service_times"    => service_times_out,
        "asset_values"     => asset_values_out,
        "UIDs"             => collect(1:num_demands)
    )

    mkpath("outputs/demands")
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    fname     = "outputs/demands/demands_$timestamp.jld2"
    @save fname demands
    @info "Demands saved to $fname"

    return demands
end




function gen_UID(demands)
    n = length(demands["sat_identifiers"])
    UID = collect(1:n)  
    return UID
end



"""
    load_demands(filename=nothing) → Dict{String, Any}

Load a demands dict from `outputs/demands/`.
- No argument: loads the most recently saved demands file.
- `filename`: loads the specified file (full path or filename within `outputs/demands/`).
"""
function load_demands(filename=nothing) :: Dict{String, Any}
    if filename !== nothing
        path = isfile(filename) ? filename : joinpath("outputs/demands", filename)
        isfile(path) || error("Demands file not found: $path")
    else
        dir   = "outputs/demands"
        isdir(dir) || error("No demands directory found at $dir — run generate_demands() first.")
        files = filter(f -> endswith(f, ".jld2"), readdir(dir))
        isempty(files) && error("No demands files found in $dir — run generate_demands() first.")
        path  = joinpath(dir, last(sort(files)))  # sort by name = sort by timestamp
        @info "Loading most recent demands file: $path"
    end
    local demands
    @load path demands

    demands["UIDs"] = gen_UID(demands)
    return demands
end
