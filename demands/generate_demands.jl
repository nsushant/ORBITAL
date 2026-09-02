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
        → Dict with sat_identifiers, demand_deadlines, available_times, service_times

Generate service demands for satellites within a ΔV range from the depot.

# Required keys in `demand_params`
| Key             | Type              | Description |
|-----------------|-------------------|-------------|
| `"deltaV_dist"` | Float64           | Max ΔV from depot [m/s] — filters candidate sats |
| `"service_times"` | [s_min, s_max]  | Service time range [days] |
| `"disttype"`    | String            | "normal" or "uniform" |
| `"seed"`        | Int               | RNG seed for reproducibility |

Either `"num_demands"` + `"time_dist"` (legacy: deadlines in [t_min, t_max],
ready at t=0) or an epoch profile:

| `"horizon_years"`    | Int        | Number of demand-arrival years |
| `"demands_per_year"` | [n_lo, n_hi] | Uniform integer draw per year |
| `"near_window_days"` | [w_lo, w_hi] | Short contract window (default 6–18 mo) |
| `"far_window_days"`  | [w_lo, w_hi] | Long contract window (default 2–3 yr) |
| `"far_frac"`         | Float64    | Fraction of demands using the far window |

Epoch demands have `available_times` (ready epoch) and
`deadline = ready + window`.

# Optional keys
| Key               | Type | Description |
|-------------------|------|-------------|
| `"num_satellites"` | Int | Fix satellite pool size (ignored if ≤ 0) |
| `"type"`           | String | "random" (default); "physical" reserved for future |
"""
function generate_demands(simulation, demand_params::Dict;
                          cost_table=nothing) :: Dict{String, Any}

    # ── Parse params ──────────────────────────────────────────────────────────
    use_epoch    = haskey(demand_params, "horizon_years")
    num_demands  = use_epoch ? 0 : Int(demand_params["num_demands"])
    dv_limit     = Float64(demand_params["deltaV_dist"])
    time_dist    = haskey(demand_params, "time_dist") ?
                   Float64.(demand_params["time_dist"]) : [0.0, 0.0]
    svc_range    = Float64.(demand_params["service_times"])    # [s_min, s_max] days
    disttype     = String(demand_params["disttype"])
    seed         = Int(demand_params["seed"])
    num_sats_raw = get(demand_params, "num_satellites", nothing)
    num_sats_opt = (num_sats_raw === nothing || Int(num_sats_raw) <= 0) ?
                   nothing : Int(num_sats_raw)
    dem_type     = get(demand_params, "type", "random")

    dem_type != "random" &&
        @warn "demand type \"$dem_type\" not yet implemented — falling back to \"random\""

    !use_epoch && num_sats_opt !== nothing && num_demands < num_sats_opt &&
        error("num_demands ($num_demands) must be >= num_satellites ($num_sats_opt)")

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

    # ── Single pass over cost table: build per-sat min ΔV from depot ────────────
    @info "Scanning cost table for candidate satellites ..."
    sat_min_dv = Dict{Int, Float64}()   # sat_idx → min ΔV from any depot

    for (key, val) in CostTable
        k_from, k_to, dep, arr = key
        k_from in depot_idx_set || continue
        k_to   in sat_idx_set   || continue
        val >= 1e7              && continue   # mask invalid entries

        cur_dv = get(sat_min_dv, k_to, Inf)
        val < cur_dv && (sat_min_dv[k_to] = val)
    end

    # ── Filter to candidates within ΔV limit ──────────────────────────────────
    candidate_sats = String[]

    for si in sat_idxs
        get(sat_min_dv, si, Inf) > dv_limit && continue
        push!(candidate_sats, simulation.names[si])
    end

    isempty(candidate_sats) &&
        error("No satellites found within deltaV_dist=$dv_limit m/s from any depot.")

    @info "generate_demands" candidate_sats=length(candidate_sats) num_demands=num_demands disttype=disttype use_epoch=use_epoch

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
    sat_identifiers, demand_deadlines, available_times, service_times_out =
        if use_epoch
            _generate_epoch_demands(pool, demand_params, svc_range, disttype, rng)
        else
            _generate_legacy_demands(pool, num_demands, time_dist, svc_range, disttype, rng)
        end
    num_demands = length(sat_identifiers)

    # Asset values: look up per-satellite value if provided, else default to V1 value
    sat_values_param = get(demand_params, "sat_values", Dict{String,Float64}())
    default_asset_val = get(demand_params, "default_asset_value", 1_251_000.0)
    asset_values_out = [get(sat_values_param, sat_identifiers[k], default_asset_val)
                        for k in 1:num_demands]

    n_near = count(k -> demand_deadlines[k] - available_times[k] < 600.0, 1:num_demands)
    @info "demand windows" n=num_demands n_near=n_near n_far=(num_demands - n_near) ready_span=(minimum(available_times), maximum(available_times)) deadline_span=(minimum(demand_deadlines), maximum(demand_deadlines))

    demands = Dict{String, Any}(
        "sat_identifiers"  => sat_identifiers,
        "demand_deadlines" => demand_deadlines,
        "available_times"  => available_times,
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


function _generate_legacy_demands(pool, num_demands, time_dist, svc_range, disttype, rng)
    sat_identifiers   = Vector{String}(undef, num_demands)
    demand_deadlines  = Vector{Float64}(undef, num_demands)
    available_times   = zeros(Float64, num_demands)
    service_times_out = Vector{Float64}(undef, num_demands)

    dl_lo = Float64(time_dist[1])
    dl_hi = Float64(time_dist[2])
    dl_hi < dl_lo && (dl_hi = dl_lo + 1.0)

    for k in 1:num_demands
        sat_identifiers[k]    = pool[rand(rng, 1:length(pool))]
        demand_deadlines[k]   = sample_from_dist(disttype, dl_lo, dl_hi, 1, rng)[1]
        service_times_out[k]  = sample_from_dist(disttype, svc_range[1], svc_range[2], 1, rng)[1]
    end
    return sat_identifiers, demand_deadlines, available_times, service_times_out
end


"""
Yearly arrivals over `horizon_years`. Each year draws n ∈ [n_lo, n_hi] ready
epochs uniformly in that year; the contract window is near (6–18 months) or
far (2–3 years). Deadline = ready + window.
"""
function _generate_epoch_demands(pool, demand_params, svc_range, disttype, rng)
    horizon_years = Int(demand_params["horizon_years"])
    dpy      = Int.(demand_params["demands_per_year"])          # [n_lo, n_hi]
    near_w   = Float64.(get(demand_params, "near_window_days", [180.0, 548.0]))
    far_w    = Float64.(get(demand_params, "far_window_days",  [730.0, 1095.0]))
    far_frac = Float64(get(demand_params, "far_frac", 0.5))
    year_len = 365.25

    sat_identifiers   = String[]
    demand_deadlines  = Float64[]
    available_times   = Float64[]
    service_times_out = Float64[]

    n_lo, n_hi = extrema(dpy)
    for y in 0:horizon_years-1
        n_y = rand(rng, n_lo:n_hi)
        t0  = y * year_len
        t1  = t0 + year_len
        for _ in 1:n_y
            ready  = t0 + (t1 - t0) * rand(rng)
            is_far = rand(rng) < far_frac
            wlo, whi = is_far ? (far_w[1], far_w[2]) : (near_w[1], near_w[2])
            window = sample_from_dist(disttype, wlo, whi, 1, rng)[1]
            push!(sat_identifiers,   pool[rand(rng, 1:length(pool))])
            push!(available_times,   ready)
            push!(demand_deadlines,  ready + window)
            push!(service_times_out, sample_from_dist(disttype, svc_range[1], svc_range[2], 1, rng)[1])
        end
    end
    return sat_identifiers, demand_deadlines, available_times, service_times_out
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
    n = length(demands["UIDs"])
    !haskey(demands, "available_times") &&
        (demands["available_times"] = zeros(Float64, n))
    return demands
end
