# starlink/generate_starlink_demands.jl
# Generate age-weighted service demands for the Starlink case study.
#
# Older satellites are weighted more heavily (higher probability of needing service).
# Service times ~ Uniform(1, 5) days.
# Deadlines ~ Uniform(50, 365) days.

using Dates, Random, JLD2, StatsBase

"""
    generate_starlink_demands(sat_names, launch_dates, sat_values; n_demands, seed) → Dict

# Arguments
- `sat_names`   — names of target satellites (same order as launch_dates)
- `launch_dates` — Date of each satellite's launch
- `sat_values`  — Dict mapping sat_name → asset replacement value [USD]
- `n_demands`   — number of demands to generate (default 200)
- `seed`        — RNG seed

Deadline is derived from satellite age: older satellites have shorter deadlines.
Service time is constant at 3.0 days.
asset_values stores the replacement value of the target satellite for each demand.

# Returns
Dict with keys: "sat_identifiers", "demand_deadlines", "service_times", "asset_values", "UIDs"
Saved to outputs/demands/starlink_demands.jld2
"""
function generate_starlink_demands(sat_names::Vector{String}, launch_dates::Vector{Date},
                                   sat_values::Dict{String,Float64};
                                   n_demands::Int=200, seed::Int=42) :: Dict{String,Any}

    rng    = MersenneTwister(seed)
    today_ = Dates.today()

    ages    = max.(1, Dates.value.(today_ .- launch_dates))   # days, floor at 1
    weights = Weights(ages ./ sum(ages))

    # Deadline per satellite: older → more urgent → shorter deadline
    min_age, max_age = minimum(ages), maximum(ages)
    age_range = max(max_age - min_age, 1)
    sat_deadline = Dict(sat_names[i] =>
        clamp(365.0 - 335.0 * (ages[i] - min_age) / age_range, 30.0, 365.0)
        for i in eachindex(sat_names))

    sat_ids     = Vector{String}(undef, n_demands)
    deadlines   = Vector{Float64}(undef, n_demands)
    svc_times   = Vector{Float64}(undef, n_demands)
    asset_vals  = Vector{Float64}(undef, n_demands)

    default_val = 1_251_000.0   # V1 value as fallback

    for k in 1:n_demands
        sat_ids[k]    = sample(rng, sat_names, weights)
        deadlines[k]  = sat_deadline[sat_ids[k]]
        svc_times[k]  = 3.0
        asset_vals[k] = get(sat_values, sat_ids[k], default_val)
    end

    demands = Dict{String,Any}(
        "sat_identifiers" => sat_ids,
        "demand_deadlines" => deadlines,
        "service_times"    => svc_times,
        "asset_values"     => asset_vals,
        "UIDs"             => collect(1:n_demands),
    )

    mkpath("outputs/demands")
    @save "outputs/demands/starlink_demands.jld2" demands
    @info "Starlink demands saved" n=n_demands max_deadline=maximum(deadlines) total_value=sum(asset_vals)

    return demands
end
