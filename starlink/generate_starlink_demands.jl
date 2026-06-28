# starlink/generate_starlink_demands.jl
# Generate age-weighted service demands for the Starlink case study.
#
# Older satellites are weighted more heavily (higher probability of needing service).
# Service times ~ Uniform(1, 5) days.
# Deadlines ~ Uniform(50, 365) days.

using Dates, Random, JLD2, StatsBase

"""
    generate_starlink_demands(sat_names, launch_dates; n_demands, seed) → Dict

# Arguments
- `sat_names`     — names of target satellites (same order as launch_dates)
- `launch_dates`  — Date of each satellite's launch
- `n_demands`     — number of demands to generate (default 200)
- `seed`          — RNG seed

Deadline is derived from satellite age: older satellites have shorter deadlines
(more urgent). Deadline = max(30, 365 - age_days), capped at 365 days.
Service time is constant at 3.0 days so n_unassigned = f2 / 3.0 exactly.

# Returns
Dict with keys: "sat_identifiers", "demand_deadlines", "service_times", "UIDs"
Saved to outputs/demands/starlink_demands.jld2
"""
function generate_starlink_demands(sat_names::Vector{String}, launch_dates::Vector{Date};
                                   n_demands::Int=200, seed::Int=42) :: Dict{String,Any}

    rng    = MersenneTwister(seed)
    today_ = Dates.today()

    ages    = max.(1, Dates.value.(today_ .- launch_dates))   # days, floor at 1
    weights = Weights(ages ./ sum(ages))

    # Deadline per satellite: older → more urgent → shorter deadline
    # Map age to [30, 365] days inverted: newest sat gets deadline=365, oldest gets 30
    min_age, max_age = minimum(ages), maximum(ages)
    age_range = max(max_age - min_age, 1)
    sat_deadline = Dict(sat_names[i] =>
        clamp(365.0 - 335.0 * (ages[i] - min_age) / age_range, 30.0, 365.0)
        for i in eachindex(sat_names))

    sat_ids   = Vector{String}(undef, n_demands)
    deadlines = Vector{Float64}(undef, n_demands)
    svc_times = Vector{Float64}(undef, n_demands)

    for k in 1:n_demands
        sat_ids[k]   = sample(rng, sat_names, weights)
        deadlines[k] = sat_deadline[sat_ids[k]]
        svc_times[k] = 3.0   # constant — n_unassigned = f2 / 3.0 exactly
    end

    demands = Dict{String,Any}(
        "sat_identifiers" => sat_ids,
        "demand_deadlines" => deadlines,
        "service_times"    => svc_times,
        "UIDs"             => collect(1:n_demands),
    )

    mkpath("outputs/demands")
    @save "outputs/demands/starlink_demands.jld2" demands
    @info "Starlink demands saved" n=n_demands max_deadline=maximum(deadlines)

    return demands
end
