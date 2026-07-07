# starlink/fetch_planet_labs.jl
# Fetch active Planet Labs TLEs from Celestrak, classify as Dove/SuperDove or SkySat,
# assign lognormal asset values, apply Weibull depreciation by launch age, and
# sample ~n_targets satellites using the same shell-stratified strategy as fetch_and_sample.jl.
#
# Returns:
#   sats         :: Vector{Sat}           — sampled Planet Labs sats (no depot)
#   launch_dates :: Vector{Date}          — one per sat
#   sat_values   :: Dict{String,Float64}  — sat_name → depreciated asset value [$]
#   sat_types    :: Dict{String,Symbol}   — sat_name → :dove or :skysat

using HTTP, JSON3, Dates, Statistics, Random
include(joinpath(@__DIR__, "..", "sim", "orbital_mechanics.jl"))

const PLANET_URL   = "https://celestrak.org/NORAD/elements/gp.php?GROUP=planet&FORMAT=JSON"
const PLANET_CACHE = joinpath(@__DIR__, "planet_catalog.json")
const PLANET_TARGET_SATS    = 124   # use entire catalog — no subsampling
const PLANET_SATS_PER_PLANE = 200   # no effective cap

# ── Shell definitions (Planet Labs SSO shells) ────────────────────────────────
const PLANET_SHELLS = [
    (inc=97.4, alt=490.0),
    (inc=97.6, alt=475.0),
    (inc=98.0, alt=510.0),
    (inc=98.2, alt=520.0),
]
const PLANET_INC_TOL = 1.0   # degrees
const PLANET_ALT_TOL = 40.0  # km

# ── Asset values (fixed replacement cost, pre-depreciation) ──────────────────
# Dove/SuperDove (~5 kg): $200K flat (Planet Labs mass-production scale)
# SkySat (~110 kg):       $3M flat (high-res precision imager)
const DOVE_VALUE   = 200_000.0    # $200K
const SKYSAT_VALUE = 3_000_000.0  # $3M

# ── Weibull depreciation (imported from sol_utils.jl if included, else define locally) ──
if !@isdefined(WEIBULL_LAMBDA)
    const _WL = 5.0; const _WK = 1.5
    _wdep(v, a) = a <= 0.0 ? v : v * exp(-(a / _WL)^_WK)
else
    _wdep(v, a) = weibull_depreciate(v, a)
end

const PLANET_SIM_START = Date(2024, 1, 1)   # reference date for age computation

# ── Helpers ───────────────────────────────────────────────────────────────────

function _fetch_planet_catalog()
    if isfile(PLANET_CACHE) && filesize(PLANET_CACHE) > 5_000
        @info "Loading cached Planet Labs catalog from $PLANET_CACHE"
        return JSON3.read(read(PLANET_CACHE))
    end
    @info "Fetching Planet Labs catalog from Celestrak …"
    resp = HTTP.get(PLANET_URL; readtimeout=60)
    write(PLANET_CACHE, resp.body)
    @info "Planet catalog cached to $PLANET_CACHE"
    return JSON3.read(resp.body)
end

function _planet_assign_shell(inc_deg, alt_km)
    for (k, sh) in enumerate(PLANET_SHELLS)
        abs(inc_deg - sh.inc) <= PLANET_INC_TOL &&
        abs(alt_km  - sh.alt) <= PLANET_ALT_TOL && return k
    end
    return 0
end

function _classify_planet_sat(obj_name)
    name = uppercase(string(obj_name))
    occursin("SKYSAT", name) && return :skysat
    return :dove
end

function _planet_parse_launch_date(s)
    isnothing(s) || s == "" && return nothing
    try; return Date(string(s), "yyyy-mm-dd"); catch; return nothing; end
end

# ── Main fetch function ───────────────────────────────────────────────────────

function fetch_and_sample_planet(; n_targets=PLANET_TARGET_SATS, seed=42,
                                   sat_offset::Int=0)
    catalog = _fetch_planet_catalog()
    @info "Planet catalog size" n=length(catalog)

    # Planet Labs sats span 350–610 km altitude — too wide for fixed shells.
    # All are SSO (96°–100°). Use RAAN-stratified sampling across all SSO sats.
    sso_inc_min = 96.0
    sso_inc_max = 100.0

    raan_planes = Dict{Int, Vector}()
    n_unrecognised = 0

    for obj in catalog
        inc = Float64(obj.INCLINATION)
        (sso_inc_min <= inc <= sso_inc_max) || (n_unrecognised += 1; continue)
        raan      = Float64(obj.RA_OF_ASC_NODE)
        plane_bin = floor(Int, raan / 5.0)
        push!(get!(raan_planes, plane_bin, []), obj)
    end
    @info "Planet SSO sats" total=sum(length(v) for v in values(raan_planes)) n_excluded=n_unrecognised

    rng     = MersenneTwister(seed)
    sampled = []

    for (_, entries) in collect(raan_planes)
        take   = min(PLANET_SATS_PER_PLANE, length(entries))
        chosen = shuffle!(rng, copy(entries))[1:take]
        for obj in chosen; push!(sampled, (obj, 1)); end   # shell_idx=1 (unused)
    end

    if length(sampled) > n_targets
        shuffle!(rng, sampled)
        sampled = sampled[1:n_targets]
    end
    @info "Planet sampled" n_sampled=length(sampled) n_target=n_targets

    sats         = Sat[]
    launch_dates = Date[]
    sat_values   = Dict{String,Float64}()
    sat_types    = Dict{String,Symbol}()

    n_dove = 0; n_skysat = 0

    for (k, (obj, sh_idx)) in enumerate(sampled)
        inc  = deg2rad(Float64(obj.INCLINATION))
        raan = deg2rad(Float64(obj.RA_OF_ASC_NODE))
        ecc  = Float64(obj.ECCENTRICITY)
        aop  = deg2rad(Float64(obj.ARG_OF_PERICENTER))
        ma   = deg2rad(Float64(obj.MEAN_ANOMALY))
        mm   = Float64(obj.MEAN_MOTION)
        sma  = (398600.4418 / (mm * 2π / 86400)^2)^(1/3)

        nu = ma   # small eccentricity approximation

        r, v = eci_from_oelem(OrbElem(sma, ecc, inc, raan, aop, nu))
        sat_name = "sat_$(sat_offset + k)"
        push!(sats, Sat(sat_name, r, v))

        # Launch date and age
        ld  = _planet_parse_launch_date(get(obj, :LAUNCH_DATE, nothing))
        ld  = something(ld, Date(2021, 1, 1))   # fallback: assume ~3-yr-old sat
        push!(launch_dates, ld)
        age_years = Dates.value(PLANET_SIM_START - ld) / 365.25

        # Satellite type and fixed base value
        obj_name = get(obj, :OBJECT_NAME, "")
        sat_type = _classify_planet_sat(obj_name)
        base_val = sat_type == :skysat ? SKYSAT_VALUE : DOVE_VALUE

        # Apply Weibull depreciation by launch age
        dep_val = _wdep(base_val, age_years)

        sat_values[sat_name] = dep_val
        sat_types[sat_name]  = sat_type
        sat_type == :skysat ? (n_skysat += 1) : (n_dove += 1)
    end

    @info "Planet Labs classification" n_dove=n_dove n_skysat=n_skysat

    return sats, launch_dates, sat_values, sat_types
end
