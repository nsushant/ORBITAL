# starlink/fetch_and_sample.jl
# Fetch active Starlink TLEs from Celestrak (JSON, no auth required),
# sample 200 satellites spread across all orbital shells, and place a depot
# at 53°/550 km at the median RAAN of the primary shell.
#
# Returns:
#   sats         :: Vector{Sat}   — 200 target sats + depot_1 (last entry)
#   launch_dates :: Vector{Date}  — one per target sat (same order)

using HTTP, JSON3, Dates, Statistics, Random
include(joinpath(@__DIR__, "..", "sim", "orbital_mechanics.jl"))

const CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=JSON"
const TARGET_SATS   = 200
const SATS_PER_PLANE = 2

# ── Shell definitions (inclination °, altitude km) ────────────────────────────
# Starlink shells currently operational
const SHELLS = [
    (inc=53.0,  alt=550.0),
    (inc=53.2,  alt=540.0),
    (inc=70.0,  alt=570.0),
    (inc=97.6,  alt=560.0),   # sun-synchronous
    (inc=43.0,  alt=530.0),
]
const INC_TOL = 1.5    # degrees
const ALT_TOL = 30.0   # km

# ── Fetch and parse ───────────────────────────────────────────────────────────

const CATALOG_CACHE = joinpath(@__DIR__, "starlink_catalog.json")

function fetch_starlink_catalog()
    if isfile(CATALOG_CACHE) && filesize(CATALOG_CACHE) > 10_000
        @info "Loading cached Starlink catalog from $CATALOG_CACHE"
        return JSON3.read(read(CATALOG_CACHE))
    end
    @info "Fetching Starlink catalog from Celestrak …"
    resp = HTTP.get(CELESTRAK_URL; readtimeout=60)
    write(CATALOG_CACHE, resp.body)
    @info "Catalog cached to $CATALOG_CACHE"
    return JSON3.read(resp.body)
end

function _mean_motion_to_sma(n_rev_per_day::Float64)
    # n in rad/s → a in km
    n = n_rev_per_day * 2π / 86400.0
    return (MU_SIM / n^2)^(1/3)
end

function _parse_launch_date(s)
    # LAUNCH_DATE field: "YYYY-MM-DD" or missing
    isnothing(s) || s == "" && return nothing
    try; return Date(string(s), "yyyy-mm-dd"); catch; return nothing; end
end

# ── Shell assignment ──────────────────────────────────────────────────────────

function _assign_shell(inc_deg, alt_km)
    for (k, sh) in enumerate(SHELLS)
        abs(inc_deg - sh.inc) <= INC_TOL && abs(alt_km - sh.alt) <= ALT_TOL && return k
    end
    return 0   # unrecognised shell
end

# ── Main ──────────────────────────────────────────────────────────────────────

function fetch_and_sample(; n_targets=TARGET_SATS, seed=42)
    catalog = fetch_starlink_catalog()
    @info "Catalog size" n=length(catalog)

    # Group by shell → by RAAN bin (plane)
    # shell_planes[shell_idx][plane_bin] = [entry, ...]
    shell_planes = [Dict{Int, Vector}() for _ in SHELLS]

    n_unrecognised = 0
    for obj in catalog
        inc  = Float64(obj.INCLINATION)
        mm   = Float64(obj.MEAN_MOTION)        # rev/day
        sma  = _mean_motion_to_sma(mm)
        alt  = sma - Re_SIM                    # km
        sh   = _assign_shell(inc, alt)
        sh == 0 && (n_unrecognised += 1; continue)

        raan      = Float64(obj.RA_OF_ASC_NODE)
        plane_bin = floor(Int, raan / 5.0)     # 5° RAAN bins → ~72 planes per shell
        planes    = shell_planes[sh]
        push!(get!(planes, plane_bin, []), obj)
    end
    @info "Shell assignment" unrecognised=n_unrecognised

    # Sample SATS_PER_PLANE per plane per shell, collect across all shells
    rng     = MersenneTwister(seed)
    sampled = []   # (obj, shell_idx)

    for (sh_idx, planes) in enumerate(shell_planes)
        for (_, entries) in collect(planes)
            take = min(SATS_PER_PLANE, length(entries))
            chosen = shuffle!(rng, entries)[1:take]
            for obj in chosen; push!(sampled, (obj, sh_idx)); end
        end
    end

    # Trim or warn
    if length(sampled) > n_targets
        shuffle!(rng, sampled)
        sampled = sampled[1:n_targets]
    end
    @info "Sampled" n_sampled=length(sampled) n_target=n_targets

    # Convert to Sat structs
    sats         = Sat[]
    launch_dates = Date[]

    primary_shell_raans = Float64[]   # for depot placement

    for (k, (obj, sh_idx)) in enumerate(sampled)
        inc  = deg2rad(Float64(obj.INCLINATION))
        raan = deg2rad(Float64(obj.RA_OF_ASC_NODE))
        ecc  = Float64(obj.ECCENTRICITY)
        aop  = deg2rad(Float64(obj.ARG_OF_PERICENTER))
        ma   = deg2rad(Float64(obj.MEAN_ANOMALY))
        mm   = Float64(obj.MEAN_MOTION)
        sma  = _mean_motion_to_sma(mm)

        # Convert mean anomaly → true anomaly (circular orbit approximation)
        nu = ma   # good to <0.1° for Starlink e<0.001

        r, v = eci_from_oelem(OrbElem(sma, ecc, inc, raan, aop, nu))
        push!(sats, Sat("sat_$k", r, v))

        ld = _parse_launch_date(get(obj, :LAUNCH_DATE, nothing))
        push!(launch_dates, something(ld, Date(2020, 1, 1)))   # fallback

        sh_idx == 1 && push!(primary_shell_raans, deg2rad(Float64(obj.RA_OF_ASC_NODE)))
    end

    # Depot at 53°/550 km, median RAAN of primary shell sats
    depot_raan = isempty(primary_shell_raans) ? 0.0 : median(primary_shell_raans)
    depot_oe   = OrbElem(Re_SIM + 550.0, 0.0, deg2rad(53.0), depot_raan, 0.0, 0.0)
    r_d, v_d   = eci_from_oelem(depot_oe)
    push!(sats, Sat("depot_1", r_d, v_d))

    @info "Depot placed" inc_deg=53.0 alt_km=550.0 raan_deg=round(rad2deg(depot_raan), digits=1)
    @info "Total nodes" sats=length(sats) targets=length(launch_dates)

    return sats, launch_dates
end
