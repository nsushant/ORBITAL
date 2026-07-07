# starlink/fetch_and_sample.jl
# Fetch active Starlink TLEs from Celestrak (JSON, no auth required),
# sample 200 satellites spread across all orbital shells, and place a depot
# at 53°/550 km at the median RAAN of the primary shell.
#
# Returns:
#   sats         :: Vector{Sat}          — 200 target sats + depot_1 (last entry)
#   launch_dates :: Vector{Date}         — one per target sat (same order)
#   sat_values   :: Dict{String,Float64} — sat_name → asset replacement value [$]

using HTTP, JSON3, Dates, Statistics, Random
include(joinpath(@__DIR__, "..", "sim", "orbital_mechanics.jl"))

const CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=JSON"
const TARGET_SATS    = 200
const SATS_PER_PLANE = 2

# ── Satellite asset values ────────────────────────────────────────────────────
# Manufacturing: $1,000/kg (mass-production rate)
# Launch: $1,850/kg internal SpaceX rate (0.5 × Falcon 9 market rate $3,700/kg)
# V1.5: $1,000/kg × 303 kg mfg + $1,850/kg × 303 kg launch = $864,150
# V2 Mini: $1,000/kg × 800 kg mfg + $1,850/kg × 800 kg launch = $2,280,000
const INT_LAUNCH_RATE = 1_850.0   # $/kg — internal SpaceX rate
const V1_MASS_KG      = 303.0     # kg — V1.5 DAS-filed mass
const V2_MASS_KG      = 800.0     # kg — V2 Mini DAS-filed mass
const V1_VALUE = (1_000.0 + INT_LAUNCH_RATE) * V1_MASS_KG   # $864,150
const V2_VALUE = (1_000.0 + INT_LAUNCH_RATE) * V2_MASS_KG   # $2,280,000

# ── Shell definitions (inclination °, altitude km) ────────────────────────────
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
    n = n_rev_per_day * 2π / 86400.0
    return (MU_SIM / n^2)^(1/3)
end

function _parse_launch_date(s)
    isnothing(s) || s == "" && return nothing
    try; return Date(string(s), "yyyy-mm-dd"); catch; return nothing; end
end

# ── V1/V2 classification from OBJECT_ID ──────────────────────────────────────
# OBJECT_ID format: "YYYY-NNNX" e.g. "2023-029B"
# V2 mini: year >= 2024, OR (year == 2023 AND launch_number >= 29)

function _classify_sat(object_id)
    s = string(object_id)
    length(s) < 8 && return :v1
    try
        year       = parse(Int, s[1:4])
        launch_num = parse(Int, s[6:8])
        year >= 2024 && return :v2
        year == 2023 && launch_num >= 29 && return :v2
    catch
    end
    return :v1
end

# ── Shell assignment ──────────────────────────────────────────────────────────

function _assign_shell(inc_deg, alt_km)
    for (k, sh) in enumerate(SHELLS)
        abs(inc_deg - sh.inc) <= INC_TOL && abs(alt_km - sh.alt) <= ALT_TOL && return k
    end
    return 0
end

# ── Main ──────────────────────────────────────────────────────────────────────

function fetch_and_sample(; n_targets=TARGET_SATS, seed=42)
    catalog = fetch_starlink_catalog()
    @info "Catalog size" n=length(catalog)

    shell_planes = [Dict{Int, Vector}() for _ in SHELLS]

    n_unrecognised = 0
    for obj in catalog
        inc  = Float64(obj.INCLINATION)
        mm   = Float64(obj.MEAN_MOTION)
        sma  = _mean_motion_to_sma(mm)
        alt  = sma - Re_SIM
        sh   = _assign_shell(inc, alt)
        sh == 0 && (n_unrecognised += 1; continue)

        raan      = Float64(obj.RA_OF_ASC_NODE)
        plane_bin = floor(Int, raan / 5.0)
        planes    = shell_planes[sh]
        push!(get!(planes, plane_bin, []), obj)
    end
    @info "Shell assignment" unrecognised=n_unrecognised

    rng     = MersenneTwister(seed)
    sampled = []   # (obj, shell_idx)

    # Count catalog sats per shell (proxy for deployment size)
    shell_counts = [sum(length(v) for v in values(shell_planes[sh]); init=0)
                    for sh in eachindex(SHELLS)]
    total_assigned = sum(shell_counts)

    if total_assigned == 0 || n_targets >= total_assigned
        # No trimming needed — take 2-per-plane across all shells
        for (sh_idx, planes) in enumerate(shell_planes)
            for (_, entries) in collect(planes)
                take   = min(SATS_PER_PLANE, length(entries))
                chosen = shuffle!(rng, copy(entries))[1:take]
                for obj in chosen; push!(sampled, (obj, sh_idx)); end
            end
        end
    else
        # Proportional shell allocation: each shell gets slots ∝ its catalog count
        shell_alloc = round.(Int, n_targets .* shell_counts ./ total_assigned)
        # Fix rounding so allocations sum to exactly n_targets
        diff = n_targets - sum(shell_alloc)
        if diff != 0
            order = sortperm(shell_counts; rev=true)
            for i in 1:abs(diff)
                shell_alloc[order[i]] += sign(diff)
            end
        end
        @info "Shell allocation" alloc=collect(zip([s.inc for s in SHELLS], shell_alloc))

        for (sh_idx, planes) in enumerate(shell_planes)
            alloc  = shell_alloc[sh_idx]
            alloc == 0 && continue
            # Shuffle the RAAN planes so we don't always pick the same ones
            plane_list = shuffle!(rng, collect(planes))
            n_taken = 0
            for (_, entries) in plane_list
                n_taken >= alloc && break
                take   = min(SATS_PER_PLANE, length(entries), alloc - n_taken)
                chosen = shuffle!(rng, copy(entries))[1:take]
                for obj in chosen; push!(sampled, (obj, sh_idx)); end
                n_taken += take
            end
        end
    end
    @info "Sampled" n_sampled=length(sampled) n_target=n_targets

    sats         = Sat[]
    launch_dates = Date[]
    sat_values   = Dict{String,Float64}()

    primary_shell_raans = Float64[]

    n_v1 = 0; n_v2 = 0

    for (k, (obj, sh_idx)) in enumerate(sampled)
        inc  = deg2rad(Float64(obj.INCLINATION))
        raan = deg2rad(Float64(obj.RA_OF_ASC_NODE))
        ecc  = Float64(obj.ECCENTRICITY)
        aop  = deg2rad(Float64(obj.ARG_OF_PERICENTER))
        ma   = deg2rad(Float64(obj.MEAN_ANOMALY))
        mm   = Float64(obj.MEAN_MOTION)
        sma  = _mean_motion_to_sma(mm)

        nu = ma   # good to <0.1° for Starlink e<0.001

        r, v = eci_from_oelem(OrbElem(sma, ecc, inc, raan, aop, nu))
        sat_name = "sat_$k"
        push!(sats, Sat(sat_name, r, v))

        ld = _parse_launch_date(get(obj, :LAUNCH_DATE, nothing))
        push!(launch_dates, something(ld, Date(2020, 1, 1)))

        # V1/V2 classification and asset value
        obj_id   = get(obj, :OBJECT_ID, "")
        version  = _classify_sat(obj_id)
        val      = version == :v2 ? V2_VALUE : V1_VALUE
        sat_values[sat_name] = val
        version == :v2 ? (n_v2 += 1) : (n_v1 += 1)

        sh_idx == 1 && push!(primary_shell_raans, deg2rad(Float64(obj.RA_OF_ASC_NODE)))
    end

    @info "Satellite classification" n_v1=n_v1 n_v2=n_v2 v1_value=V1_VALUE v2_value=V2_VALUE

    # Depot at 53°/550 km, median RAAN of primary shell sats
    depot_raan = isempty(primary_shell_raans) ? 0.0 : median(primary_shell_raans)
    depot_oe   = OrbElem(Re_SIM + 550.0, 0.0, deg2rad(53.0), depot_raan, 0.0, 0.0)
    r_d, v_d   = eci_from_oelem(depot_oe)
    push!(sats, Sat("depot_1", r_d, v_d))

    @info "Depot placed" inc_deg=53.0 alt_km=550.0 raan_deg=round(rad2deg(depot_raan), digits=1)
    @info "Total nodes" sats=length(sats) targets=length(launch_dates)

    # Save sat_values and launch_dates for Python/Julia consumers
    mkpath(joinpath(@__DIR__, "..", "outputs"))
    open(joinpath(@__DIR__, "..", "outputs", "sat_values.json"), "w") do f
        JSON3.write(f, sat_values)
    end
    @info "Saved sat_values.json" path="outputs/sat_values.json"

    launch_dates_dict = Dict(sats[i].name => string(launch_dates[i])
                             for i in eachindex(launch_dates))
    open(joinpath(@__DIR__, "..", "outputs", "sat_launch_dates.json"), "w") do f
        JSON3.write(f, launch_dates_dict)
    end
    @info "Saved sat_launch_dates.json" path="outputs/sat_launch_dates.json"

    return sats, launch_dates, sat_values
end
