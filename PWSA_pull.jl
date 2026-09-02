# PWSA_pull.jl
# Download current CelesTrak GP elements for launched T0 and T1 PWSA satellites
# and write them to pwsa_t0_t1_orbital_elements.csv.
#
# Run: julia --project=. PWSA_pull.jl
#
# CelesTrak CSV OMMs do not include SEMIMAJOR_AXIS; it is derived from mean
# motion. USSF-124 also carried two MDA HBTSS satellites — those are dropped.

using Pkg
Pkg.activate(@__DIR__; io=devnull)

using HTTP
using CSV
using DataFrames

const GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
const OUT_CSV = joinpath(@__DIR__, "pwsa_t0_t1_orbital_elements.csv")

# WGS-84 / SGP4 conventional constants (km, km³/s²)
const MU_KM3_S2 = 398600.4418
const RE_KM     = 6378.137

const HTTP_HEADERS = [
    "User-Agent" => "basic_project PWSA_pull.jl (research; T0/T1 orbital elements)",
    "Accept"     => "text/csv",
]

# Known COSPAR designators for deployed PWSA T0 / T1 missions.
# T1 Tracking has not launched as of this catalog; T1TL-A/D/F are not yet flown.
# T1TL-B/C/E each carried 21 T1 Transport satellites.
const LAUNCHES = [
    ("T0_launch_1", "2023-050", "T0"),
    ("T0_launch_2", "2023-133", "T0"),
    ("T0_USSF124",  "2024-028", "T0"),
    ("T1TL_B",      "2025-203", "T1"),
    ("T1TL_C",      "2025-230", "T1"),
    ("T1TL_E",      "2026-163", "T1"),
]

const WANTED = [
    "MISSION",
    "TRANCHE",
    "INTDES",
    "OBJECT_NAME",
    "OBJECT_ID",
    "NORAD_CAT_ID",
    "EPOCH",
    "SEMIMAJOR_AXIS",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "MEAN_MOTION",
    "APOAPSIS",
    "PERIAPSIS",
]

function get_gp_by_launch(intdes::AbstractString; retries::Int=8)
    url = "$(GP_URL)?INTDES=$(intdes)&FORMAT=CSV"
    last_err = nothing
    for attempt in 1:retries
        r = try
            HTTP.get(url; headers=HTTP_HEADERS, status_exception=false,
                     readtimeout=60, connecttimeout=30)
        catch err
            last_err = err
            wait = min(90.0, 8.0 * 2^(attempt - 1))
            @warn "Request error for $(intdes) (attempt $(attempt)/$(retries)); retrying in $(round(Int, wait))s" exception=err
            sleep(wait)
            continue
        end
        if r.status == 200
            body = String(r.body)
            startswith(strip(body), "OBJECT_NAME") || error(
                "CelesTrak returned a non-CSV body for $(intdes) (empty launch or HTML error)."
            )
            return CSV.read(IOBuffer(body), DataFrame)
        end
        last_err = ErrorException("CelesTrak query failed for $(intdes): HTTP $(r.status)")
        r.status in (429, 503) || throw(last_err)
        wait = min(90.0, 8.0 * 2^(attempt - 1))
        @warn "HTTP $(r.status) for $(intdes) (attempt $(attempt)/$(retries)); retrying in $(round(Int, wait))s"
        sleep(wait)
    end
    throw(last_err)
end

function sma_from_n(n_rev_per_day)
    n = Float64(n_rev_per_day)
    (n > 0 && isfinite(n)) || return missing
    n_rad_s = n * 2π / 86400.0
    return (MU_KM3_S2 / n_rad_s^2)^(1 / 3)
end

function is_pwsa_payload(name)::Bool
    n = uppercase(string(name))
    occursin("HBTSS", n) && return false
    occursin("R/B", n) && return false
    occursin(r"\bDEB\b", n) && return false
    occursin("FALCON", n) && return false
    return true
end

function main()
    frames = DataFrame[]

    for (mission, intdes, tranche) in LAUNCHES
        println("Downloading $(mission) ($(intdes))...")
        df = try
            get_gp_by_launch(intdes)
        catch err
            @warn "Skipping $(mission)" exception=err
            sleep(5.0)
            continue
        end

        if nrow(df) == 0
            @warn "No GP rows for $(mission) ($(intdes))"
            sleep(1.0)
            continue
        end

        if "OBJECT_NAME" in names(df)
            n_before = nrow(df)
            df = filter(:OBJECT_NAME => is_pwsa_payload, df)
            n_drop = n_before - nrow(df)
            n_drop > 0 && println("  dropped $(n_drop) non-PWSA object(s) (R/B, debris, HBTSS)")
        end

        df[!, :MISSION] .= mission
        df[!, :TRANCHE] .= tranche
        df[!, :INTDES]  .= intdes

        if "MEAN_MOTION" in names(df)
            df[!, :SEMIMAJOR_AXIS] = sma_from_n.(df.MEAN_MOTION)
            if "ECCENTRICITY" in names(df)
                e = df.ECCENTRICITY
                a = df.SEMIMAJOR_AXIS
                df[!, :APOAPSIS]  = [ismissing(ai) ? missing : ai * (1 + ei) - RE_KM
                                     for (ai, ei) in zip(a, e)]
                df[!, :PERIAPSIS] = [ismissing(ai) ? missing : ai * (1 - ei) - RE_KM
                                     for (ai, ei) in zip(a, e)]
            end
        end

        println("  kept $(nrow(df)) payload(s)")
        push!(frames, df)
        sleep(5.0)  # CelesTrak courtesy delay
    end

    isempty(frames) && error("No CelesTrak rows downloaded.")

    all_data = reduce((a, b) -> vcat(a, b; cols=:union), frames)
    cols = [c for c in WANTED if c in names(all_data)]
    pwsadata = all_data[:, cols]

    sort_cols = [c for c in ["TRANCHE", "MISSION", "RA_OF_ASC_NODE"] if c in names(pwsadata)]
    isempty(sort_cols) || sort!(pwsadata, sort_cols)

    CSV.write(OUT_CSV, pwsadata)

    println()
    println("Saved $(nrow(pwsadata)) catalog objects to $(OUT_CSV)")
    if "TRANCHE" in names(pwsadata)
        for g in groupby(pwsadata, :TRANCHE)
            println("  $(g.TRANCHE[1]): $(nrow(g))")
        end
    end
    println()
    show(pwsadata; allrows=true, allcols=true)
    println()
end

main()
