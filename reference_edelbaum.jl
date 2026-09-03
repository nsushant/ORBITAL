# reference_edelbaum.jl — dump the Julia Edelbaum model on a fixed geometry set,
# so the Python port can be gated numerically (docs/edelbaum_port_audit.md).
#
# Run:  julia --project=. reference_edelbaum.jl
# Writes: outputs/edelbaum_reference_julia.csv

include(joinpath(@__DIR__, "cost", "edelbaum_transfer.jl"))

using Printf
using Random

const MASS   = 335.0
const ISP    = 2800.0
const THRUST = 1.0e-4          # 100 mN in kg*km/s^2; enough geometries are feasible
const RE_STD = 6378.137

# Same population and draw as the Python side: read the instance population and
# take ordered pairs from a fixed seed so both stacks see identical geometries.
# Columns are looked up by header name, not by position. They used to be read
# positionally, and when the population gained the full orbital element set the
# script started parsing an object name as a semi-major axis and died with
# `cannot parse "STARLINK-3801" as Float64`. A named lookup fails loudly and
# specifically if the schema moves again.
pop = Tuple{Float64,Float64}[]
open(joinpath(@__DIR__, "outputs", "instance_population.csv")) do io
    header = split(strip(readline(io)), ',')
    col = Dict(strip(h) => k for (k, h) in enumerate(header))
    for c in ("a_km", "incl_deg")
        haskey(col, c) || error("instance_population.csv has no '$c' column; " *
                                "found: " * join(header, ", "))
    end
    ia, ii = col["a_km"], col["incl_deg"]
    for line in eachline(io)
        isempty(strip(line)) && continue
        f = split(line, ',')
        # The depot is a node of the simulation, not a transfer endpoint here.
        haskey(col, "group") && strip(f[col["group"]]) == "depot" && continue
        push!(pop, (parse(Float64, f[ia]), deg2rad(parse(Float64, f[ii]))))
    end
end
@info "population" n=length(pop)

rng = MersenneTwister(20240902)
rows = NTuple{6,Float64}[]
while length(rows) < 300
    i = rand(rng, 1:length(pop)); j = rand(rng, 1:length(pop))
    i == j && continue
    (a0, i0) = pop[i]; (af, iff) = pop[j]
    gap = (rand(rng) * 2 - 1) * pi
    tof = 120.0 + rand(rng) * 245.0
    push!(rows, (a0, i0, af, iff, gap, tof))
end

open(joinpath(@__DIR__, "outputs", "edelbaum_reference_julia.csv"), "w") do io
    println(io, "a0_km,incl0_rad,af_km,inclf_rad,raan_gap_rad,tof_days,dv_m_s")
    for (a0, i0, af, iff, gap, tof) in rows
        # Julia takes the required absolute RAAN change: propagate the target's
        # node to arrival, exactly as the Python harness does.
        raan_arr = gap + (-1.5 * J2_ED * RE_ED^2 * sqrt(MU_ED / af^7) * cos(iff)) * tof * 86400.0
        res = edelbaum_transfer_cost(a0, i0, 0.0, af, iff, raan_arr, tof;
                                     m=MASS, Isp=ISP, T=THRUST)
        dv = res["deltaV_total"]
        @printf(io, "%.6f,%.9f,%.6f,%.9f,%.9f,%.6f,%.6f\n",
                a0, i0, af, iff, gap, tof, dv)
    end
end
@info "wrote outputs/edelbaum_reference_julia.csv" n=length(rows)
