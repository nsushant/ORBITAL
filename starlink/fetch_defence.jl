# starlink/fetch_defence.jl
# Expand cost/Defence_constellation.json into circular LEO sats + replacement values.
# Full GAO T0–T2 architecture (447 sats). T2 Transport Alpha Low and High are
# both at 81°. Replacement value = manufacturing + launch.
# RAAN lists are modelling assumptions (evenly spaced planes), not operational
# SDA RAANs.

using JSON3

if !@isdefined(Sat)
    include(joinpath(@__DIR__, "..", "sim", "orbital_mechanics.jl"))
end

const DEFENCE_JSON = joinpath(@__DIR__, "..", "cost", "Defence_constellation.json")
const RE_DEF       = 6371.0

function _n_units(layer)
    haskey(layer, "units") && return Int(layer["units"])
    haskey(layer, "units_current") && return Int(layer["units_current"])
    return Int(layer["units_original"])
end

function _mfg_usd(layer)
    if haskey(layer, "manufacturing_cost_usd_million")
        return Float64(layer["manufacturing_cost_usd_million"]) * 1e6
    end
    if haskey(layer, "unit_cost_usd_million")
        return Float64(layer["unit_cost_usd_million"]) * 1e6
    end
    prices = Float64[Float64(v) for v in values(layer["unit_price_usd_million"])]
    return sum(prices) / max(length(prices), 1) * 1e6
end

function _launch_usd(layer)
    haskey(layer, "launch_cost_usd_million") || return 0.0
    return Float64(layer["launch_cost_usd_million"]) * 1e6
end

"""Replacement value = manufacturing + allocated launch cost."""
_unit_usd(layer) = _mfg_usd(layer) + _launch_usd(layer)

function _raans(orbit, key="raan_deg")
    return Float64[Float64(v) for v in orbit[key]]
end

function _split_n(n::Int, n_planes::Int)
    n_planes <= 0 && error("n_planes must be > 0")
    base = div(n, n_planes)
    r    = n - base * n_planes
    return [base + (i <= r ? 1 : 0) for i in 1:n_planes]
end

function _add_planes!(sats, values, types, clients;
                      n::Int, a_km::Float64, inc_deg::Float64, raans::Vector{Float64},
                      type::String, client::String, price::Float64)
    n <= 0 && return
    counts = _split_n(n, length(raans))
    inc = deg2rad(inc_deg)
    for (p, raan_deg) in enumerate(raans)
        np = counts[p]
        np == 0 && continue
        Ω = deg2rad(raan_deg)
        for j in 1:np
            nu = 2π * (j - 1) / np
            r, v = eci_from_oelem(OrbElem(a_km, 0.0, inc, Ω, 0.0, nu))
            name = "sat_$(length(sats) + 1)"
            push!(sats, Sat(name, r, v))
            values[name]  = price
            types[name]   = type
            clients[name] = client
        end
    end
end

function _add_simple!(sats, values, types, clients, layer, type::String, client::String)
    orbit = layer["orbit"]
    _add_planes!(sats, values, types, clients;
                 n= _n_units(layer),
                 a_km=Float64(orbit["semi_major_axis_km"]),
                 inc_deg=Float64(orbit["inclination_deg"]),
                 raans=_raans(orbit),
                 type=type, client=client, price=_unit_usd(layer))
end

"""
    fetch_defence_constellation(; json_path, client) → sats, values, types, clients

`sats` has no depot. `client` is the BCR payer id (default `"sda"`).
Satellite values are manufacturing + launch (replacement cost).
"""
function fetch_defence_constellation(; json_path::String=DEFENCE_JSON, client::String="sda")
    raw = JSON3.read(read(json_path, String))
    sats    = Sat[]
    values  = Dict{String,Float64}()
    types   = Dict{String,String}()
    clients = Dict{String,String}()

    for (tranche, role) in (("T0", "tracking"), ("T0", "transport"),
                            ("T1", "tracking"), ("T1", "transport"),
                            ("T2", "tracking"))
        _add_simple!(sats, values, types, clients, raw[tranche][role],
                     "$(tranche)_$(role)", client)
    end

    α = raw["T2"]["transport_alpha"]
    a_α = Float64(α["orbit"]["semi_major_axis_km"])
    p_α = _unit_usd(α)
    inc_low  = Float64(α["orbit"]["inclination_deg"]["low"])
    inc_high = Float64(α["orbit"]["inclination_deg"]["high"])
    _add_planes!(sats, values, types, clients;
                 n=4 * 19, a_km=a_α, inc_deg=inc_low,
                 raans=_raans(α["orbit"]["raan_deg"], "low_inclination"),
                 type="T2_transport_alpha_low", client=client, price=p_α)
    _add_planes!(sats, values, types, clients;
                 n=6 * 4, a_km=a_α, inc_deg=inc_high,
                 raans=_raans(α["orbit"]["raan_deg"], "high_inclination"),
                 type="T2_transport_alpha_high", client=client, price=p_α)

    β = raw["T2"]["transport_beta"]
    _add_simple!(sats, values, types, clients, β, "T2_transport_beta", client)

    γ = raw["T2"]["transport_gamma"]
    _add_simple!(sats, values, types, clients, γ, "T2_transport_gamma", client)

    return sats, values, types, clients
end
