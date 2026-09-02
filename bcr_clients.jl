# Per-client demand tagging for implied-contract BCR.
# Map file: outputs/sat_clients.json  (sat_name → client_id).
# Override path with ENV SAT_CLIENTS_PATH. A later defence instance replaces the JSON.

using JSON3

const SAT_CLIENTS_PATH = get(ENV, "SAT_CLIENTS_PATH", "outputs/sat_clients.json")
const _CLIENT_PREF     = ["starlink", "planet"]

function default_client_id(sat_name::AbstractString; n_starlink::Int=100)
    m = match(r"^sat_(\d+)$", sat_name)
    m === nothing && return "unknown"
    return parse(Int, m.captures[1]) <= n_starlink ? "starlink" : "planet"
end

function load_sat_clients(path::AbstractString=SAT_CLIENTS_PATH;
                          n_starlink::Int=100, sim=nothing)
    if isfile(path)
        raw = JSON3.read(read(path, String))
        return Dict{String,String}(string(k) => string(v) for (k, v) in pairs(raw))
    end
    sim === nothing && return Dict{String,String}()
    d = Dict{String,String}()
    for name in sim.names
        startswith(name, "sat_") || continue
        d[name] = default_client_id(name; n_starlink=n_starlink)
    end
    return d
end

function write_sat_clients(d::Dict{String,String}, path::AbstractString=SAT_CLIENTS_PATH)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, d)
    end
    return path
end

function ordered_client_ids(sat_clients::Dict{String,String})
    ids = unique(collect(values(sat_clients)))
    ordered = [c for c in _CLIENT_PREF if c in ids]
    append!(ordered, sort([c for c in ids if !(c in _CLIENT_PREF)]))
    return ordered
end

function demand_client_tdv(demands, sat_clients::Dict{String,String})
    ids  = demands["sat_identifiers"]
    vals = Float64.(demands["asset_values"])
    tdv  = Dict{String,Float64}()
    for (sat, v) in zip(ids, vals)
        c = get(sat_clients, sat, "unknown")
        tdv[c] = get(tdv, c, 0.0) + v
    end
    return tdv
end

function client_unserved(unas, sat_clients::Dict{String,String}, client_ids)
    out = Dict(c => 0.0 for c in client_ids)
    unas === nothing && return out
    ids  = unas["sat_identifiers"]
    vals = haskey(unas, "asset_values") ? Float64.(unas["asset_values"]) :
           zeros(length(ids))
    for (sat, v) in zip(ids, vals)
        c = get(sat_clients, sat, "unknown")
        haskey(out, c) && (out[c] += v)
    end
    return out
end

"""Front matrix (n_cols × n_sol): f1, f2_total, f3, f2_<client>…  Valid solutions only."""
function client_front_matrix(archive, sat_clients, client_ids)
    valid = [i for i in eachindex(archive.solutions)
             if archive.total_deltaV[i] < INFEASIBLE_LEG_COST]
    n = length(valid)
    k = 3 + length(client_ids)
    data = Matrix{Float64}(undef, k, n)
    for (j, i) in enumerate(valid)
        data[1, j] = archive.total_deltaV[i]
        data[2, j] = archive.total_serv_time_unassigned[i]
        data[3, j] = Float64(archive.total_vehicles_used[i])
        uk = client_unserved(archive.unassigned_sets[i], sat_clients, client_ids)
        for (cidx, c) in enumerate(client_ids)
            data[3 + cidx, j] = uk[c]
        end
    end
    return data, valid
end

function client_column_names(client_ids)
    return "f1_dv,f2_unrecovered_value,f3_vehicles," *
           join(["f2_$(c)" for c in client_ids], ",")
end
