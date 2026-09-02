# sweep_depot_location.jl
# For each depot (a, i) on a grid: rebuild Edelbaum depot↔sat legs on the existing
# sat↔sat table, run MDLS, write the Pareto front. Does not overwrite Lu or
# Edelbaum production tables.
#
# Run: julia --project=. -t auto sweep_depot_location.jl
# Requires: outputs/cost_table_edelbaum.jld2 and bcr_mixed_7000 demands.

const GATESTS_INCLUDE = true
include("algoMDLS.jl")
include("servicer_params.jl")
include("bcr_clients.jl")

using JLD2
using HDF5
using Printf
using ProgressMeter

# ── Grid (edit here) ──────────────────────────────────────────────────────────
# 50 km SMA × 1° inclination on 76–84°. Map bilinear-fills 4-corner quads.
# 17 altitudes × 9 inclinations = 153 locations.
const RE_KM        = 6371.0
const ALT_KM       = collect(400.0:50.0:1200.0)      # 17 altitudes
const INC_DEG      = collect(76.0:1.0:84.0)          # 9 inclinations
const RAAN0        = 0.0
const TOF_STEP     = 15.0
const T_END        = 400.0
const MDLS_ITERS   = 100
const N_VEHICLES   = 100
const REFUEL_TIME  = 0.5
const ORACLE       = :edelbaum
const SCENARIO     = "bcr_mixed_$(Int(DV_BUDGET))"
const TRIAL        = 1

const CT_PATH  = get(ENV, "COST_TABLE_PATH", "outputs/cost_table_edelbaum.jld2")
const PT_PATH  = get(ENV, "PHASING_TABLE_PATH", "outputs/phasing_time_table_edelbaum.jld2")
const DEM_PATH = "outputs/long_horizon_demands/$(SCENARIO)_01.jld2"
const H5_OUT   = get(ENV, "DEPOT_H5", "outputs/depot_location_sweep.h5")
const RES_DIR  = get(ENV, "DEPOT_RES_DIR", "outputs/depot_location_sweep")

# ── Synthetic circular + J2-RAAN depot OE ─────────────────────────────────────
function synthesize_depot_oe(times::Vector{Float64}, a::Float64, inc::Float64;
                             raan0::Float64=0.0, nu0::Float64=0.0)
    n  = length(times)
    oe = Matrix{Float64}(undef, 4, n)
    Ωdot = raan_drift_rate(a, inc)
    n_m  = sqrt(MU_LT / a^3)
    @inbounds for k in 1:n
        t = times[k]
        oe[1, k] = a
        oe[2, k] = inc
        oe[3, k] = raan0 + Ωdot * t
        oe[4, k] = mod(nu0 + n_m * t, 2π)
    end
    return oe
end

function split_sat_depot_tables(CostTable, PhasingTimeTable, depot_set::Set{Int})
    sat_mintof = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    sat_min_dv_pairs = Dict{Tuple{Int,Int}, Float64}()
    n_nodes = 0
    for ((from, to, dep, arr), dv) in CostTable
        n_nodes = max(n_nodes, from, to)
        (from in depot_set || to in depot_set) && continue
        Tp = get(PhasingTimeTable, (from, to, dep, arr), 0.0)
        true_tof = (arr - dep) + Tp
        key = (from, to)
        if dv < 1e7 && (!haskey(sat_mintof, key) || true_tof < sat_mintof[key][1])
            sat_mintof[key] = (true_tof, dv)
        end
        if dv < get(sat_min_dv_pairs, key, Inf)
            sat_min_dv_pairs[key] = dv
        end
    end
    sat_min_dv = fill(Inf, n_nodes, n_nodes)
    for ((i, j), dv) in sat_min_dv_pairs
        sat_min_dv[i, j] = dv
    end
    return sat_mintof, sat_min_dv, n_nodes
end

function patch_depot_mintof(sat_mintof, depot_ct, depot_pt)
    mintof = copy(sat_mintof)
    for ((from, to, dep, arr), dv) in depot_ct
        dv >= 1e7 && continue
        Tp       = get(depot_pt, (from, to, dep, arr), 0.0)
        true_tof = (arr - dep) + Tp
        key      = (from, to)
        if !haskey(mintof, key) || true_tof < mintof[key][1]
            mintof[key] = (true_tof, dv)
        end
    end
    return mintof
end

function patch_min_dv(sat_min_dv, depot_ct)
    tab = copy(sat_min_dv)
    for ((i, j, _, _), dv) in depot_ct
        dv < tab[i, j] && (tab[i, j] = dv)
    end
    return tab
end

function loc_group_name(a_km, inc_deg)
    replace(@sprintf("a%.0f_i%.2f", a_km, inc_deg), "." => "p")
end

# ── Load once ─────────────────────────────────────────────────────────────────
isfile(CT_PATH)  || error("Missing $CT_PATH — run bash run_bcr_edelbaum.sh first")
isfile(DEM_PATH) || error("Missing $DEM_PATH — generate bcr_mixed demands first")

mkpath(RES_DIR)
sim = load_sim()
depots = findall(startswith("depot"), sim.names)
sats   = findall(startswith("sat"),   sim.names)
isempty(depots) && error("No depot node in simulation.h5")
depot_idx = only(depots)
depot_set = Set(depots)
plane_groups = get_plane_groups(sim, sats)

@info "Loading Edelbaum cost table (sat↔sat reused) …"
CostTable        = load(CT_PATH, "CostTable")
PhasingTimeTable = isfile(PT_PATH) ? load(PT_PATH, "PhasingTimeTable") :
                   Dict{Tuple{Int,Int,Float64,Float64},Float64}()
@info "Tables loaded" n=length(CostTable)

@info "Caching sat↔sat min-TOF / min-ΔV (skip depot keys) …"
sat_mintof, sat_min_dv, n_nodes = split_sat_depot_tables(CostTable, PhasingTimeTable, depot_set)

arrivals   = collect(TOF_STEP:TOF_STEP:T_END)
departures = collect(TOF_STEP:TOF_STEP:T_END)
depot_to_sat_keys = Tuple{Int,Int,Float64,Float64}[
    (depot_idx, sat, dep, dep + arr)
    for arr in arrivals for dep in departures for sat in sats
]
sat_to_depot_keys = Tuple{Int,Int,Float64,Float64}[
    (sat, depot_idx, dep, dep + arr)
    for arr in arrivals for dep in departures for sat in sats
]

@info "Loading satellite OE …"
sat_oe = load_oe_data(sim, sats)

@info "Loading demands …"
demands = load(DEM_PATH, "demands")
demands["UIDs"] = collect(1:length(demands["sat_identifiers"]))
total_demand_value = haskey(demands, "asset_values") ? sum(demands["asset_values"]) : 0.0
sat_clients = load_sat_clients(sim=sim)
client_ids  = ordered_client_ids(sat_clients)
tdv_k       = demand_client_tdv(demands, sat_clients)

grid = vec([(RE_KM + alt, inc) for alt in ALT_KM, inc in INC_DEG])
n_loc = length(grid)
@info "Depot (a, i) sweep" n_alt=length(ALT_KM) n_inc=length(INC_DEG) n_loc=n_loc oracle=ORACLE thrust_mN=THRUST_N*1e3 tag=THRUST_TAG

# ── Sweep ─────────────────────────────────────────────────────────────────────
p = Progress(n_loc; desc="  depot locations:   ", barlen=40, showspeed=true)
t_all = time()

h5open(H5_OUT, "w") do fid
    create_group(fid, "loc")
end

for (k, (a_km, inc_deg)) in enumerate(grid)
    inc = deg2rad(inc_deg)
    oe_data = copy(sat_oe)
    oe_data[depot_idx] = synthesize_depot_oe(sim.times, a_km, inc; raan0=RAAN0)

    CT_d2s, PT_d2s = build_cost_table_depot_to_sat_fast(
        sim, depot_to_sat_keys, plane_groups;
        oracle=ORACLE, Isp=ISP_S, T=T_FORCE, oe_data=oe_data, verbose=false)
    CT_s2d, PT_s2d = build_cost_table_sat_to_depot_fast(
        sim, sat_to_depot_keys, plane_groups;
        oracle=ORACLE, Isp=ISP_S, T=T_FORCE, oe_data=oe_data, verbose=false)

    merge!(CostTable,        CT_d2s, CT_s2d)
    merge!(PhasingTimeTable, PT_d2s, PT_s2d)
    depot_ct = merge(CT_d2s, CT_s2d)
    depot_pt = merge(PT_d2s, PT_s2d)

    mintof = patch_depot_mintof(sat_mintof, depot_ct, depot_pt)
    min_dv = patch_min_dv(sat_min_dv, depot_ct)

    init_sol, init_unas = make_init_schedule(demands, sim;
        nvehicles=N_VEHICLES, refuel_time=REFUEL_TIME, dv_budget=DV_BUDGET,
        min_tof_table=mintof)
    archive = MDLS(MDLS_ITERS, demands, sim, CostTable, mintof, min_dv;
                   nvehicles=N_VEHICLES, dv_budget=DV_BUDGET,
                   init_sol=init_sol, init_unassigned=init_unas)

    data, _ = client_front_matrix(archive, sat_clients, client_ids)

    loc = loc_group_name(a_km, inc_deg)
    csv_path = joinpath(RES_DIR, "mdls_$(loc).csv")
    open(csv_path, "w") do io
        println(io, client_column_names(client_ids))
        for j in 1:size(data, 2)
            print(io, join((round(data[r, j]; digits=6) for r in 1:size(data, 1)), ","))
            println(io)
        end
    end

    if size(data, 2) > 0
        h5open(H5_OUT, "r+") do fid
            locroot = fid["loc"]
            haskey(locroot, loc) && delete_object(locroot, loc)
            g = create_group(locroot, loc)
            attrs(g)["a_km"]     = a_km
            attrs(g)["inc_deg"]  = inc_deg
            attrs(g)["alt_km"]   = a_km - RE_KM
            attrs(g)["thrust_N"] = THRUST_N
            attrs(g)["Isp_s"]    = ISP_S
            mdls = create_group(g, "mdls")
            scen = create_group(mdls, SCENARIO)
            scen["trial_01"] = data
            ds = scen["trial_01"]
            attrs(ds)["columns"]            = client_column_names(client_ids)
            attrs(ds)["total_demand_value"] = total_demand_value
            attrs(ds)["clients"]            = join(client_ids, ",")
            attrs(ds)["fee_model"]          = "implied_contract"
            attrs(ds)["a_km"]               = a_km
            attrs(ds)["inc_deg"]            = inc_deg
            for c in client_ids
                attrs(ds)["total_demand_value_$(c)"] = get(tdv_k, c, 0.0)
            end
        end
    end

    next!(p)
end

@info "Sweep done" elapsed_min=round((time() - t_all) / 60; digits=1) h5=H5_OUT
println("Plot: python plot_depot_bcr_map.py --h5 $H5_OUT")
