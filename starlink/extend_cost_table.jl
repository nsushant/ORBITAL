# starlink/extend_cost_table.jl
# Builds the mixed-fleet (Starlink + Planet Labs) cost table.
#
# Two modes:
#   FRESH BUILD  — no existing outputs/cost_table.jld2: computes all entries from scratch.
#   INCREMENTAL  — existing SL-only table found: reuses SL↔SL and SL↔depot entries,
#                  only computes new PL-involving entries.
#
# Prerequisite: run_mixed_fleet.jl must have been run first (outputs/simulation.h5).
#
# Run: julia --project=. -t auto starlink/extend_cost_table.jl
# Output: outputs/cost_table.jld2 + outputs/phasing_time_table.jld2

const NUMEXP_INCLUDE  = true
const GATESTS_INCLUDE = true
include(joinpath(@__DIR__, "..", "algoMDLS.jl"))
include(joinpath(@__DIR__, "fetch_and_sample.jl"))
include(joinpath(@__DIR__, "fetch_planet_labs.jl"))

using JLD2

const N_STARLINK = 100
const TOF_STEP   = 15.0
const T_END      = 400.0

# Cost-table fidelity: true → Section 4.3 continuous-thrust NLP, false → 4.2 impulsive.
const CONTINUOUS_THRUST = false
const FMAX_KMS2         = 3.5e-6   # km/s² (paper value; adjust for your servicer)

# ── Step 1: rebuild mixed sim object ─────────────────────────────────────────
@info "Loading mixed-fleet simulation …"
sl_sats, _, _ = fetch_and_sample(n_targets=N_STARLINK, seed=42)
sl_sats_only  = filter(s -> !startswith(s.name, "depot"), sl_sats)
pl_sats, _, _, _ = fetch_and_sample_planet(n_targets=PLANET_TARGET_SATS, seed=42, sat_offset=N_STARLINK)
depot_oe  = OrbElem(6371.0 + 560.0, 0.0, deg2rad(97.6), 0.0, 0.0, 0.0)
r_d, v_d  = eci_from_oelem(depot_oe)
depot_sat = Sat("depot_1", r_d, v_d)
all_sats  = vcat(sl_sats_only, pl_sats, [depot_sat])

sim_params = Dict("J2" => true, "dt" => 60.0, "t_end" => T_END * 86400.0)
sim_mixed  = gen_simulation_from_sats(all_sats, sim_params)
@info "Sim loaded" n_nodes=length(sim_mixed.names)

# ── Step 2: define key sets ───────────────────────────────────────────────────
arrivals   = collect(TOF_STEP:TOF_STEP:T_END)
departures = collect(TOF_STEP:TOF_STEP:T_END)

sats_all = findall(n -> startswith(n, "sat"),   sim_mixed.names)
sats_sl  = filter(i -> i <= N_STARLINK, sats_all)
sats_pl  = filter(i -> i > N_STARLINK,  sats_all)
depots   = findall(n -> startswith(n, "depot"), sim_mixed.names)

NEW_DEPOT_IDX = only(depots)

plane_groups_all = get_plane_groups(sim_mixed, sats_all)
plane_groups_pl  = get_plane_groups(sim_mixed, sats_pl)
@info "Plane groups" n_all=length(plane_groups_all) n_pl=length(plane_groups_pl)

# ── Step 3: choose mode ───────────────────────────────────────────────────────
ct_path = "outputs/cost_table.jld2"
INCREMENTAL = isfile(ct_path)

if INCREMENTAL
    @info "Existing cost table found — incremental mode (reusing SL↔SL entries)"

    local CostTable_old
    @load ct_path CostTable
    CostTable_old = CostTable

    old_all_indices = union(Set(k[1] for k in keys(CostTable_old)),
                            Set(k[2] for k in keys(CostTable_old)))
    OLD_DEPOT_IDX = maximum(old_all_indices)
    @info "Depot index" old=OLD_DEPOT_IDX new=NEW_DEPOT_IDX

    # Translate reusable SL↔SL and SL↔depot entries
    CostTable_base = Dict{Tuple{Int,Int,Float64,Float64}, Float64}()
    for (k, v) in CostTable_old
        fi, ti, dep, arr = k
        is_sl(i)  = i <= N_STARLINK
        is_dep(i) = i == OLD_DEPOT_IDX
        (is_sl(fi) || is_dep(fi)) || continue
        (is_sl(ti) || is_dep(ti)) || continue
        new_fi = is_dep(fi) ? NEW_DEPOT_IDX : fi
        new_ti = is_dep(ti) ? NEW_DEPOT_IDX : ti
        CostTable_base[(new_fi, new_ti, dep, arr)] = v
    end
    @info "Reused cost entries" n=length(CostTable_base)

    # Load and translate existing phasing table
    pt_path = "outputs/phasing_time_table.jld2"
    PT_base = Dict{Tuple{Int,Int,Float64,Float64}, Float64}()
    if isfile(pt_path)
        local PhasingTimeTable
        @load pt_path PhasingTimeTable
        for (k, v) in PhasingTimeTable
            fi, ti, dep, arr = k
            is_sl(i)  = i <= N_STARLINK
            is_dep(i) = i == OLD_DEPOT_IDX
            (is_sl(fi) || is_dep(fi)) || continue
            (is_sl(ti) || is_dep(ti)) || continue
            new_fi = is_dep(fi) ? NEW_DEPOT_IDX : fi
            new_ti = is_dep(ti) ? NEW_DEPOT_IDX : ti
            PT_base[(new_fi, new_ti, dep, arr)] = v
        end
    end
    @info "Reused phasing entries" n=length(PT_base)

    # Keys: only PL-involving entries
    sat_to_sat_keys = Tuple{Int,Int,Float64,Float64}[
        (si, sj, dep, dep + arr)
        for arr in arrivals, dep in departures
        for si in sats_all, sj in sats_all
        if si != sj && (si > N_STARLINK || sj > N_STARLINK)
    ]
    depot_to_sat_keys = Tuple{Int,Int,Float64,Float64}[
        (depot, sat, dep, dep + arr)
        for arr in arrivals, dep in departures
        for sat in sats_pl, depot in depots
    ]
    sat_to_depot_keys = Tuple{Int,Int,Float64,Float64}[
        (sat, depot, dep, dep + arr)
        for arr in arrivals, dep in departures
        for sat in sats_pl, depot in depots
    ]
else
    @info "No existing cost table — fresh build for all $(length(sats_all)) sats + depot"

    CostTable_base = Dict{Tuple{Int,Int,Float64,Float64}, Float64}()
    PT_base        = Dict{Tuple{Int,Int,Float64,Float64}, Float64}()

    sat_to_sat_keys = Tuple{Int,Int,Float64,Float64}[
        (si, sj, dep, dep + arr)
        for arr in arrivals, dep in departures
        for si in sats_all, sj in sats_all
        if si != sj
    ]
    depot_to_sat_keys = Tuple{Int,Int,Float64,Float64}[
        (depot, sat, dep, dep + arr)
        for arr in arrivals, dep in departures
        for sat in sats_all, depot in depots
    ]
    sat_to_depot_keys = Tuple{Int,Int,Float64,Float64}[
        (sat, depot, dep, dep + arr)
        for arr in arrivals, dep in departures
        for sat in sats_all, depot in depots
    ]
end

@info "Key counts" sat_sat=length(sat_to_sat_keys) depot_to_sat=length(depot_to_sat_keys) sat_to_depot=length(sat_to_depot_keys)

# ── Step 4: compute ───────────────────────────────────────────────────────────
@info "Computing sat↔sat entries …"
CT_sat, PT_sat = build_cost_table_fast(sim_mixed, sat_to_sat_keys, plane_groups_all;
                                       continuous=CONTINUOUS_THRUST, fmax=FMAX_KMS2)

@info "Computing depot→sat entries …"
CT_dep_to_sat, PT_dep_to_sat = build_cost_table_depot_to_sat_fast(sim_mixed, depot_to_sat_keys,
                                                                    INCREMENTAL ? plane_groups_pl : plane_groups_all;
                                                                    continuous=CONTINUOUS_THRUST, fmax=FMAX_KMS2)

@info "Computing sat→depot entries …"
CT_sat_to_dep, PT_sat_to_dep = build_cost_table_sat_to_depot_fast(sim_mixed, sat_to_depot_keys,
                                                                    INCREMENTAL ? plane_groups_pl : plane_groups_all;
                                                                    continuous=CONTINUOUS_THRUST, fmax=FMAX_KMS2)

# ── Step 5: merge and save ────────────────────────────────────────────────────
@info "Merging …"
CostTable        = merge(CostTable_base, CT_sat, CT_dep_to_sat, CT_sat_to_dep)
PhasingTimeTable = merge(PT_base, PT_sat, PT_dep_to_sat, PT_sat_to_dep)
@info "Total entries" cost=length(CostTable) phasing=length(PhasingTimeTable)

@save "outputs/cost_table.jld2" CostTable
@save "outputs/phasing_time_table.jld2" PhasingTimeTable
@info "Saved mixed-fleet tables" path="outputs/"

println()
println("═" ^ 60)
println("  Mixed-fleet cost table saved to outputs/cost_table.jld2")
println("  Entries: $(length(CostTable))")
println("  Mode: $(INCREMENTAL ? "incremental" : "fresh build")")
println("═" ^ 60)
println()
println("Next:")
println("  rm outputs/min_tof_table.jld2   # must regenerate from new phasing data")
println("  julia --project=. generate_experiment_demands.jl")
println("  julia --project=. generate_sensitivity_demands.jl")
