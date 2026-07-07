# starlink/run_mixed_fleet.jl
# Builds a mixed Starlink + Planet Labs simulation and cost table.
# Samples 100 Starlink sats + 100 Planet Labs sats, places a depot in SSO,
# propagates orbits, and generates outputs/cost_table_mixed.jld2.
#
# Run: julia --project=. -t auto starlink/run_mixed_fleet.jl
# Outputs:
#   outputs/simulation_mixed.h5
#   outputs/cost_table_mixed.jld2
#   outputs/mixed_sat_values.json
#   outputs/mixed_launch_dates.json

const NUMEXP_INCLUDE  = true
const GATESTS_INCLUDE = true
include(joinpath(@__DIR__, "..", "algoMDLS.jl"))
include(joinpath(@__DIR__, "fetch_and_sample.jl"))
include(joinpath(@__DIR__, "fetch_planet_labs.jl"))

using JLD2, JSON3, Dates

const MIXED_STARLINK_COUNT = 100
const MIXED_PLANET_COUNT   = PLANET_TARGET_SATS   # 124 — full catalog, no subsampling
const MIXED_SEED           = 42
const SIM_START_DATE       = Date(2024, 1, 1)

# ── Step 1: Fetch Starlink satellites (100, no depot) ─────────────────────────
@info "Fetching Starlink satellites …"
sl_sats, sl_dates, sl_values = fetch_and_sample(n_targets=MIXED_STARLINK_COUNT, seed=MIXED_SEED)
# Remove depot appended by fetch_and_sample
sl_sats_only = filter(s -> !startswith(s.name, "depot"), sl_sats)
@info "Starlink sats sampled" n=length(sl_sats_only)

# Apply Weibull depreciation to Starlink values using actual launch dates
sl_dates_nonempty = sl_dates   # one per sat, same order as sl_sats_only
sl_values_dep = Dict{String,Float64}()
for (i, sat) in enumerate(sl_sats_only)
    age = Dates.value(SIM_START_DATE - sl_dates_nonempty[i]) / 365.25
    sl_values_dep[sat.name] = weibull_depreciate(sl_values[sat.name], age)
end

# ── Step 2: Fetch Planet Labs satellites (100, offset naming to avoid collision) ──
@info "Fetching Planet Labs satellites …"
pl_offset = length(sl_sats_only)
pl_sats, pl_dates, pl_values, pl_types = fetch_and_sample_planet(
    n_targets=MIXED_PLANET_COUNT, seed=MIXED_SEED, sat_offset=pl_offset)
@info "Planet Labs sats sampled" n=length(pl_sats)

# ── Step 3: Place depot in SSO (co-planar with Planet Labs + Starlink SSO shell) ──
# Depot at 97.6°, 560 km (Starlink SSO shell altitude) — median RAAN of Starlink SSO sats
starlink_sso_sats = filter(s -> !startswith(s.name, "depot"), sl_sats_only)
depot_raan = 0.0   # default; ideally median RAAN of SSO-shell Starlink sats
depot_oe   = OrbElem(6371.0 + 560.0, 0.0, deg2rad(97.6), depot_raan, 0.0, 0.0)
r_d, v_d   = eci_from_oelem(depot_oe)
depot_sat  = Sat("depot_1", r_d, v_d)

all_sats = vcat(sl_sats_only, pl_sats, [depot_sat])
@info "Total simulation nodes" n=length(all_sats) starlink=length(sl_sats_only) planet=length(pl_sats) depot=1

# ── Step 4: Build simulation HDF5 ─────────────────────────────────────────────
sim_params = Dict(
    "J2"    => true,
    "dt"    => 60.0,
    "t_end" => 400.0 * 86400.0,   # 400-day cost table (modular wrapping for 5-yr missions)
)

@info "Building mixed-fleet simulation …"
sim_mixed = gen_simulation_from_sats(all_sats, sim_params)
@info "Simulation built" n_nodes=length(sim_mixed.names)
# Note: saved to outputs/simulation.h5 (default path)

# ── Step 5: Cost table built separately (incremental extension) ─────────────────
# Run after this script completes:
#   julia --project=. -t auto starlink/extend_cost_table.jl

# ── Step 6: Save combined sat_values and launch_dates ─────────────────────────
mixed_sat_values   = merge(sl_values_dep, pl_values)
mixed_launch_dates = Dict{String,String}()
for (i, sat) in enumerate(sl_sats_only)
    mixed_launch_dates[sat.name] = string(sl_dates_nonempty[i])
end
for (i, sat) in enumerate(pl_sats)
    mixed_launch_dates[sat.name] = string(pl_dates[i])
end

mkpath("outputs")
open("outputs/mixed_sat_values.json", "w") do f
    JSON3.write(f, mixed_sat_values)
end
open("outputs/mixed_launch_dates.json", "w") do f
    JSON3.write(f, mixed_launch_dates)
end
open("outputs/mixed_sat_types.json", "w") do f
    # :dove/:skysat for Planet Labs; :v1/:v2 for Starlink
    types_str = Dict(k => string(v) for (k, v) in pl_types)
    JSON3.write(f, types_str)
end

@info "Mixed-fleet setup complete"
println()
println("═" ^ 60)
println("  Outputs:")
println("    outputs/simulation.h5       (mixed-fleet simulation)")
println("    outputs/mixed_sat_values.json")
println("    outputs/mixed_launch_dates.json")
println("    outputs/mixed_sat_types.json")
println()
println("Next:")
println("  julia --project=. -t auto starlink/extend_cost_table.jl")
println("    outputs/mixed_sat_values.json")
println("    outputs/mixed_launch_dates.json")
println("    outputs/mixed_sat_types.json")
println("═" ^ 60)
println()
println("Next:")
println("  julia --project=. generate_experiment_demands.jl")
println("  julia --project=. generate_sensitivity_demands.jl")
