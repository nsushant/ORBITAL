# run_defence_fleet.jl
# Replace the mixed Starlink/Planet instance with the SDA PWSA catalog
# in cost/Defence_constellation.json (GAO T0–T2, 447 sats; T2 Alpha Low and
# High both at 81°). Replacement value = manufacturing + launch. One BCR
# client ("sda"). Depot at 81°, 1000 km, RAAN=0.
#
# Does not touch outputs/cost_table.jld2 (Lu). Overwrites outputs/simulation.h5.
#
# Run: julia --project=. -t auto run_defence_fleet.jl
# Then: bash run_bcr_edelbaum.sh     # Edelbaum table + bcr_mixed demands + MDLS

const NUMEXP_INCLUDE  = true
const GATESTS_INCLUDE = true
include("algoMDLS.jl")
include("starlink/fetch_defence.jl")

using JSON3

const DEPOT_ALT_KM  = 1000.0
const DEPOT_INC_DEG = 81.0
const DEPOT_RAAN    = 0.0

sats, sat_values, sat_types, sat_clients = fetch_defence_constellation()
@info "Defence sats" n=length(sats) catalog_B=round(sum(v for v in Base.values(sat_values))/1e9; digits=3)

depot_oe = OrbElem(RE_DEF + DEPOT_ALT_KM, 0.0, deg2rad(DEPOT_INC_DEG),
                   deg2rad(DEPOT_RAAN), 0.0, 0.0)
r_d, v_d = eci_from_oelem(depot_oe)
all_sats = vcat(sats, [Sat("depot_1", r_d, v_d)])

sim_params = Dict(
    "J2"    => true,
    "dt"    => 60.0,
    "t_end" => 400.0 * 86400.0,
)

@info "Propagating defence simulation (overwrites outputs/simulation.h5) …"
sim = gen_simulation_from_sats(all_sats, sim_params; prompt=false)
@info "Simulation built" n_nodes=length(sim.names)

mkpath("outputs")
open("outputs/mixed_sat_values.json", "w") do f
    JSON3.write(f, sat_values)
end
open("outputs/mixed_sat_types.json", "w") do f
    JSON3.write(f, sat_types)
end
open("outputs/sat_clients.json", "w") do f
    JSON3.write(f, sat_clients)
end

@info "Wrote outputs/mixed_sat_values.json, mixed_sat_types.json, sat_clients.json"
println()
println("Next: bash run_bcr_edelbaum.sh")
println("  (rebuilds Edelbaum table on this sim, then bcr_mixed_7000 MDLS)")
