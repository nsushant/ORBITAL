# build_edelbaum_cost_table.jl
# Builds a mixed-fleet cost table with the Edelbaum oracle.
# Does not touch outputs/cost_table.jld2 (Lu production table).
#
# Run: julia --project=. -t auto build_edelbaum_cost_table.jl

# Tagged paths when THRUST_N is set via ENV (see run_depot_thrust_maps.sh).
# Default 10 mN still writes the untagged Edelbaum production names unless
# COST_TABLE_PATH is set.

include("servicer_params.jl")
include("sim/propagator.jl")
include("fuel_cost_calc/gen_cost_table.jl")

const CT_PATH     = get(ENV, "COST_TABLE_PATH", "outputs/cost_table_edelbaum.jld2")
const PT_PATH     = get(ENV, "PHASING_TABLE_PATH", "outputs/phasing_time_table_edelbaum.jld2")
const MINTOF_PATH = get(ENV, "MIN_TOF_PATH", "outputs/min_tof_table_edelbaum.jld2")
const TOF_STEP    = 15.0    # must match snap_cost grid
const T_END       = 400.0   # mixed-fleet sim coverage

sim = load_sim()
@info "Building Edelbaum cost table" n_nodes=length(sim.names) tof_step=TOF_STEP t_end=T_END Isp=ISP_S thrust_mN=THRUST_N*1e3 T=T_FORCE

mkpath("outputs")
t0 = time()
gen_cost_table(sim; tof_step=TOF_STEP, t_end=T_END, oracle=:edelbaum,
               ct_path=CT_PATH, pt_path=PT_PATH, prompt=false,
               Isp=ISP_S, T=T_FORCE)
@info "Cost table done" elapsed_min=round((time()-t0)/60; digits=1)

@info "Building min-TOF table …"
build_min_tof_table(; ct_path=CT_PATH, pt_path=PT_PATH, out_path=MINTOF_PATH, force=true)
@info "Wrote" CT_PATH PT_PATH MINTOF_PATH
