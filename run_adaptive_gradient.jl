include("plots/gradients.jl")
include("sim/propagator.jl")
include("fuel_cost_calc/gen_cost_table.jl")
include("fuel_cost_calc/adaptive_grid.jl")

sim = load_sim()
base_ct = load_cost_table()
ag = load_adaptive_cost_table()

fig = plot_refinement_comparison("depot_1", "sat_400", base_ct, ag, sim)
save("adaptive_gradient_comparison_depot_1_sat_400.png", fig)
@info "Saved adaptive_gradient_comparison_depot_1_sat_400.png"

fig2 = plot_grid_comparison("depot_1", "sat_400", base_ct, ag, sim)
save("adaptive_grid_structure_depot_1_sat_400.png", fig2)
@info "Saved adaptive_grid_structure_depot_1_sat_400.png"
