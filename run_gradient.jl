include("plots/gradients.jl")
include("sim/propagator.jl")
include("fuel_cost_calc/gen_cost_table.jl")

sim = load_sim()
cost_table = load_cost_table()

fig = plot_gradient("depot_1", "sat_400", cost_table, sim)
save("gradient_depot1_sat400.png", fig)
@info "Saved gradient_depot1_sat400.png"
