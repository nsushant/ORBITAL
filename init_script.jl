using LinearAlgebra

include("sim/propagator.jl")
include("fuel_cost_calc/gen_cost_table.jl")
include("demands/generate_demands.jl")


constellation_params = Dict("type"=>"delta_walker",
                            "num_planes" => 10.0, 
                            "num_satellites"=> 300, 
                            "phasing"=>1.0,
                            "inclination" => 53.0, 
                            "altitude" => 550.0 
                           )

depot_params = Dict( "type" => "depot", 
                     "altitude" => 600.0,
                     "inclination" => 53.0, 
                     "RAAN" => 0.0,
                     "e" => 0.001
                   ) 


seconds_in_5_years = 5.0 * 365.0 * 24.0 * 60.0 * 60.0 
sim_params = Dict( "J2"=>true,  "dt"=> 60, "t_end"=>seconds_in_5_years)



# to propagate single shell 
simulation = gen_simulation([constellation_params,depot_params], sim_params)

# to propagate multiple shells 
# simulation = gen_simulation([constellation_params1, constellation_params2, depot_params1, depot_params2], sim_params)


# now generate pairwise cost tables 
# cost table [i,j,departure_epoch,arrival_time]
# This might take a while, will write a binary file to outputs/ 
# by default it will creta a HT LT hybrid cost table with lowest cost dv in eah cell 
# if you want LT only just supply arg type = "LT"

cost_table = gen_cost_table(simulation)



demand_params= Dict(
                     "num_demands" => IWOULDLIKETHISTOZORKWHYDOESITNOT//???? "100",
                     "num_satellites" => 100, 
                     "sim_object" => simulation,
                     "type" => "random", 
                     "seed" => 42,
                     "disttype" => "normal",
                     "deltaV_dist" => 10000, # max separation of demands in m/s from depot 
                     "time_dist" => [10,365], # time span over which demands will be distributed
                     "service_times" => [1.0, 5.0] # service time range [days]
                   )



demands = generate_demands(simulation, demand_params)


# ── Visualisation ──────────────────────────────────────────────────────────────
include("plots/visualise.jl")

depot_name = first(filter(n -> startswith(n, "depot"), simulation.names))
sat_name   = first(filter(n -> startswith(n, "sat"),   simulation.names))

fig1 = plotdvtable(depot_name, sat_name, cost_table, simulation)
save("outputs/dv_depot_to_sat.png", fig1)

fig2 = plotdemands(demands, cost_table, simulation)
save("outputs/demands.png", fig2)

# you can also generate 
# 1. demand_identifiers, demand_deadlines, service_times = generate_demands(num_demands=64, simulation, type="physical")
#    where demands are generated using the weibull distribution, atmospheric model etc.
# 2. min separation here describes the minimum time separation between demands 
















