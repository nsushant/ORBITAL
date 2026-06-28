# generate_experiment_demands.jl
# Phase 1 of the numerical experiment pipeline.
# Generates demand JLD2 files and greedy JSON seeds for 4 scenarios × N_TRIALS.
# Output: outputs/exp_demands/{scenario}_{trial:02d}.jld2
#         outputs/exp_demands/{scenario}_{trial:02d}_greedy.json
#
# Run: julia --project=. generate_experiment_demands.jl

const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2

const N_DEMANDS = 200
const N_TRIALS  = 10
const OUT_DIR   = joinpath(@__DIR__, "outputs", "exp_demands")

const INSTANCES = [
    (name   = "tight_normal",
     params = Dict("num_demands"   => N_DEMANDS,
                   "type"          => "random",
                   "disttype"      => "normal",
                   "deltaV_dist"   => 8000.0,
                   "time_dist"     => [10.0, 100.0],
                   "service_times" => [1.0, 3.0])),

    (name   = "loose_uniform",
     params = Dict("num_demands"   => N_DEMANDS,
                   "type"          => "random",
                   "disttype"      => "uniform",
                   "deltaV_dist"   => 8000.0,
                   "time_dist"     => [50.0, 365.0],
                   "service_times" => [1.0, 5.0])),

    (name   = "tight_low_dv",
     params = Dict("num_demands"   => N_DEMANDS,
                   "type"          => "random",
                   "disttype"      => "normal",
                   "deltaV_dist"   => 5000.0,
                   "time_dist"     => [10.0, 200.0],
                   "service_times" => [1.0, 4.0])),

    (name   = "loose_high_dv",
     params = Dict("num_demands"   => N_DEMANDS,
                   "type"          => "random",
                   "disttype"      => "uniform",
                   "deltaV_dist"   => 12000.0,
                   "time_dist"     => [100.0, 365.0],
                   "service_times" => [1.0, 5.0])),
]

mkpath(OUT_DIR)
sim = load_sim()

for inst in INSTANCES
    @info "─── Scenario: $(inst.name) ───"
    for trial in 1:N_TRIALS
        seed      = trial * 137 + Int(hash(inst.name) % 1000)
        params    = merge(inst.params, Dict("seed" => seed))
        trial_str = lpad(trial, 2, '0')

        @info "  Generating demands" trial=trial seed=seed
        demands = generate_demands(sim, params)

        # Save demands JLD2 — readable by Python loaders.load_demands()
        dem_path = joinpath(OUT_DIR, "$(inst.name)_$(trial_str).jld2")
        @save dem_path demands
        @info "  Saved demands" path=dem_path n=length(demands["UIDs"])

        # Greedy warm-start (Python GAs load this via load_greedy_from_json)
        @info "  Building greedy schedule …"
        init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=20)
        greedy_path = joinpath(OUT_DIR, "$(inst.name)_$(trial_str)_greedy.json")
        save_schedule_json(init_sol, init_unas, greedy_path)
        @info "  Saved greedy" path=greedy_path n_vehicles=length(init_sol)
    end
end

@info "Done. Generated $(length(INSTANCES) * N_TRIALS) demand+greedy pairs." OUT_DIR
