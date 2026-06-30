# generate_sensitivity_demands.jl
# Phase 1 of the sensitivity analysis pipeline.
# Generates demand JLD2 files + greedy JSON seeds for all 4 sub-experiments × levels × trials.
# Output: outputs/sensitivity_demands/{se}_{level}_{trial:02d}.jld2
#         outputs/sensitivity_demands/{se}_{level}_{trial:02d}_greedy.json
#
# Run: julia --project=. generate_sensitivity_demands.jl

const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2
using Random

const N_TRIALS    = 5
const OUT_DIR     = joinpath(@__DIR__, "outputs", "sensitivity_demands")
const REFUEL_TIME = 0.5   # days — must match GA (run_ga_trial.py)

# ── Sub-experiment definitions ────────────────────────────────────────────────
# Each entry: (se_name, levels, base_params, seed_fn)
# seed_fn(trial, level_label) → Int seed for generate_demands

const SE1_LEVELS = [10, 50, 100, 150, 200]
const SE2_LEVELS = ["normal", "uniform"]
const SE3_LEVELS = [3000, 5000, 8000, 12000]
const SE4_LEVELS = [1500.0, 3000.0, 5000.0, 8000.0]

const DEFAULT_ASSET_VALUE = 1_251_000.0   # V1 Starlink replacement value [USD]
const V1_VALUE    = 1_251_000.0           # $/sat
const V2_VALUE    = 3_250_000.0           # $/sat
const V2_FRACTION = 0.30                  # ~30% of current Starlink fleet is V2-mini

subexperiments = [
    # SE1 — instance size
    (name      = "size",
     levels    = [string(v) for v in SE1_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"        => parse(Int, lv),
                               "type"               => "random",
                               "disttype"           => "uniform",
                               "deltaV_dist"        => 8000.0,
                               "time_dist"          => [50.0, 365.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => min(100, parse(Int, lv)),
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137 + parse(Int, lv),
     greedy_dv_fn   = (lv) -> 5000.0),

    # SE2 — distribution type
    (name      = "disttype",
     levels    = SE2_LEVELS,
     base_fn   = (lv) -> Dict("num_demands"        => 200,
                               "type"               => "random",
                               "disttype"           => lv,
                               "deltaV_dist"        => 8000.0,
                               "time_dist"          => [50.0, 365.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137 + Int(abs(hash(lv)) % 1000),
     greedy_dv_fn   = (lv) -> 5000.0),

    # SE3 — ΔV threshold
    (name      = "dv",
     levels    = [string(v) for v in SE3_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"        => 200,
                               "type"               => "random",
                               "disttype"           => "uniform",
                               "deltaV_dist"        => Float64(parse(Int, lv)),
                               "time_dist"          => [50.0, 365.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137 + parse(Int, lv) ÷ 100,
     greedy_dv_fn   = (lv) -> 5000.0),

    # SE4 — vehicle ΔV budget: greedy warm start uses the actual budget level
    (name      = "dvbudget",
     levels    = [string(Int(v)) for v in SE4_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"        => 200,
                               "type"               => "random",
                               "disttype"           => "uniform",
                               "deltaV_dist"        => 8000.0,
                               "time_dist"          => [50.0, 365.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137,
     greedy_dv_fn   = (lv) -> parse(Float64, lv)),   # uses actual budget for warm start
]

mkpath(OUT_DIR)
sim = load_sim()

# Optional filter: julia generate_sensitivity_demands.jl [se_name ...]
# e.g. julia generate_sensitivity_demands.jl dvbudget
#      julia generate_sensitivity_demands.jl size disttype
# No args → run all sub-experiments
filter_names = isempty(ARGS) ? nothing : Set(ARGS)
active_ses   = filter_names === nothing ? subexperiments :
               filter(se -> se.name in filter_names, subexperiments)
isempty(active_ses) && error("No sub-experiments matched: $(ARGS). Valid names: $(join([s.name for s in subexperiments], ", "))")

total = sum(length(se.levels) * N_TRIALS for se in active_ses)
done  = Ref(0)

for se in active_ses
    @info "─── Sub-experiment: $(se.name) ───"
    for lv in se.levels
        for trial in 1:N_TRIALS
            seed      = se.seed_fn(trial, lv)

            # Randomly assign V1/V2 asset values to all simulation satellites for this trial
            rng_sat       = MersenneTwister(seed)
            sat_names_all = filter(n -> startswith(n, "sat"), sim.names)
            sat_values_trial = Dict{String,Float64}(
                n => (rand(rng_sat) < V2_FRACTION ? V2_VALUE : V1_VALUE)
                for n in sat_names_all
            )

            params    = merge(se.base_fn(lv), Dict("seed" => seed, "sat_values" => sat_values_trial))
            trial_str = lpad(trial, 2, '0')
            key       = "$(se.name)_$(lv)"

            @info "  Generating" key=key trial=trial seed=seed
            demands = generate_demands(sim, params)

            # Save demand JLD2 (Python-readable via loaders.load_demands)
            dem_path = joinpath(OUT_DIR, "$(key)_$(trial_str).jld2")
            @save dem_path demands
            @info "  Saved demands" path=dem_path n=length(demands["UIDs"])

            # Greedy warm-start (uses budget-appropriate dv_budget for SE4)
            greedy_dv = se.greedy_dv_fn(lv)
            init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=20,
                                                     refuel_time=REFUEL_TIME,
                                                     dv_budget=greedy_dv)
            greedy_path = joinpath(OUT_DIR, "$(key)_$(trial_str)_greedy.json")
            save_schedule_json(init_sol, init_unas, greedy_path)

            done[] += 1
            @info "  Progress" done=done[] total=total
        end
    end
end

@info "Done. Generated $(done[]) demand+greedy pairs." OUT_DIR
