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

const N_TRIALS = 10
const OUT_DIR  = joinpath(@__DIR__, "outputs", "sensitivity_demands")

# ── Sub-experiment definitions ────────────────────────────────────────────────
# Each entry: (se_name, levels, base_params, seed_fn)
# seed_fn(trial, level_label) → Int seed for generate_demands

const SE1_LEVELS = [10, 50, 100, 150, 200]
const SE2_LEVELS = ["normal", "uniform"]
const SE3_LEVELS = [3000, 5000, 8000, 12000]
const SE4_LEVELS = [1500.0, 3000.0, 5000.0, 8000.0]

subexperiments = [
    # SE1 — instance size
    (name      = "size",
     levels    = [string(v) for v in SE1_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"   => parse(Int, lv),
                               "type"          => "random",
                               "disttype"      => "uniform",
                               "deltaV_dist"   => 8000.0,
                               "time_dist"     => [50.0, 365.0],
                               "service_times" => [1.0, 5.0],
                               "num_satellites"=> min(100, parse(Int, lv))),
     seed_fn   = (trial, lv) -> trial * 137 + parse(Int, lv)),

    # SE2 — distribution type
    (name      = "disttype",
     levels    = SE2_LEVELS,
     base_fn   = (lv) -> Dict("num_demands"   => 200,
                               "type"          => "random",
                               "disttype"      => lv,
                               "deltaV_dist"   => 8000.0,
                               "time_dist"     => [50.0, 365.0],
                               "service_times" => [1.0, 5.0],
                               "num_satellites"=> 100),
     seed_fn   = (trial, lv) -> trial * 137 + Int(abs(hash(lv)) % 1000)),

    # SE3 — ΔV threshold
    (name      = "dv",
     levels    = [string(v) for v in SE3_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"   => 200,
                               "type"          => "random",
                               "disttype"      => "uniform",
                               "deltaV_dist"   => Float64(parse(Int, lv)),
                               "time_dist"     => [50.0, 365.0],
                               "service_times" => [1.0, 5.0],
                               "num_satellites"=> 100),
     seed_fn   = (trial, lv) -> trial * 137 + parse(Int, lv) ÷ 100),

    # SE4 — vehicle ΔV budget (demand scenario is fixed; dv_budget only affects MDLS at run time)
    (name      = "dvbudget",
     levels    = [string(Int(v)) for v in SE4_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"   => 200,
                               "type"          => "random",
                               "disttype"      => "uniform",
                               "deltaV_dist"   => 8000.0,
                               "time_dist"     => [50.0, 365.0],
                               "service_times" => [1.0, 5.0],
                               "num_satellites"=> 100),
     seed_fn   = (trial, lv) -> trial * 137),
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
            params    = merge(se.base_fn(lv), Dict("seed" => seed))
            trial_str = lpad(trial, 2, '0')
            key       = "$(se.name)_$(lv)"

            @info "  Generating" key=key trial=trial seed=seed
            demands = generate_demands(sim, params)

            # Save demand JLD2 (Python-readable via loaders.load_demands)
            dem_path = joinpath(OUT_DIR, "$(key)_$(trial_str).jld2")
            @save dem_path demands
            @info "  Saved demands" path=dem_path n=length(demands["UIDs"])

            # Greedy warm-start
            init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=20)
            greedy_path = joinpath(OUT_DIR, "$(key)_$(trial_str)_greedy.json")
            save_schedule_json(init_sol, init_unas, greedy_path)

            done[] += 1
            @info "  Progress" done=done[] total=total
        end
    end
end

@info "Done. Generated $(done[]) demand+greedy pairs." OUT_DIR
