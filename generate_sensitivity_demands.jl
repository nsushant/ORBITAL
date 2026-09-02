# generate_sensitivity_demands.jl
# Phase 1 of the sensitivity analysis pipeline.
# Generates demand JLD2 files + greedy JSON seeds for all 4 sub-experiments × levels × trials.
# Output: outputs/sensitivity_demands/{se}_{level}_{trial:02d}.jld2
#         outputs/sensitivity_demands/{se}_{level}_{trial:02d}_greedy.json
#
# Run: julia --project=. generate_sensitivity_demands.jl

const GATESTS_INCLUDE = true
include("algoMDLS.jl")

using JLD2, JSON3, Dates
using Random
using Statistics

const N_TRIALS    = 1
const REFUEL_TIME = 0.5   # days — must match GA (run_ga_trial.py)

# Optional --outdir=<path> flag (must appear before SE name filters)
_outdir_arg = filter(a -> startswith(a, "--outdir="), ARGS)
const OUT_DIR = isempty(_outdir_arg) ?
    joinpath(@__DIR__, "outputs", "sensitivity_demands") :
    _outdir_arg[1][length("--outdir=")+1:end]

# ── Sub-experiment definitions ────────────────────────────────────────────────
# Each entry: (se_name, levels, base_params, seed_fn)
# seed_fn(trial, level_label) → Int seed for generate_demands

const SE1_LEVELS = [10, 50, 100, 150, 200]
const SE2_LEVELS = ["normal", "uniform"]
const SE3_LEVELS = [3000, 5000, 8000, 12000]
const SE4_LEVELS = [1500.0, 3000.0, 5000.0, 8000.0, 10000.0]
const SE5_LEVELS = [7000.0]

# Per-satellite replacement values from SSCM CER (Foreman et al. 2016) + Wright's law
# (b=0.67, inflation x2.005 FY2000→2026, alpha=0.2 for mature bus)
# V1.5: m_dry=294 kg, N=5000 → $250K/sat
# V2 Mini: m_dry=776 kg, N=2000 → $1.45M/sat
const V1_VALUE    = 250_000.0
const V2_VALUE    = 1_450_000.0
const DEFAULT_ASSET_VALUE = 463_927.0  # number-weighted avg of 224 satellites from mixed_sat_values.json
const V2_FRACTION = 0.30        # ~30% of current Starlink fleet is V2 Mini
const SIM_START_DATE = Date(2024, 1, 1)   # reference epoch for age computation

# Load Starlink launch dates saved by fetch_and_sample
const LAUNCH_DATES_PATH = joinpath(@__DIR__, "outputs", "sat_launch_dates.json")
const _SAT_LAUNCH_DATES = isfile(LAUNCH_DATES_PATH) ?
    JSON3.read(read(LAUNCH_DATES_PATH), Dict{String,String}) : Dict{String,String}()
isempty(_SAT_LAUNCH_DATES) &&
    @warn "sat_launch_dates.json not found — using fallback age of 2.0 years for all sats"

# Load Planet Labs pre-computed values (fixed+Weibull, saved by run_mixed_fleet.jl)
const MIXED_VALUES_PATH = joinpath(@__DIR__, "outputs", "mixed_sat_values.json")
const _MIXED_SAT_VALUES = isfile(MIXED_VALUES_PATH) ?
    JSON3.read(read(MIXED_VALUES_PATH), Dict{String,Float64}) : Dict{String,Float64}()

function depreciated_sat_values(sim, seed)
    rng = MersenneTwister(seed)
    sat_names = filter(n -> startswith(n, "sat"), sim.names)
    Dict{String,Float64}(
        n => if haskey(_MIXED_SAT_VALUES, n)
                _MIXED_SAT_VALUES[n]   # Planet Labs — fixed value + Weibull at fetch time
             else
                base = rand(rng) < V2_FRACTION ? V2_VALUE : V1_VALUE
                ld   = get(_SAT_LAUNCH_DATES, n, nothing)
                age  = isnothing(ld) ? 2.0 :
                       max(0.0, Dates.value(SIM_START_DATE - Date(ld)) / 365.25)
                weibull_depreciate(base, age)  # Starlink — Weibull by actual launch date
             end
        for n in sat_names
    )
end

subexperiments = [
    # SE1 — instance size
    (name      = "size",
     levels    = [string(v) for v in SE1_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"        => parse(Int, lv),
                               "type"               => "random",
                               "disttype"           => "uniform",
                               "deltaV_dist"        => 8000.0,
                               "time_dist"          => [50.0, 1825.0],
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
                               "time_dist"          => [50.0, 1825.0],
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
                               "time_dist"          => [50.0, 1825.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137 + parse(Int, lv) ÷ 100,
     greedy_dv_fn   = (lv) -> 5000.0),

    # SE4 — vehicle ΔV budget: greedy warm start uses the actual budget level
    # 10,000 m/s level uses 5-year time horizon (1825 days) for long-horizon missions
    (name      = "dvbudget",
     levels    = [string(Int(v)) for v in SE4_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"        => 200,
                               "type"               => "random",
                               "disttype"           => "uniform",
                               "deltaV_dist"        => 8000.0,
                               "time_dist"          => [50.0, 1825.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137,
     greedy_dv_fn   = (lv) -> parse(Float64, lv)),   # uses actual budget for warm start

    # SE5 — BCR study: finer ΔV budget grid, 1 trial only
    (name      = "bcr_dvbudget",
     levels    = [string(Int(v)) for v in SE5_LEVELS],
     base_fn   = (lv) -> Dict("num_demands"        => 200,
                               "type"               => "random",
                               "disttype"           => "uniform",
                               "deltaV_dist"        => 8000.0,
                               "time_dist"          => [50.0, 1825.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137,
     greedy_dv_fn   = (lv) -> parse(Float64, lv)),

    # SE7 — BCR mixed-fleet: 60–90 demands/year over 5 years, mixed 6–18 month
    # and 2–3 year contract windows. Sampled from all catalog sats (no pool cap).
    (name      = "bcr_mixed",
     levels    = [string(Int(v)) for v in SE5_LEVELS],
     base_fn   = (lv) -> begin
         Dict("type"               => "random",
              "disttype"           => "uniform",
              "deltaV_dist"        => 10000.0,
              "time_dist"          => [180.0, 1095.0],
              "service_times"      => [1.0, 5.0],
              "horizon_years"      => 5,
              "demands_per_year"   => [60, 90],
              "near_window_days"   => [180.0, 548.0],   # 6–18 months
              "far_window_days"    => [730.0, 1095.0],  # 2–3 years
              "far_frac"           => 0.5,
              "sat_values"         => Dict{String,Float64}(),
              "default_asset_value" => DEFAULT_ASSET_VALUE)
     end,
     seed_fn        = (trial, lv) -> trial * 137,
     greedy_dv_fn   = (lv) -> parse(Float64, lv)),

    # SE6 — mixed-fleet: 100 Starlink + 100 Planet Labs, 10k m/s budget, 5-year horizon
    # Requires outputs/simulation.h5 and cost_table.jld2 from run_mixed_fleet.jl
    (name      = "mixed_fleet",
     levels    = ["tight_normal", "loose_uniform"],
     base_fn   = (lv) -> Dict("num_demands"        => 200,
                               "type"               => "random",
                               "disttype"           => lv == "tight_normal" ? "normal" : "uniform",
                               "deltaV_dist"        => 10000.0,
                               "time_dist"          => [50.0, 1825.0],
                               "service_times"      => [1.0, 5.0],
                               "num_satellites"     => 100,
                               "default_asset_value" => DEFAULT_ASSET_VALUE),
     seed_fn        = (trial, lv) -> trial * 137,
     greedy_dv_fn   = (lv) -> 10000.0),
]

mkpath(OUT_DIR)
sim = load_sim()

# Optional filter: julia generate_sensitivity_demands.jl [se_name ...]
# e.g. julia generate_sensitivity_demands.jl dvbudget
#      julia generate_sensitivity_demands.jl size disttype
# No args → run all sub-experiments
_se_args     = filter(a -> !startswith(a, "--outdir="), ARGS)
filter_names = isempty(_se_args) ? nothing : Set(_se_args)
active_ses   = filter_names === nothing ? subexperiments :
               filter(se -> se.name in filter_names, subexperiments)
isempty(active_ses) && error("No sub-experiments matched: $(ARGS). Valid names: $(join([s.name for s in subexperiments], ", "))")

total = sum(length(se.levels) * N_TRIALS for se in active_ses)
done  = Ref(0)

# Load tables once — reused across all sub-experiments and trials
ct_path     = get(ENV, "COST_TABLE_PATH", joinpath(@__DIR__, "outputs", "cost_table.jld2"))
mintof_path = get(ENV, "MIN_TOF_PATH", nothing)
@info "Loading cost table …" ct_path
CostTable = load(ct_path, "CostTable")
@info "Cost table loaded" n_entries=length(CostTable)
@info "Building min-TOF table …"
MinTOFTable = if mintof_path !== nothing && isfile(mintof_path)
    load(mintof_path, "MinTOFTable")
else
    build_min_tof_table(; ct_path=ct_path)
end

for se in active_ses
    @info "─── Sub-experiment: $(se.name) ───"
    for lv in se.levels
        for trial in 1:N_TRIALS
            seed      = se.seed_fn(trial, lv)

            # Assign V1/V2 base values + apply Weibull depreciation by actual launch date
            sat_values_trial = depreciated_sat_values(sim, seed)

            params    = merge(se.base_fn(lv), Dict("seed" => seed, "sat_values" => sat_values_trial))
            if se.name == "bcr_mixed"
                params = merge(params, Dict(
                    "default_asset_value" => isempty(_MIXED_SAT_VALUES) ?
                        DEFAULT_ASSET_VALUE : mean(values(_MIXED_SAT_VALUES))))
            end
            trial_str = lpad(trial, 2, '0')
            key       = "$(se.name)_$(lv)"

            @info "  Generating" key=key trial=trial seed=seed
            demands = generate_demands(sim, params; cost_table=CostTable)

            # Save demand JLD2 (Python-readable via loaders.load_demands)
            dem_path = joinpath(OUT_DIR, "$(key)_$(trial_str).jld2")
            @save dem_path demands
            @info "  Saved demands" path=dem_path n=length(demands["UIDs"])

            # Greedy warm-start (uses budget-appropriate dv_budget for SE4)
            greedy_dv = se.greedy_dv_fn(lv)
            init_sol, init_unas = make_init_schedule(demands, sim; nvehicles=20,
                                                     refuel_time=REFUEL_TIME,
                                                     dv_budget=greedy_dv,
                                                     min_tof_table=MinTOFTable)
            greedy_path = joinpath(OUT_DIR, "$(key)_$(trial_str)_greedy.json")
            save_schedule_json(init_sol, init_unas, greedy_path)

            done[] += 1
            @info "  Progress" done=done[] total=total
        end
    end
end

@info "Done. Generated $(done[]) demand+greedy pairs." OUT_DIR
