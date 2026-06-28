# numerical_experiments_sensitivity.jl
# OFAT sensitivity analysis — 3 sub-experiments, one factor varied at a time.
#
# Sub-experiment 1: Instance size  (num_demands ∈ {10,50,100,150,200})
# Sub-experiment 2: Demand distrib (disttype   ∈ {"normal","uniform"})
# Sub-experiment 3: ΔV threshold  (deltaV_dist ∈ {3000,5000,8000,12000})
#
# Run: julia --project=. -t auto numerical_experiments_sensitivity.jl

using CSV, DataFrames, Printf, PyCall
const moocore = pyimport("moocore")

const NUMEXP_INCLUDE = true   # suppress single-run block in GATests.jl
include("GATests.jl")         # loads run_all_algorithms, make_context, copy_schedule, etc.

# ─────────────────────────────────────────────────────────────────────────────
#  Shared settings
# ─────────────────────────────────────────────────────────────────────────────

const N_TRIALS_SENS = 10
const ALG_NAMES_SENS = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]

function _hv_sens(front::Matrix{Float64}, ref::Vector{Float64}) :: Float64
    @assert size(front, 2) == 3
    return moocore.hypervolume(front, ref=ref, maximise=false)
end

function _norm_front(front::Matrix{Float64},
                     ideal::Vector{Float64},
                     rng::Vector{Float64}) :: Matrix{Float64}
    return (front .- ideal') ./ rng'
end

# ─────────────────────────────────────────────────────────────────────────────
#  Generic sub-experiment runner
#
#  `levels`      — Vector of named tuples (label, params) where
#                  label is the factor value as a string (used in CSV),
#                  params is the full demand_params Dict.
#  `factor_col`  — Symbol: column name for the factor in the output CSV
#                  (e.g. :num_demands, :disttype, :deltaV_dist)
#  `csv_path`    — Output CSV file path
#  `seed_fn`     — Function (trial, level_idx) → Int seed
# ─────────────────────────────────────────────────────────────────────────────

function run_subexperiment(levels::Vector,
                            factor_col::Symbol,
                            csv_path::String,
                            seed_fn::Function)

    n_levels = length(levels)
    @info "─── Sub-experiment: $factor_col  ($n_levels levels × $N_TRIALS_SENS trials × $(length(ALG_NAMES_SENS)) algs) ───"

    # ── Phase 1: collect raw fronts ──────────────────────────────────────────
    # front_store[(level_idx, alg)] = Vector{Matrix{Float64}} (one per trial)
    front_store  = Dict{Tuple{Int,String}, Vector{Matrix{Float64}}}()
    time_store   = Dict{Tuple{Int,String}, Vector{Float64}}()
    for li in 1:n_levels, alg in ALG_NAMES_SENS
        front_store[(li, alg)] = Matrix{Float64}[]
        time_store[(li,  alg)] = Float64[]
    end

    for (li, lv) in enumerate(levels)
        label, base_params = lv.label, lv.params
        @info "  Level $li/$n_levels: $factor_col = $label"

        for trial in 1:N_TRIALS_SENS
            seed         = seed_fn(trial, li, lv)
            trial_params = merge(base_params, Dict("seed" => seed))

            trial_demands = generate_demands(_ga_sim, trial_params)
            ctx = make_context(trial_demands, _ga_sim, cost_table, mintof_table,
                               min_dv_tab; nvehicles = 20)

            init_copies = ntuple(_ -> (copy_schedule(ctx.init_sched), ctx.init_unas), 4)
            results     = run_all_algorithms(ctx; budget_evals = BUDGET_EVALS,
                                             init_copies = init_copies)

            for alg in ALG_NAMES_SENS
                front, t = results[alg]
                front_fin = front[vec(all(isfinite, front; dims=2)), :]
                push!(front_store[(li, alg)], front_fin)
                push!(time_store[(li,  alg)], t)
            end

            @info "    Trial $trial/$N_TRIALS_SENS done"
        end
    end

    # ── Phase 2: global normalisation within this sub-experiment ─────────────
    all_pts = vcat([front_store[(li, alg)][t]
                    for li  in 1:n_levels
                    for alg in ALG_NAMES_SENS
                    for t   in 1:N_TRIALS_SENS
                    if size(front_store[(li, alg)][t], 1) > 0]...)

    if size(all_pts, 1) == 0
        @warn "No finite points in sub-experiment $factor_col — skipping CSV write"
        return
    end

    se_ideal = vec(minimum(all_pts; dims=1))
    se_nadir = vec(maximum(all_pts; dims=1))
    se_range = max.(se_nadir .- se_ideal, 1e-10)
    se_ref   = ones(3) .* 1.1

    @info "  Normalisation" se_ideal se_nadir

    # ── Phase 3: compute HV and write CSVs ───────────────────────────────────

    # HV summary
    factor_vals  = String[]
    alg_vals     = String[]
    trial_vals   = Int[]
    hv_vals      = Float64[]
    elapsed_vals = Float64[]

    # Pareto front points (for shadow/knee plots)
    pf_factor  = String[]
    pf_alg     = String[]
    pf_trial   = Int[]
    pf_f1_norm = Float64[]
    pf_f2_norm = Float64[]
    pf_f3_norm = Float64[]
    pf_f1      = Float64[]
    pf_f2      = Float64[]
    pf_f3      = Float64[]

    for (li, lv) in enumerate(levels)
        label = lv.label
        for alg in ALG_NAMES_SENS
            for trial in 1:N_TRIALS_SENS
                front_fin = front_store[(li, alg)][trial]
                front_u   = size(front_fin, 1) > 0 ? unique(front_fin, dims=1) : front_fin
                hv = if size(front_u, 1) == 0
                    0.0
                else
                    _hv_sens(_norm_front(front_u, se_ideal, se_range), se_ref)
                end
                push!(factor_vals,  label)
                push!(alg_vals,     alg)
                push!(trial_vals,   trial)
                push!(hv_vals,      hv)
                push!(elapsed_vals, time_store[(li, alg)][trial])

                # Append individual front points
                if size(front_u, 1) > 0
                    norm = _norm_front(front_u, se_ideal, se_range)
                    for i in 1:size(front_u, 1)
                        push!(pf_factor,  label)
                        push!(pf_alg,     alg)
                        push!(pf_trial,   trial)
                        push!(pf_f1_norm, norm[i, 1])
                        push!(pf_f2_norm, norm[i, 2])
                        push!(pf_f3_norm, norm[i, 3])
                        push!(pf_f1,      front_u[i, 1])
                        push!(pf_f2,      front_u[i, 2])
                        push!(pf_f3,      front_u[i, 3])
                    end
                end
            end
        end
    end

    mkpath(dirname(csv_path))

    df_hv = DataFrame(
        factor_col   => factor_vals,
        :algorithm   => alg_vals,
        :trial       => trial_vals,
        :hypervolume => hv_vals,
        :elapsed     => elapsed_vals,
    )
    CSV.write(csv_path, df_hv)
    @info "  Saved: $csv_path  ($(nrow(df_hv)) rows)"

    pf_csv = replace(csv_path, ".csv" => "_fronts.csv")
    df_pf = DataFrame(
        factor_col          => pf_factor,
        :algorithm          => pf_alg,
        :trial              => pf_trial,
        :f1_dv_norm         => pf_f1_norm,
        :f2_unassigned_norm => pf_f2_norm,
        :f3_vehicles_norm   => pf_f3_norm,
        :f1_dv              => pf_f1,
        :f2_unassigned_time => pf_f2,
        :f3_vehicles        => pf_f3,
    )
    CSV.write(pf_csv, df_pf)
    @info "  Saved: $pf_csv  ($(nrow(df_pf)) front points)"
end

# ═════════════════════════════════════════════════════════════════════════════
#  Sub-experiment 1 — Instance size
#  Factor: num_demands ∈ {10, 50, 100, 150, 200}
#  Fixed:  disttype="uniform", deltaV_dist=8000, time_dist=[50,365], service_times=[1,5]
# ═════════════════════════════════════════════════════════════════════════════

const SE1_SIZES = [10, 50, 100, 150, 200]

se1_levels = [
    (label  = string(nd),
     params = Dict(
         "num_demands"    => nd,
         "num_satellites" => min(100, nd),   # clamp for small sizes
         "type"           => "random",
         "disttype"       => "uniform",
         "deltaV_dist"    => 8000,
         "time_dist"      => [50.0, 365.0],
         "service_times"  => [1.0,  5.0],
     ))
    for nd in SE1_SIZES
]

run_subexperiment(
    se1_levels,
    :num_demands,
    joinpath(@__DIR__, "outputs", "sensitivity_size.csv"),
    (trial, li, lv) -> trial * 137 + parse(Int, lv.label),   # seed = trial*137 + nd
)

# ═════════════════════════════════════════════════════════════════════════════
#  Sub-experiment 2 — Distribution type
#  Factor: disttype ∈ {"normal", "uniform"}
#  Fixed:  num_demands=200, num_satellites=100, deltaV_dist=8000,
#          time_dist=[50,365], service_times=[1,5]
# ═════════════════════════════════════════════════════════════════════════════

se2_levels = [
    (label  = dt,
     params = Dict(
         "num_demands"    => 200,
         "num_satellites" => 100,
         "type"           => "random",
         "disttype"       => dt,
         "deltaV_dist"    => 8000,
         "time_dist"      => [50.0, 365.0],
         "service_times"  => [1.0,  5.0],
     ))
    for dt in ["normal", "uniform"]
]

run_subexperiment(
    se2_levels,
    :disttype,
    joinpath(@__DIR__, "outputs", "sensitivity_disttype.csv"),
    (trial, li, lv) -> trial * 137 + Int(abs(hash(lv.label)) % 1000),
)

# ═════════════════════════════════════════════════════════════════════════════
#  Sub-experiment 3 — ΔV threshold
#  Factor: deltaV_dist ∈ {3000, 5000, 8000, 12000}
#  Fixed:  num_demands=200, num_satellites=100, disttype="uniform",
#          time_dist=[50,365], service_times=[1,5]
# ═════════════════════════════════════════════════════════════════════════════

const SE3_DV = [3000, 5000, 8000, 12000]

se3_levels = [
    (label  = string(dv),
     params = Dict(
         "num_demands"    => 200,
         "num_satellites" => 100,
         "type"           => "random",
         "disttype"       => "uniform",
         "deltaV_dist"    => dv,
         "time_dist"      => [50.0, 365.0],
         "service_times"  => [1.0,  5.0],
     ))
    for dv in SE3_DV
]

run_subexperiment(
    se3_levels,
    :deltaV_dist,
    joinpath(@__DIR__, "outputs", "sensitivity_dv.csv"),
    (trial, li, lv) -> trial * 137 + parse(Int, lv.label) ÷ 100,
)

@info "All sensitivity sub-experiments complete."
@printf "\nOutput files (HV summary + Pareto front points per sub-experiment):\n"
@printf "  outputs/sensitivity_size.csv            — HV (%d rows)\n"     5*N_TRIALS_SENS*length(ALG_NAMES_SENS)
@printf "  outputs/sensitivity_size_fronts.csv     — front points\n"
@printf "  outputs/sensitivity_disttype.csv        — HV (%d rows)\n"     2*N_TRIALS_SENS*length(ALG_NAMES_SENS)
@printf "  outputs/sensitivity_disttype_fronts.csv — front points\n"
@printf "  outputs/sensitivity_dv.csv              — HV (%d rows)\n"     4*N_TRIALS_SENS*length(ALG_NAMES_SENS)
@printf "  outputs/sensitivity_dv_fronts.csv       — front points\n"
