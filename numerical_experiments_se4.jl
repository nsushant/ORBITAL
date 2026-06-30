# numerical_experiments_se4.jl
# SE4: Sensitivity to vehicle ΔV budget (fuel per leg between depot visits)
#
# Factor: dv_budget ∈ {1500, 3000, 5000, 8000} m/s
# Fixed:  num_demands=200, num_satellites=100, disttype="uniform",
#         deltaV_dist=8000, time_dist=[50,365], service_times=[1,5]
#
# Run: julia --project=. -t auto numerical_experiments_se4.jl

using CSV, DataFrames, Printf, PyCall, JSON3
const moocore = pyimport("moocore")

const NUMEXP_INCLUDE = true
include("GATests.jl")

const SAT_VALUES_PATH_SE4 = joinpath(@__DIR__, "outputs", "sat_values.json")
const SAT_VALUES_SE4 = isfile(SAT_VALUES_PATH_SE4) ?
    JSON3.read(read(SAT_VALUES_PATH_SE4), Dict{String,Float64}) :
    Dict{String,Float64}()
isempty(SAT_VALUES_SE4) && @warn "sat_values.json not found — run starlink/run_starlink.jl first"

# ─────────────────────────────────────────────────────────────────────────────
#  Shared settings
# ─────────────────────────────────────────────────────────────────────────────

const N_TRIALS_SE4   = 10
const ALG_NAMES_SE4  = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]

function _hv_se4(front::Matrix{Float64}, ref::Vector{Float64}) :: Float64
    @assert size(front, 2) == 3
    return moocore.hypervolume(front, ref=ref, maximise=false)
end

function _norm_front_se4(front::Matrix{Float64},
                          ideal::Vector{Float64},
                          rng::Vector{Float64}) :: Matrix{Float64}
    return (front .- ideal') ./ rng'
end

# ─────────────────────────────────────────────────────────────────────────────
#  Sub-experiment runner (standalone copy — does not depend on sensitivity file)
# ─────────────────────────────────────────────────────────────────────────────

function run_subexperiment_se4(levels::Vector,
                                factor_col::Symbol,
                                csv_path::String,
                                seed_fn::Function)

    n_levels = length(levels)
    @info "─── SE4: $factor_col  ($n_levels levels × $N_TRIALS_SE4 trials × $(length(ALG_NAMES_SE4)) algs) ───"

    front_store = Dict{Tuple{Int,String}, Vector{Matrix{Float64}}}()
    time_store  = Dict{Tuple{Int,String}, Vector{Float64}}()
    for li in 1:n_levels, alg in ALG_NAMES_SE4
        front_store[(li, alg)] = Matrix{Float64}[]
        time_store[(li,  alg)] = Float64[]
    end

    for (li, lv) in enumerate(levels)
        label, base_params = lv.label, lv.params
        @info "  Level $li/$n_levels: $factor_col = $label"

        for trial in 1:N_TRIALS_SE4
            seed         = seed_fn(trial, li, lv)
            trial_params = merge(base_params, Dict("seed"       => seed,
                                                   "sat_values" => SAT_VALUES_SE4))

            trial_demands = generate_demands(_ga_sim, trial_params)

            # Key difference from SE1-3: pass dv_budget from params to make_context
            ctx = make_context(trial_demands, _ga_sim, cost_table, mintof_table,
                               min_dv_tab;
                               nvehicles  = 20,
                               dv_budget  = Float64(base_params["dv_budget"]))

            init_copies = ntuple(_ -> (copy_schedule(ctx.init_sched), ctx.init_unas), 4)
            results     = run_all_algorithms(ctx; budget_evals = BUDGET_EVALS,
                                             init_copies = init_copies)

            total_val = sum(trial_demands["asset_values"])
            for alg in ALG_NAMES_SE4
                front, t  = results[alg]
                front_fin = front[vec(all(isfinite, front; dims=2)), :]
                tagged    = hcat(front_fin, fill(total_val, size(front_fin, 1)))
                push!(front_store[(li, alg)], tagged)
                push!(time_store[(li,  alg)], t)
            end

            @info "    Trial $trial/$N_TRIALS_SE4 done"
        end
    end

    # ── Normalisation (per sub-experiment) ───────────────────────────────────
    all_pts = vcat([front_store[(li, alg)][t][:, 1:3]
                    for li  in 1:n_levels
                    for alg in ALG_NAMES_SE4
                    for t   in 1:N_TRIALS_SE4
                    if size(front_store[(li, alg)][t], 1) > 0]...)

    if size(all_pts, 1) == 0
        @warn "No finite points — skipping CSV write"
        return
    end

    se_ideal = vec(minimum(all_pts; dims=1))
    se_nadir = vec(maximum(all_pts; dims=1))
    se_range = max.(se_nadir .- se_ideal, 1e-10)
    se_ref   = ones(3) .* 1.1
    @info "  Normalisation" se_ideal se_nadir

    # ── Build output rows ─────────────────────────────────────────────────────
    factor_vals  = String[]; alg_vals = String[]; trial_vals = Int[]
    hv_vals      = Float64[]; elapsed_vals = Float64[]

    pf_factor  = String[]; pf_alg = String[]; pf_trial = Int[]
    pf_f1_norm = Float64[]; pf_f2_norm = Float64[]; pf_f3_norm = Float64[]
    pf_f1      = Float64[]; pf_f2      = Float64[]; pf_f3      = Float64[]
    pf_valrec  = Float64[]; pf_totval  = Float64[]

    for (li, lv) in enumerate(levels)
        label = lv.label
        for alg in ALG_NAMES_SE4
            for trial in 1:N_TRIALS_SE4
                front_4   = front_store[(li, alg)][trial]
                front_fin = front_4[:, 1:3]
                total_val = size(front_4, 1) > 0 ? front_4[1, 4] : 0.0
                front_u   = size(front_fin, 1) > 0 ? unique(front_fin, dims=1) : front_fin
                hv = if size(front_u, 1) == 0
                    0.0
                else
                    _hv_se4(_norm_front_se4(front_u, se_ideal, se_range), se_ref)
                end
                push!(factor_vals, label); push!(alg_vals, alg)
                push!(trial_vals, trial);  push!(hv_vals, hv)
                push!(elapsed_vals, time_store[(li, alg)][trial])

                if size(front_u, 1) > 0
                    norm = _norm_front_se4(front_u, se_ideal, se_range)
                    for i in 1:size(front_u, 1)
                        push!(pf_factor, label); push!(pf_alg, alg); push!(pf_trial, trial)
                        push!(pf_f1_norm, norm[i,1]); push!(pf_f2_norm, norm[i,2]); push!(pf_f3_norm, norm[i,3])
                        push!(pf_f1, front_u[i,1]);   push!(pf_f2, front_u[i,2]);   push!(pf_f3, front_u[i,3])
                        push!(pf_valrec, total_val - front_u[i,2]); push!(pf_totval, total_val)
                    end
                end
            end
        end
    end

    mkpath(dirname(csv_path))

    df_hv = DataFrame(factor_col   => factor_vals, :algorithm   => alg_vals,
                      :trial       => trial_vals,  :hypervolume => hv_vals,
                      :elapsed     => elapsed_vals)
    CSV.write(csv_path, df_hv)
    @info "  Saved: $csv_path  ($(nrow(df_hv)) rows)"

    pf_csv = replace(csv_path, ".csv" => "_fronts.csv")
    df_pf  = DataFrame(factor_col               => pf_factor,  :algorithm               => pf_alg,
                       :trial                   => pf_trial,
                       :f1_dv_norm              => pf_f1_norm, :f2_unrecovered_norm     => pf_f2_norm,
                       :f3_vehicles_norm        => pf_f3_norm,
                       :f1_dv                   => pf_f1,      :f2_unrecovered_value    => pf_f2,
                       :f3_vehicles             => pf_f3,      :value_recovered         => pf_valrec,
                       :total_demand_value      => pf_totval)
    CSV.write(pf_csv, df_pf)
    @info "  Saved: $pf_csv  ($(nrow(df_pf)) front points)"
end

# ═════════════════════════════════════════════════════════════════════════════
#  SE4 — Vehicle ΔV budget
# ═════════════════════════════════════════════════════════════════════════════

const SE4_DV_BUDGET = [1500.0, 3000.0, 5000.0, 8000.0]

se4_levels = [
    (label  = string(Int(dv)),
     params = Dict(
         "num_demands"    => 200,
         "num_satellites" => 100,
         "type"           => "random",
         "disttype"       => "uniform",
         "deltaV_dist"    => 8000,
         "time_dist"      => [50.0, 365.0],
         "service_times"  => [1.0,  5.0],
         "dv_budget"      => dv,
     ))
    for dv in SE4_DV_BUDGET
]

run_subexperiment_se4(
    se4_levels,
    :dv_budget,
    joinpath(@__DIR__, "outputs", "sensitivity_dvbudget.csv"),
    (trial, li, lv) -> trial * 137 + Int(parse(Float64, lv.label)),
)

@info "SE4 complete."
@printf "\nOutputs:\n"
@printf "  outputs/sensitivity_dvbudget.csv        — HV (%d rows)\n"  4*N_TRIALS_SE4*length(ALG_NAMES_SE4)
@printf "  outputs/sensitivity_dvbudget_fronts.csv — Pareto front points\n"
