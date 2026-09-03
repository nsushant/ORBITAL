# export_knee_schedules.jl
# Re-runs MDLS, NSGA-III, and PSO (MOPSO-CD) on tight_normal trial 1,
# finds the knee point on each algorithm's Pareto front, and saves the
# corresponding vehicle schedule as a JSON file.
#
# Run: julia --project=. -t auto export_knee_schedules.jl
# Output: outputs/knee_schedule_MDLS.json
#         outputs/knee_schedule_NSGA-III.json
#         outputs/knee_schedule_MOPSO-CD.json

const NUMEXP_INCLUDE  = true
const GATESTS_INCLUDE = true
include("algoMDLS.jl")
include("GATests.jl")

using JLD2, JSON3, LinearAlgebra

# ── fixed normalisation bounds (matches numerical experiments) ────────────────
const _AVG_ASSET = 0.30 * 3_250_000.0 + 0.70 * 1_330_550.0
const FIXED_NADIR = Float64[1_000_000.0, 200 * _AVG_ASSET * 1.10, 25.0]
const REF_NORM    = [1.1, 1.1, 1.1]
const PENALTY     = 9.0e8

const BUDGET_EVALS_EXPORT = 10_000
const SCENARIO   = "tight_normal"
const TRIAL      = 1
const OUT_DIR    = joinpath(@__DIR__, "outputs")

# ── load infrastructure ───────────────────────────────────────────────────────
@info "Loading cost table and sim …"
cost_table   = load_cost_table()
mintof_table = build_min_tof_table()
min_dv_tab   = build_min_dv_table(cost_table, maximum(k[1] for k in keys(cost_table)))
_ga_sim      = load_sim()

# ── load pre-saved demands ────────────────────────────────────────────────────
trial_str = lpad(TRIAL, 2, '0')
dem_path  = joinpath(OUT_DIR, "exp_demands", "$(SCENARIO)_$(trial_str).jld2")
isfile(dem_path) || error("Demand file not found: $dem_path")
@load dem_path demands
@info "Loaded demands" n=length(demands["UIDs"]) scenario=SCENARIO trial=TRIAL

# ── build context ─────────────────────────────────────────────────────────────
ctx = make_context(demands, _ga_sim, cost_table, mintof_table, min_dv_tab;
                   nvehicles=20, dv_budget=DV_BUDGET)

# ── helper: compute knee in normalised space ──────────────────────────────────
# Returns index into the *ordered* array (dv, unrecov, veh).
# Uses data-driven ideal (min across front) + fixed nadir.
function find_knee(pts_dv::Vector{Float64},
                   pts_unrecov::Vector{Float64},
                   pts_veh::Vector{Float64}) :: Int
    n = length(pts_dv)
    @assert n == length(pts_unrecov) == length(pts_veh)
    # reorder to match normalisation: [dv, veh, unrecov] → [dv, unrecov, veh]
    F = hcat(pts_dv, pts_unrecov, pts_veh)   # n × 3
    # filter out penalty solutions
    valid = [F[i, 2] < PENALTY for i in 1:n]
    any(valid) || return 1
    F_valid = F[valid, :]
    ideal   = vec(minimum(F_valid; dims=1))
    rng     = max.(FIXED_NADIR .- ideal, 1e-10)
    norms   = [(F_valid[i, :] .- ideal) ./ rng for i in 1:size(F_valid, 1)]
    l2      = [norm(v) for v in norms]
    knee_in_valid = argmin(l2)
    # map back to original index
    valid_indices = findall(valid)
    return valid_indices[knee_in_valid]
end

# ═════════════════════════════════════════════════════════════════════════════
#  1. MDLS
# ═════════════════════════════════════════════════════════════════════════════
@info "Running MDLS …"
mdls_iters = max(1, BUDGET_EVALS_EXPORT ÷ 9)
arch_mdls  = MDLS(mdls_iters, demands, _ga_sim, cost_table, mintof_table, min_dv_tab;
                  init_sol = copy_schedule(ctx.init_sched),
                  init_unassigned = ctx.init_unas,
                  dv_budget = DV_BUDGET)

@info "MDLS done" n_solutions=length(arch_mdls.solutions)

knee_idx_mdls = find_knee(arch_mdls.total_deltaV,
                           arch_mdls.total_serv_time_unassigned,
                           Float64.(arch_mdls.total_vehicles_used))
sched_mdls = arch_mdls.solutions[knee_idx_mdls]
unas_mdls  = arch_mdls.unassigned_sets[knee_idx_mdls]

out_mdls = joinpath(OUT_DIR, "knee_schedule_MDLS.json")
save_schedule_json(sched_mdls, unas_mdls, out_mdls)
@info "Saved MDLS knee schedule" path=out_mdls f1=arch_mdls.total_deltaV[knee_idx_mdls] f3=arch_mdls.total_vehicles_used[knee_idx_mdls]


println()
println("═" ^ 60)
println("  Knee schedule saved:")
println("  MDLS → $out_mdls")
println("═" ^ 60)
println()
println("Next: python3 plot_gantt.py")
