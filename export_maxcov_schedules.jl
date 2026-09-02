# export_maxcov_schedules.jl
# Runs MDLS and NSGA-III on tight_normal trial 1 with a large DV budget,
# selects the max-coverage solution (min f2), and saves schedules as JSON.
#
# Run: julia --project=. -t auto export_maxcov_schedules.jl
# Output: outputs/maxcov_schedule_MDLS.json
#         outputs/maxcov_schedule_NSGA-III.json

const NUMEXP_INCLUDE  = true
const GATESTS_INCLUDE = true
include("algoMDLS.jl")
include("GATests.jl")

using JLD2, JSON3, LinearAlgebra

const BUDGET_EVALS  = 10_000
const SCENARIO      = "tight_normal"
const TRIAL         = 1
const OUT_DIR       = joinpath(@__DIR__, "outputs")
const DV_BUDGET_MAX = 8000.0   # large budget to allow intermediate depot visits

const PENALTY = 9.0e8
const REF_NORM = [1.1, 1.1, 1.1]

# ── load infrastructure ───────────────────────────────────────────────────────
@info "Loading cost table and sim …"
cost_table   = load_cost_table()
mintof_table = build_min_tof_table()
min_dv_tab   = build_min_dv_table(cost_table, maximum(k[1] for k in keys(cost_table)))
_ga_sim      = load_sim()

trial_str = lpad(TRIAL, 2, '0')
dem_path  = joinpath(OUT_DIR, "exp_demands", "$(SCENARIO)_$(trial_str).jld2")
isfile(dem_path) || error("Demand file not found: $dem_path")
@load dem_path demands
@info "Loaded demands" n=length(demands["UIDs"]) scenario=SCENARIO trial=TRIAL

ctx = make_context(demands, _ga_sim, cost_table, mintof_table, min_dv_tab;
                   nvehicles=20, dv_budget=DV_BUDGET_MAX)

function find_maxcov(f2s::Vector{Float64}) :: Int
    valid = findall(f2 -> f2 < PENALTY, f2s)
    isempty(valid) && return 1
    return valid[argmin(f2s[valid])]
end

# ═════════════════════════════════════════════════════════════════════════════
#  1. MDLS
# ═════════════════════════════════════════════════════════════════════════════
@info "Running MDLS (DV=$(DV_BUDGET_MAX) m/s) …"
arch_mdls = MDLS(max(1, BUDGET_EVALS ÷ 9), demands, _ga_sim, cost_table, mintof_table, min_dv_tab;
                 init_sol        = copy_schedule(ctx.init_sched),
                 init_unassigned = ctx.init_unas,
                 dv_budget       = DV_BUDGET_MAX)

idx_mdls   = find_maxcov(arch_mdls.total_serv_time_unassigned)
sched_mdls = arch_mdls.solutions[idx_mdls]
unas_mdls  = arch_mdls.unassigned_sets[idx_mdls]

out_mdls = joinpath(OUT_DIR, "maxcov_schedule_MDLS.json")
save_schedule_json(sched_mdls, unas_mdls, out_mdls)
@info "Saved MDLS max-coverage schedule" path=out_mdls f2=arch_mdls.total_serv_time_unassigned[idx_mdls] f3=arch_mdls.total_vehicles_used[idx_mdls]


println()
println("═" ^ 60)
println("  Max-coverage schedule saved:")
println("  MDLS → $out_mdls")
println("═" ^ 60)
println()
println("Next: python plot_gantt.py --mdls outputs/maxcov_schedule_MDLS.json")
