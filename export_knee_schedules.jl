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

# ═════════════════════════════════════════════════════════════════════════════
#  2. NSGA-III  (local variant that returns the final population)
# ═════════════════════════════════════════════════════════════════════════════
function run_nsga3_export(ctx::RunContext; budget_evals::Int=BUDGET_EVALS_EXPORT)
    refs      = das_dennis(3, REF_H)
    pop_size  = size(refs, 1)
    @info "Running NSGA-III export …" budget_evals pop_size

    seed_sched = sched_eval(copy_schedule(ctx.init_sched), ctx.init_unas)
    seed_tour  = encode_tour(seed_sched, ctx)
    pop        = Vector{TourIndividual}(undef, pop_size)
    pop[1]     = seed_tour
    Threads.@threads for i in 2:pop_size
        pop[i] = tour_mutate(seed_tour, ctx; use_timing_mutation=false)
    end

    n_evaluated = pop_size
    while n_evaluated < budget_evals
        fronts = nondominated_sort(pop)
        rank   = zeros(Int, pop_size)
        for (r, front) in enumerate(fronts), i in front
            rank[i] = r
        end
        offspring = Vector{TourIndividual}(undef, pop_size)
        Threads.@threads for i in 1:pop_size
            a, b = rand(1:pop_size), rand(1:pop_size)
            p1   = rank[a] <= rank[b] ? a : b
            a, b = rand(1:pop_size), rand(1:pop_size)
            p2   = rank[a] <= rank[b] ? a : b
            child       = rand() < 0.8 ? ox_crossover(pop[p1], pop[p2], ctx) : pop[p1]
            offspring[i] = tour_mutate(child, ctx; use_timing_mutation=true)
        end
        n_evaluated += pop_size
        pop = nsga3_select(vcat(pop, offspring), pop_size, refs)
    end
    @info "NSGA-III export done" n_evaluated
    return pop  # Vector{TourIndividual}
end

@info "Running NSGA-III …"
nsga3_pop = run_nsga3_export(ctx; budget_evals=BUDGET_EVALS_EXPORT)

f1s = [ind.f1 for ind in nsga3_pop]
f2s = [ind.f2 for ind in nsga3_pop]
f3s = [ind.f3 for ind in nsga3_pop]
knee_idx_nsga3 = find_knee(f1s, f2s, f3s)
ind_knee_nsga3 = nsga3_pop[knee_idx_nsga3]

decoded_nsga3  = decode_tour(ind_knee_nsga3, ctx)
out_nsga3      = joinpath(OUT_DIR, "knee_schedule_NSGA-III.json")
save_schedule_json(decoded_nsga3.schedule, decoded_nsga3.unassigned, out_nsga3)
@info "Saved NSGA-III knee schedule" path=out_nsga3 f1=ind_knee_nsga3.f1 f3=ind_knee_nsga3.f3

# ═════════════════════════════════════════════════════════════════════════════
#  3. MOPSO-CD  (local variant that returns the final archive)
# ═════════════════════════════════════════════════════════════════════════════
function run_pso_export(ctx::RunContext;
                        budget_evals::Int = BUDGET_EVALS_EXPORT,
                        pop_size::Int     = POP_SIZE,
                        c1::Float64       = 0.3,
                        c2::Float64       = 0.3,
                        archive_size::Int = 50)
    @info "Running MOPSO-CD export …" budget_evals pop_size

    seed_sched = sched_eval(copy_schedule(ctx.init_sched), ctx.init_unas)
    seed_tour  = encode_tour(seed_sched, ctx)
    particles  = Vector{Particle}(undef, pop_size)
    particles[1] = Particle(seed_sched, seed_sched)
    Threads.@threads for i in 2:pop_size
        ind = decode_tour(tour_mutate(seed_tour, ctx; use_timing_mutation=false), ctx)
        particles[i] = Particle(ind, ind)
    end

    arch = PSOArchive(archive_size)
    for p in particles
        _pso_archive_add!(arch, p.position)
    end

    n_evaluated = pop_size
    while n_evaluated < budget_evals
        for particle in particles
            n_evaluated >= budget_evals && break
            new_pos = particle.position
            if rand() < c2 && !isempty(arch.members)
                new_pos = brx_crossover(_pso_archive_gbest(arch), new_pos, ctx)
            end
            if rand() < c1
                new_pos = brx_crossover(particle.pbest, new_pos, ctx)
            end
            new_pos           = pso_inertia_mutate(new_pos, ctx; use_timing_mutation=true)
            n_evaluated      += 1
            particle.pbest    = _pso_update_pbest(particle, new_pos)
            particle.position = new_pos
            _pso_archive_add!(arch, new_pos)
        end
    end
    @info "MOPSO-CD export done" n_evaluated archive_size=length(arch.members)
    return arch
end

@info "Running MOPSO-CD …"
pso_arch = run_pso_export(ctx; budget_evals=BUDGET_EVALS_EXPORT)

pf1s = [m.f1 for m in pso_arch.members]
pf2s = [m.f2 for m in pso_arch.members]
pf3s = [m.f3 for m in pso_arch.members]
knee_idx_pso = find_knee(pf1s, pf2s, pf3s)
ind_knee_pso = pso_arch.members[knee_idx_pso]

out_pso = joinpath(OUT_DIR, "knee_schedule_MOPSO-CD.json")
save_schedule_json(ind_knee_pso.schedule, ind_knee_pso.unassigned, out_pso)
@info "Saved MOPSO-CD knee schedule" path=out_pso f1=ind_knee_pso.f1 f3=ind_knee_pso.f3

# ── summary ───────────────────────────────────────────────────────────────────
println()
println("═" ^ 60)
println("  Knee point schedules saved:")
println("  MDLS     → $out_mdls")
println("  NSGA-III → $out_nsga3")
println("  MOPSO-CD → $out_pso")
println("═" ^ 60)
println()
println("Next: python plot_gantt.py")
