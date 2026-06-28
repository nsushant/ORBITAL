# numerical_experiments.jl — multi-instance hypervolume benchmark
# Runs MDLS, NSGA-III, MOEA/D, PSO and reports
# median hypervolume ± IQR across N_TRIALS independent trials per instance.
#
# Run: julia --project=. -t auto numerical_experiments.jl

using CSV, DataFrames, Printf, PyCall
const moocore = pyimport("moocore")

function compute_hypervolume(front::Matrix{Float64}, ref::Vector{Float64})
    @assert size(front, 2) == 3 "Expected 3 objectives, got $(size(front, 2))"
    @assert length(ref) == 3    "ref must have 3 components"
    return moocore.hypervolume(front, ref=ref, maximise=false)
end

const NUMEXP_INCLUDE = true
const GATESTS_INCLUDE = true
include("GATests.jl")   # loads all algorithm functions; skips single-run block due to NUMEXP_INCLUDE

# ═════════════════════════════════════════════════════════════════════════════
#  INSTANCE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

# All instances share the same problem size (num_demands, num_satellites).
# Only the temporal spread (time_dist) and ΔV accessibility (deltaV_dist)
# vary across instances, so mean HV comparisons are meaningful.

const N_DEMANDS = 200
const N_SATS    = 100

const INSTANCES = [
    (name   = "tight_normal",
     params = Dict("num_demands"    => N_DEMANDS,
                   "num_satellites" => N_SATS,
                   "type"           => "random",
                   "disttype"       => "normal",
                   "deltaV_dist"    => 8000,
                   "time_dist"      => [10, 100],
                   "service_times"  => [1.0, 3.0])),

    (name   = "loose_uniform",
     params = Dict("num_demands"    => N_DEMANDS,
                   "num_satellites" => N_SATS,
                   "type"           => "random",
                   "disttype"       => "uniform",
                   "deltaV_dist"    => 8000,
                   "time_dist"      => [50, 365],
                   "service_times"  => [1.0, 5.0])),

    (name   = "tight_low_dv",
     params = Dict("num_demands"    => N_DEMANDS,
                   "num_satellites" => N_SATS,
                   "type"           => "random",
                   "disttype"       => "normal",
                   "deltaV_dist"    => 5000,
                   "time_dist"      => [10, 200],
                   "service_times"  => [1.0, 4.0])),

    (name   = "loose_high_dv",
     params = Dict("num_demands"    => N_DEMANDS,
                   "num_satellites" => N_SATS,
                   "type"           => "random",
                   "disttype"       => "uniform",
                   "deltaV_dist"    => 12000,
                   "time_dist"      => [100, 365],
                   "service_times"  => [1.0, 5.0])),
]

const N_TRIALS    = 10
const BUDGET_EVALS = 10_000  # solution evaluations — same for every algorithm
const ALG_NAMES   = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]

# ═════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT LOOP
# ═════════════════════════════════════════════════════════════════════════════

# front_store[inst_name][alg] = Vector of fronts (one per trial)
front_store  = Dict(inst.name => Dict(a => Matrix{Float64}[] for a in ALG_NAMES) for inst in INSTANCES)
time_results = Dict(inst.name => Dict(a => Float64[]         for a in ALG_NAMES) for inst in INSTANCES)

for inst in INSTANCES
    @info "─── Instance: $(inst.name) ───"
    for trial in 1:N_TRIALS
        @info "  Trial $trial / $N_TRIALS"

        # New demand draw each trial; same scenario characteristics, fixed size
        trial_params  = merge(inst.params, Dict("seed" => trial * 137 + hash(inst.name) % 1000))
        trial_demands = generate_demands(_ga_sim, trial_params)
        ctx = make_context(trial_demands, _ga_sim, cost_table, mintof_table,
                           min_dv_tab; nvehicles=20)

        # 4 explicit deep copies — one per algorithm, identical starting point
        init_copies = ntuple(_ -> (copy_schedule(ctx.init_sched), ctx.init_unas), 4)

        trial_results = run_all_algorithms(ctx; budget_evals = BUDGET_EVALS,
                                           init_copies = init_copies)

        for alg in ALG_NAMES
            front, t = trial_results[alg]
            push!(front_store[inst.name][alg], front)
            push!(time_results[inst.name][alg], t)
        end

        # Per-trial checkpoint (crash-safe)
        chk_path   = joinpath(@__DIR__, "outputs", "checkpoint.csv")
        mkpath(dirname(chk_path))
        chk_header = !isfile(chk_path)
        open(chk_path, "a") do io
            chk_header && println(io, "instance,budget_evals,algorithm,trial,front_size,elapsed")
            for alg in ALG_NAMES
                front = front_store[inst.name][alg][end]
                t     = time_results[inst.name][alg][end]
                println(io, "$(inst.name),$BUDGET_EVALS,$alg,$trial,$(size(front, 1)),$t")
            end
        end
        @info "  Checkpoint written" trial inst.name
    end
end

# ── Phase 2: compute one global fixed reference point from ALL fronts ──────────
all_finite_pts = vcat([
    let f = front_store[inst.name][alg][trial]
        f[vec(all(isfinite, f; dims=2)), :]
    end
    for inst  in INSTANCES
    for alg   in ALG_NAMES
    for trial in 1:N_TRIALS
    if size(front_store[inst.name][alg][trial], 1) > 0
]...)

if size(all_finite_pts, 1) == 0
    error("No finite points found across any front — cannot compute reference point")
end

# Normalise objectives to [0,1] using global ideal/nadir so all three axes
# contribute equally to the hypervolume.
const GLOBAL_IDEAL = vec(minimum(all_finite_pts; dims=1))
const GLOBAL_NADIR = vec(maximum(all_finite_pts; dims=1))
const GLOBAL_RANGE = max.(GLOBAL_NADIR .- GLOBAL_IDEAL, 1e-10)
const GLOBAL_REF   = ones(3) .* 1.1   # fixed reference in normalised [0,1]³ space

@info "Normalisation bounds" GLOBAL_IDEAL GLOBAL_NADIR
@info "Normalised reference point" GLOBAL_REF

function normalise_front(front::Matrix{Float64}) :: Matrix{Float64}
    return (front .- GLOBAL_IDEAL') ./ GLOBAL_RANGE'
end

# ── Phase 3: compute HV on normalised fronts using fixed [1.1,1.1,1.1] ref ───
hv_results = Dict(inst.name => Dict(a => Float64[] for a in ALG_NAMES) for inst in INSTANCES)

for inst in INSTANCES
    for trial in 1:N_TRIALS
        for alg in ALG_NAMES
            front     = front_store[inst.name][alg][trial]
            front_fin = front[vec(all(isfinite, front; dims=2)), :]
            front_fin = unique(front_fin, dims=1)
            hv = if size(front_fin, 1) == 0
                0.0
            else
                compute_hypervolume(normalise_front(front_fin), GLOBAL_REF)
            end
            push!(hv_results[inst.name][alg], hv)
            @info "  $alg | $(inst.name) | trial $trial → HV = $hv"
        end
    end
end

# ── Save normalised Pareto front points from every trial ─────────────────────
front_rows = NamedTuple{(:instance, :algorithm, :trial,
                         :f1_dv_norm, :f2_unassigned_norm, :f3_vehicles_norm,
                         :f1_dv, :f2_unassigned_time, :f3_vehicles),
                        NTuple{9, Any}}[]
for inst in INSTANCES
    for alg in ALG_NAMES
        for trial in 1:N_TRIALS
            front = front_store[inst.name][alg][trial]
            front_fin = front[vec(all(isfinite, front; dims=2)), :]
            front_fin = unique(front_fin, dims=1)
            size(front_fin, 1) == 0 && continue
            norm = normalise_front(front_fin)
            for i in 1:size(front_fin, 1)
                push!(front_rows, (
                    instance           = inst.name,
                    algorithm          = alg,
                    trial              = trial,
                    f1_dv_norm         = norm[i, 1],
                    f2_unassigned_norm = norm[i, 2],
                    f3_vehicles_norm   = norm[i, 3],
                    f1_dv              = front_fin[i, 1],
                    f2_unassigned_time = front_fin[i, 2],
                    f3_vehicles        = front_fin[i, 3],
                ))
            end
        end
    end
end

# ═════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═════════════════════════════════════════════════════════════════════════════

function _median_iqr(v::Vector{Float64})
    isempty(v) && return (0.0, 0.0)
    s = sort(v)
    n = length(s)
    med = n % 2 == 0 ? (s[n÷2] + s[n÷2+1]) / 2.0 : s[(n+1)÷2]
    q1  = s[max(1, n÷4)]
    q3  = s[min(n, (3n)÷4 + 1)]
    return (med, q3 - q1)
end

println()
@printf "\nEval budget: %d  |  %d demands  |  %d trials per instance\n" BUDGET_EVALS N_DEMANDS N_TRIALS
for inst in INSTANCES
    @printf "\nInstance: %s\n" inst.name
    println("  " * "─"^68)
    @printf "  %-12s  %16s  %12s  %10s\n" "Algorithm" "Median HV" "IQR" "Median [s]"
    println("  " * "─"^68)
    for alg in ALG_NAMES
        med_hv, iqr_hv = _median_iqr(hv_results[inst.name][alg])
        med_t,  _       = _median_iqr(time_results[inst.name][alg])
        @printf "  %-12s  %16.4e  %12.2e  %10.1f\n" alg med_hv iqr_hv med_t
    end
end
println()

# ═════════════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

rows = NamedTuple{(:instance, :budget_evals, :algorithm, :trial, :hypervolume, :elapsed),
                  NTuple{6, Any}}[]

for inst in INSTANCES
    for alg in ALG_NAMES
        for trial in eachindex(hv_results[inst.name][alg])
            push!(rows, (instance     = inst.name,
                         budget_evals = BUDGET_EVALS,
                         algorithm    = alg,
                         trial        = trial,
                         hypervolume  = hv_results[inst.name][alg][trial],
                         elapsed      = time_results[inst.name][alg][trial]))
        end
    end
end

outdir   = joinpath(@__DIR__, "outputs")
mkpath(outdir)
csv_path = joinpath(outdir, "numerical_experiments.csv")
CSV.write(csv_path, DataFrame(rows))
@info "Results saved" csv_path

fronts_csv = joinpath(outdir, "pareto_fronts.csv")
CSV.write(fronts_csv, DataFrame(front_rows))
@info "Pareto front points saved" fronts_csv
