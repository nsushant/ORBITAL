# starlink/run_starlink.jl
# End-to-end Starlink case study driver.
#
# Run: julia --project=. -t auto starlink/run_starlink.jl

using CSV, DataFrames, Printf, PyCall
const moocore = pyimport("moocore")

const NUMEXP_INCLUDE = true
include(joinpath(@__DIR__, "..", "GATests.jl"))
include(joinpath(@__DIR__, "fetch_and_sample.jl"))
include(joinpath(@__DIR__, "generate_starlink_demands.jl"))

# ── Settings ─────────────────────────────────────────────────────────────────

const N_TRIALS_SL  = 5
const ALG_NAMES_SL = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]

# ── Step 1: fetch TLEs and sample 200 satellites ──────────────────────────────

sats, launch_dates = fetch_and_sample(n_targets=200, seed=42)
sat_names = [s.name for s in sats if !startswith(s.name, "depot")]

# ── Step 2: generate age-weighted demands ─────────────────────────────────────

const NUMEXP_MAX_DEADLINE = 365.0   # days — max time_dist across numerical experiment scenarios

demands = generate_starlink_demands(sat_names, launch_dates; n_demands=200, seed=42)
max_deadline = maximum(demands["demand_deadlines"])   # days
@info "Demand window" max_deadline_days=round(max_deadline, digits=1)

# ── Step 3: simulate up to max_deadline ───────────────────────────────────────

t_end_days = max(1.5 * max_deadline, 1.5 * NUMEXP_MAX_DEADLINE)

sim_params = Dict(
    "J2"    => true,
    "dt"    => 60.0,                            # 1-minute integrator step
    "t_end" => t_end_days * 86400.0,
)
sim = gen_simulation_from_sats(sats, sim_params)

# ── Step 4: compute cost table (monthly resolution, AMR refines further) ──────
base_ct    = gen_cost_table(sim; tof_step=30.0, t_end=t_end_days)
mintof_tab = build_min_tof_table()

# ── Step 5: adaptive mesh refinement ─────────────────────────────────────────

@info "Running adaptive mesh refinement …"
adap_ct = gen_adaptive_cost_table(sim, base_ct; target_dv=50.0, compute_costs=true)

# ── Step 6: run all algorithms (N_TRIALS_SL trials) ──────────────────────────

min_dv_tab = build_min_dv_table(base_ct, length(sim.names))

# Normalisation helpers
function _hv(front, ref)
    @assert size(front, 2) == 3
    moocore.hypervolume(front, ref=ref, maximise=false)
end
function _norm(front, ideal, rng)
    (front .- ideal') ./ rng'
end

front_store = Dict{Tuple{Int,String}, Matrix{Float64}}()
time_store  = Dict{Tuple{Int,String}, Float64}()

for trial in 1:N_TRIALS_SL
    seed_t = trial * 137 + 999
    trial_demands = Dict{String,Any}(k => v for (k,v) in demands)
    trial_demands["seed"] = seed_t

    ctx = make_context(trial_demands, sim, base_ct, mintof_tab, min_dv_tab; nvehicles=20)
    init_copies = ntuple(_ -> (copy_schedule(ctx.init_sched), ctx.init_unas), 4)
    results = run_all_algorithms(ctx; budget_evals=BUDGET_EVALS, init_copies=init_copies)

    for alg in ALG_NAMES_SL
        front, t = results[alg]
        front_fin = front[vec(all(isfinite, front; dims=2)), :]
        front_store[(trial, alg)] = front_fin
        time_store[(trial, alg)]  = t
    end
    @info "Trial $trial/$N_TRIALS_SL done"
end

# ── Normalise and compute HV ──────────────────────────────────────────────────

all_pts = vcat([front_store[(t, a)]
                for t in 1:N_TRIALS_SL, a in ALG_NAMES_SL
                if size(front_store[(t,a)], 1) > 0]...)

sl_ideal = vec(minimum(all_pts; dims=1))
sl_nadir = vec(maximum(all_pts; dims=1))
sl_range = max.(sl_nadir .- sl_ideal, 1e-10)
sl_ref   = ones(3) .* 1.1
@info "Normalisation" sl_ideal sl_nadir

# ── Build output CSVs ─────────────────────────────────────────────────────────

alg_col  = String[]; trial_col = Int[]; hv_col = Float64[]; elapsed_col = Float64[]
pf_alg   = String[]; pf_trial = Int[]
pf_f1n   = Float64[]; pf_f2n = Float64[]; pf_f3n = Float64[]
pf_f1    = Float64[]; pf_f2  = Float64[]; pf_f3  = Float64[]

for trial in 1:N_TRIALS_SL, alg in ALG_NAMES_SL
    front = front_store[(trial, alg)]
    front_u = size(front,1) > 0 ? unique(front, dims=1) : front
    hv = size(front_u,1) == 0 ? 0.0 :
         _hv(_norm(front_u, sl_ideal, sl_range), sl_ref)

    push!(alg_col, alg); push!(trial_col, trial)
    push!(hv_col, hv);   push!(elapsed_col, time_store[(trial,alg)])

    if size(front_u,1) > 0
        norm = _norm(front_u, sl_ideal, sl_range)
        for i in 1:size(front_u,1)
            push!(pf_alg, alg); push!(pf_trial, trial)
            push!(pf_f1n, norm[i,1]); push!(pf_f2n, norm[i,2]); push!(pf_f3n, norm[i,3])
            push!(pf_f1, front_u[i,1]); push!(pf_f2, front_u[i,2]); push!(pf_f3, front_u[i,3])
        end
    end
end

mkpath("outputs")

CSV.write("outputs/starlink_hv.csv",
    DataFrame(algorithm=alg_col, trial=trial_col,
              hypervolume=hv_col, elapsed=elapsed_col))

CSV.write("outputs/starlink_pareto_fronts.csv",
    DataFrame(algorithm=pf_alg, trial=pf_trial,
              f1_dv_norm=pf_f1n, f2_unassigned_norm=pf_f2n, f3_vehicles_norm=pf_f3n,
              f1_dv=pf_f1, f2_unassigned_time=pf_f2, f3_vehicles=pf_f3))

@info "Outputs saved"
@printf "\n  outputs/starlink_hv.csv\n"
@printf "  outputs/starlink_pareto_fronts.csv\n"
