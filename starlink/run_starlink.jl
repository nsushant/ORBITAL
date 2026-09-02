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

# Cost-table fidelity: true → Section 4.3 continuous-thrust NLP (per plane-pair
# spiral), false → Section 4.2 impulsive estimate. fmax in km/s² (3.5e-6 = paper).
const CONTINUOUS_THRUST = false
const FMAX_KMS2         = 3.5e-6

# ── Step 1: fetch TLEs and sample 200 satellites ──────────────────────────────

sats, launch_dates, sat_values = fetch_and_sample(n_targets=200, seed=42)
sat_names = [s.name for s in sats if !startswith(s.name, "depot")]

# ── Step 2: generate age-weighted demands ─────────────────────────────────────

const NUMEXP_MAX_DEADLINE = 365.0   # days — max time_dist across numerical experiment scenarios

demands = generate_starlink_demands(sat_names, launch_dates, sat_values; n_demands=200, seed=42)
max_deadline = maximum(demands["demand_deadlines"])   # days
total_demand_value = sum(demands["asset_values"])
@info "Demand window" max_deadline_days=round(max_deadline, digits=1) total_demand_value=total_demand_value

# ── Step 3: simulate up to max_deadline ───────────────────────────────────────

t_end_days = max(1.5 * max_deadline, 1.5 * NUMEXP_MAX_DEADLINE)

sim_params = Dict(
    "J2"    => true,
    "dt"    => 60.0,
    "t_end" => t_end_days * 86400.0,
)
sim = gen_simulation_from_sats(sats, sim_params)

# ── Step 4: compute cost table (monthly resolution, AMR refines further) ──────
base_ct    = gen_cost_table(sim; tof_step=30.0, t_end=t_end_days,
                            continuous=CONTINUOUS_THRUST, fmax=FMAX_KMS2)
mintof_tab = build_min_tof_table()

# ── Step 5: adaptive mesh refinement ─────────────────────────────────────────

@info "Running adaptive mesh refinement …"
adap_ct = gen_adaptive_cost_table(sim, base_ct; target_dv=50.0, compute_costs=true,
                                  continuous=CONTINUOUS_THRUST, fmax=FMAX_KMS2)

# ── Step 6: run all algorithms (N_TRIALS_SL trials) ──────────────────────────

min_dv_tab = build_min_dv_table(base_ct, length(sim.names))

function _hv(front, ref)
    @assert size(front, 2) == 3
    moocore.hypervolume(front, ref=ref, maximise=false)
end
function _norm(front, ideal, rng)
    (front .- ideal') ./ rng'
end

# front_store: 4-column matrix [f1, f2_value, f3, n_unassigned] per (trial, alg)
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
        front_3col, t = results[alg]   # [f1, f2, f3] matrix from algorithm
        front_fin = front_3col[vec(all(isfinite, front_3col; dims=2)), :]
        # Store as 4-column: append n_unassigned derived from f2 and total_demand_value
        # n_unassigned is embedded per-individual by run_all_algorithms via SchedIndividual
        # front here is already [f1, f2_value, f3, n_unassigned]
        front_store[(trial, alg)] = front_fin
        time_store[(trial, alg)]  = t
    end
    @info "Trial $trial/$N_TRIALS_SL done"
end

# ── Normalise using only objective cols 1:3 ───────────────────────────────────

n_obj_cols = min(3, size(first(values(front_store)), 2))

all_pts = vcat([front_store[(t, a)][:, 1:n_obj_cols]
                for t in 1:N_TRIALS_SL, a in ALG_NAMES_SL
                if size(front_store[(t,a)], 1) > 0]...)

sl_ideal = vec(minimum(all_pts; dims=1))
sl_nadir = vec(maximum(all_pts; dims=1))
sl_range = max.(sl_nadir .- sl_ideal, 1e-10)
sl_ref   = ones(3) .* 1.1
@info "Normalisation" sl_ideal sl_nadir

# ── Build output CSVs ─────────────────────────────────────────────────────────

alg_col  = String[]; trial_col = Int[]; hv_col = Float64[]; elapsed_col = Float64[]
pf_alg   = String[]; pf_trial  = Int[]
pf_f1n   = Float64[]; pf_f2n = Float64[]; pf_f3n = Float64[]
pf_f1    = Float64[]; pf_f2  = Float64[]; pf_f3  = Float64[]
pf_nserv = Int[];     pf_valrec = Float64[]

for trial in 1:N_TRIALS_SL, alg in ALG_NAMES_SL
    front    = front_store[(trial, alg)]
    front_3  = front[:, 1:3]
    front_u  = size(front_3,1) > 0 ? unique(front_3, dims=1) : front_3
    hv = size(front_u,1) == 0 ? 0.0 :
         _hv(_norm(front_u, sl_ideal, sl_range), sl_ref)

    push!(alg_col, alg); push!(trial_col, trial)
    push!(hv_col, hv);   push!(elapsed_col, time_store[(trial,alg)])

    if size(front_u,1) > 0
        norm = _norm(front_u, sl_ideal, sl_range)
        # Rebuild mapping from unique front_3 rows back to full 4-col front
        for i in 1:size(front_u,1)
            f1v = front_u[i,1]; f2v = front_u[i,2]; f3v = front_u[i,3]
            # find matching row in full front for n_unassigned
            row_idx = findfirst(j -> front[j,1] ≈ f1v && front[j,2] ≈ f2v && front[j,3] ≈ f3v,
                                1:size(front,1))
            n_unas   = (size(front,2) >= 4 && row_idx !== nothing) ? Int(front[row_idx,4]) : 0
            n_serv   = length(sat_names) - n_unas
            val_rec  = total_demand_value - f2v

            push!(pf_alg, alg); push!(pf_trial, trial)
            push!(pf_f1n, norm[i,1]); push!(pf_f2n, norm[i,2]); push!(pf_f3n, norm[i,3])
            push!(pf_f1, f1v); push!(pf_f2, f2v); push!(pf_f3, f3v)
            push!(pf_nserv, n_serv); push!(pf_valrec, val_rec)
        end
    end
end

mkpath("outputs")

CSV.write("outputs/starlink_hv.csv",
    DataFrame(algorithm=alg_col, trial=trial_col,
              hypervolume=hv_col, elapsed=elapsed_col))

CSV.write("outputs/starlink_pareto_fronts.csv",
    DataFrame(algorithm=pf_alg, trial=pf_trial,
              f1_dv_norm=pf_f1n, f2_unrecovered_norm=pf_f2n, f3_vehicles_norm=pf_f3n,
              f1_dv=pf_f1, f2_unrecovered_value=pf_f2, f3_vehicles=pf_f3,
              n_serviced=pf_nserv, value_recovered=pf_valrec))

@info "Outputs saved"
@printf "\n  outputs/starlink_hv.csv\n"
@printf "  outputs/starlink_pareto_fronts.csv\n"
