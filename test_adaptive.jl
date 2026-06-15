# test_adaptive.jl
# Test the adaptive mesh refinement pipeline
#
# Phase 1 (fast ~16s): gradient analysis → adaptive grid structure
# Phase 2 (opt):        compute missing costs for depot↔sat pairs

include("sim/propagator.jl")
include("sol_utils.jl")
include("fuel_cost_calc/adaptive_grid.jl")
include("plots/gradients.jl")

sim = load_sim()
base_ct = load_cost_table()

@info "=== Adaptive Mesh Refinement Test ==="
@info "Base cost table: $(length(base_ct)) entries"

# ── Phase 1: Build adaptive grid from gradient analysis ────────────────────
@info "\n--- Phase 1: gradient analysis (target_dv = 200 m/s) ---"
t0 = time()
ag = gen_adaptive_cost_table(sim, deepcopy(base_ct), target_dv=200.0, compute_costs=false)
t1 = time() - t0
@info "Done in $(round(t1; digits=2))s" grid_size=grid_size(ag) n_pairs=length(ag.pairs)

# Grid stats
n_nan    = sum(p -> count(!isfinite, p.values), ag.pairs)
n_finite = grid_size(ag) - n_nan
n_refined   = count(p -> length(p.deps) > 26 || length(p.arrs) > 26, ag.pairs)
n_coarsened = count(p -> length(p.deps) < 26 || length(p.arrs) < 26, ag.pairs)
@info "Grid cells" total=grid_size(ag) finite=n_finite nan=n_nan
@info "Pairs" total=length(ag.pairs) refined=n_refined coarsened=n_coarsened

# ── Lookup smoke test (depot → sat) ────────────────────────────────────────
di = findfirst(n -> startswith(n, "depot"), sim.names)
si = findfirst(n -> startswith(n, "sat"),   sim.names)

if di !== nothing && si !== nothing
    dep_name = sim.names[di]; sat_name = sim.names[si]
    @info "\n--- Lookup: $dep_name ($di) → $sat_name ($si) ---"
    for (dep, arr) in [(15.0, 30.0), (30.0, 60.0), (45.0, 90.0), (100.0, 150.0)]
        vo = snap_cost(base_ct, di, si, dep, arr)
        va = snap_cost(ag, di, si, dep, arr)
        ok = isapprox(vo, va; rtol=0.05) ? "✓" : "✗"
        println("  snap($dep,$arr): orig=$vo  adaptive=$va  $ok")
    end
end

# ── Per-pair grid summary ──────────────────────────────────────────────────
println("\n--- Sample pair grid sizes ---")
for k in 1:min(8, length(ag.pairs))
    p = ag.pairs[k]
    print("  ($(p.from_idx),$(p.to_idx)): $(length(p.deps))d × $(length(p.arrs))a")
    println(" = $(length(p.deps)*length(p.arrs))  (NaN: $(count(!isfinite, p.values)))")
end

# ── Plots ──────────────────────────────────────────────────────────────────
if di !== nothing && si !== nothing
    dep_name = sim.names[di]; sat_name = sim.names[si]
    @info "\n--- Plots ---"

    fig1 = plot_gradient(dep_name, sat_name, base_ct, sim)
    save("gradient_$(dep_name)_$(sat_name).png", fig1)
    @info "  gradient: saved"

    fig2 = plot_refinement_grid(dep_name, sat_name, ag, sim)
    save("adaptive_grid_$(dep_name)_$(sat_name).png", fig2)
    @info "  adaptive grid overlay: saved"

    fig3 = plot_refinement_comparison(dep_name, sat_name, base_ct, ag, sim)
    save("adaptive_comparison_$(dep_name)_$(sat_name).png", fig3)
    @info "  comparison (uniform vs adaptive): saved"
end

# ── Optional: compute missing costs for depot→sat pairs ────────────────────
# Uncomment to run Phase 2 (orbital mechanics — may take hours)
# @info "\n--- Phase 2: computing missing costs for depot↔sat pairs ---"
# n = compute_missing_costs!(ag, sim, base_ct; filter_depot_sat=true)
# @info "Computed $n new entries"

@info "\n=== Test complete ==="
