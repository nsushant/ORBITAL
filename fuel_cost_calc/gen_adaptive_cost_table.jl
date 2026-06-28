# fuel_cost_calc/gen_adaptive_cost_table.jl
# Generate the adaptive mesh refined cost table.
#
# Usage:
#   julia fuel_cost_calc/gen_adaptive_cost_table.jl
#
# Tunables — edit consts below:

const TARGET_DV = 50.0           # m/s — gradient threshold for refinement
const MAX_CONCURRENT = Threads.nthreads()
const CHECKPOINT_INTERVAL = 50
const RUN_PHASE2 = true          # set false for gradient-only dry run
const RESUME = false             # set true to reload cached and skip Phase 1
const PLOT_PAIR = ("depot_1", "sat_400")

# ── Dependencies ─────────────────────────────────────────────────────

include(joinpath(@__DIR__, "..", "sim", "propagator.jl"))
include(joinpath(@__DIR__, "..", "sol_utils.jl"))
include(joinpath(@__DIR__, "..", "plots", "gradients.jl"))
include(joinpath(@__DIR__, "adaptive_grid.jl"))
include(joinpath(@__DIR__, "gen_cost_table.jl"))

# ── Main ─────────────────────────────────────────────────────────────

sim = load_sim()
base_ct = load_cost_table()
@info "Loaded" sim=length(sim.names) cost_table=length(base_ct)

if RESUME && isfile(ADAPTIVE_GRID_PATH)
    @info "Resuming from cached $ADAPTIVE_GRID_PATH"
    ag = load_adaptive_cost_table()
else
    RESUME || (isfile(ADAPTIVE_GRID_PATH) && (rm(ADAPTIVE_GRID_PATH); @info "Removed old cache"))

    # ── Phase 1: gradient analysis → adaptive grid structure ──────────
    @info "Phase 1: building adaptive grid (target_dv = $TARGET_DV)"
    t0 = time()
    ag = gen_adaptive_cost_table(sim, base_ct, target_dv=TARGET_DV, compute_costs=false)
    @info "Phase 1 done" elapsed="$(@sprintf("%.1f", time()-t0))s" grid_size=grid_size(ag)
end

# ── Comparison plot: uniform vs AMR grid structure ──────────────────

if haskey(sim.id_to_idx, PLOT_PAIR[1]) && haskey(sim.id_to_idx, PLOT_PAIR[2])
    fig = plot_grid_comparison(PLOT_PAIR[1], PLOT_PAIR[2], base_ct, ag, sim)
    outpath = joinpath(@__DIR__, "..", "plots",
                       "amr_grid_comparison_$(PLOT_PAIR[1])_$(PLOT_PAIR[2]).png")
    save(outpath, fig)
    @info "Saved comparison plot: $outpath"
else
    @warn "Plot pair $PLOT_PAIR not found in simulation, skipping plot"
end

# ── Phase 2: compute missing costs (orbital mechanics) ──────────────

if RUN_PHASE2
    @info "Phase 2: computing missing costs"
    compute_missing_costs!(ag, sim, base_ct;
        filter_depot_sat=true,
        max_concurrent=MAX_CONCURRENT,
        checkpoint_interval=CHECKPOINT_INTERVAL)
end

# ── Summary ──────────────────────────────────────────────────────────

n_nan = sum(p -> count(!isfinite, p.values), ag.pairs)
n_finite = grid_size(ag) - n_nan
@info "=== Summary ==="
@info "Grid cells" total=grid_size(ag) finite=n_finite nan=n_nan
@info "Output: $ADAPTIVE_GRID_PATH"
