# fuel_cost_calc/adaptive_grid.jl
# Adaptive Mesh Refinement for the cost table
#
# Two-phase approach:
#   Phase 1 (fast):  Analyse gradient on existing data, build adaptive grid
#                     structure. Missing cells are NaN.
#   Phase 2 (opt):   Batch-compute missing costs for chosen pairs.
#
# Usage:
#   base_ct = load_cost_table()
#   ag = gen_adaptive_cost_table(load_sim(), base_ct, target_dv=50.0)
#   snap_cost(ag, from, to, dep, arr)  # nearest-finite-neighbour lookup

using JLD2
using ProgressMeter
using Printf

const ADAPTIVE_GRID_PATH = "outputs/cost_table_adaptive.jld2"
const MIN_GRID_STEP = 1.0      # days — never split below this
const MAX_GRID_STEP = Inf       # days — no coarsening cap (low-gradient regions merge freely)
const DV_THRESHOLD  = 1e7

# ── Data structures ──────────────────────────────────────────────────────────

mutable struct AdaptivePair
    from_idx::Int
    to_idx::Int
    deps::Vector{Float64}
    arrs::Vector{Float64}
    values::Matrix{Float64}
end

struct AdaptiveGrid
    pairs::Vector{AdaptivePair}
    lookup::Dict{Tuple{Int,Int}, Int}
end

grid_size(ag::AdaptiveGrid) = sum(p -> length(p.deps) * length(p.arrs), ag.pairs)

# ── Group entries by (from, to) — O(N) over the Dict ────────────────────────

function _group_by_pair(base_ct::Dict)
    groups = Dict{Tuple{Int,Int}, Vector{Tuple{Float64,Float64,Float64}}}()
    sizehint!(groups, 100000)
    for ((f, t, dep, arr), dv) in base_ct
        dv >= DV_THRESHOLD && continue
        push!(get!(groups, (f, t)) do
            Vector{Tuple{Float64,Float64,Float64}}()
        end, (dep, arr, dv))
    end
    return groups
end

# ── Build per-pair matrix from its pre-filtered entries ─────────────────────

function _build_matrix(deps, arrs, entries)
    n, m = length(deps), length(arrs)
    Z = fill(NaN, n, m)
    dl = Dict{Float64,Int}(d => i for (i, d) in enumerate(deps))
    al = Dict{Float64,Int}(a => j for (j, a) in enumerate(arrs))
    for (dep, arr, dv) in entries
        i, j = get(dl, dep, 0), get(al, arr, 0)
        i > 0 && j > 0 && (Z[i, j] = dv)
    end
    return Z
end

# ── Per-axis gradient (centered finite differences) ─────────────────────────

function _per_axis_gradients(Z, deps, arrs)
    n, m = size(Z)
    dd = fill(NaN, n, m)
    da = fill(NaN, n, m)
    for i in 2:n-1, j in 2:m-1
        isfinite(Z[i-1,j]) && isfinite(Z[i+1,j]) && isfinite(Z[i,j-1]) && isfinite(Z[i,j+1]) || continue
        dd[i,j] = (Z[i+1,j] - Z[i-1,j]) / (deps[i+1] - deps[i-1])
        da[i,j] = (Z[i,j+1] - Z[i,j-1]) / (arrs[j+1] - arrs[j-1])
    end
    return dd, da
end

# ── Build a single refined axis from gradient analysis ──────────────────────

function _refine_axis(coords, grad_matrix, target_dv, axis)
    n = length(coords)
    n <= 1 && return copy(coords)

    function max_grad_at(k)
        mg = 0.0
        if axis == :dep
            (2 <= k <= n-1) || return mg
            @inbounds for j in 1:size(grad_matrix, 2)
                v = grad_matrix[k, j]; isfinite(v) || continue
                mg = max(mg, abs(v))
            end
        else
            (2 <= k <= size(grad_matrix, 2)-1) || return mg
            @inbounds for i in 1:size(grad_matrix, 1)
                v = grad_matrix[i, k]; isfinite(v) || continue
                mg = max(mg, abs(v))
            end
        end
        return mg
    end

    any(k -> max_grad_at(k) > 0, 1:n) || return copy(coords)

    out = Float64[coords[1]]
    for k in 1:n-1
        step = coords[k+1] - coords[k]
        if step / 2 < MIN_GRID_STEP
            push!(out, coords[k+1]); continue
        end
        mg = max(max_grad_at(k), max_grad_at(k+1))
        dv = mg * step
        if dv > target_dv
            push!(out, (coords[k] + coords[k+1]) / 2)
            push!(out, coords[k+1])
        elseif dv < target_dv / 3 && step < MAX_GRID_STEP
            # coarsen: skip coords[k+1]
        else
            push!(out, coords[k+1])
        end
    end
    return out
end

# ── Build adaptive grid structure for one pair (no new cost computation) ─────

function _build_refined_grid(deps, arrs, entries, target_dv)
    Z = _build_matrix(deps, arrs, entries)
    dd, da = _per_axis_gradients(Z, deps, arrs)

    new_deps = _refine_axis(deps, dd, target_dv, :dep)
    new_arrs = _refine_axis(arrs, da, target_dv, :arr)

    # Build adaptive matrix (NaN for not-yet-computed points)
    Z_new = _build_matrix(new_deps, new_arrs, entries)

    return new_deps, new_arrs, Z_new
end

# ── Iterative multi-grid refinement (no new cost computation) ───────────────

function refine_grid_structure!(ag, from_idx, to_idx, entries, target_dv)
    p = ag.pairs[ag.lookup[(from_idx, to_idx)]]

    for phase in (200.0, 100.0, target_dv)
        for iter in 1:20
            nd, na, nz = _build_refined_grid(p.deps, p.arrs, entries, phase)
            if length(nd) == length(p.deps) && length(na) == length(p.arrs)
                break
            end
            p.deps, p.arrs, p.values = nd, na, nz
        end
    end
end

# ── Helper: wait for any of N tasks to finish ──────────────────────────────

function _waitforany(tasks::Vector{Task})
    while true
        for (i, t) in enumerate(tasks)
            istaskstarted(t) || continue
            if istaskdone(t)
                return i
            end
        end
        sleep(0.05)
    end
end

# ── Batch-fill missing costs with @spawn per (from, to) pair ────────────────

function compute_missing_costs!(ag::AdaptiveGrid, sim, base_ct::Dict;
                                 filter_depot_sat::Bool = true,
                                 max_concurrent::Int = Threads.nthreads() * 2,
                                 checkpoint_interval::Int = 50)

    # 1. Identify depot nodes (cache-friendly single pass)
    depot_set = filter_depot_sat ?
        Set{Int}(findall(n -> startswith(n, "depot"), sim.names)) : nothing

    # 2. Collect NaN keys grouped by (from, to) pair
    pair_keys = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Float64,Float64}}}()
    for p in ag.pairs
        filter_depot_sat && !(p.from_idx in depot_set || p.to_idx in depot_set) && continue
        keys = Tuple{Int,Int,Float64,Float64}[]
        for i in 1:length(p.deps), j in 1:length(p.arrs)
            isfinite(p.values[i,j]) || push!(keys, (p.from_idx, p.to_idx, p.deps[i], p.arrs[j]))
        end
        isempty(keys) || (pair_keys[(p.from_idx, p.to_idx)] = keys)
    end

    isempty(pair_keys) && return @info "No missing costs to compute"
    total_keys = sum(length, values(pair_keys))
    sorted = sort(collect(pair_keys); by=last, rev=true)
    n_pairs = length(sorted)

    @info "Phase 2: computing $total_keys missing entries across $n_pairs pairs" max_concurrent=max_concurrent

    # 3. Seed initial batch of @spawn tasks
    n_initial = min(max_concurrent, n_pairs)
    active_tasks = Vector{Task}(undef, n_initial)
    active_info = Vector{Tuple{Int,Int}}(undef, n_initial)
    pair_idx = 0
    for i in 1:n_initial
        pair_idx += 1
        (f, t), keys = sorted[pair_idx]
        active_tasks[i] = Threads.@spawn build_cost_table(sim, keys)
        active_info[i] = (f, t)
    end

    completed = 0
    next_save = checkpoint_interval
    t_start = time()

    while completed < n_pairs
        # Wait for any task to finish
        done_idx = _waitforany(active_tasks)
        vals = fetch(active_tasks[done_idx])
        f, t = active_info[done_idx]
        p = ag.pairs[ag.lookup[(f, t)]]

        # Merge into base_ct and update AdaptivePair.values in-place
        for ((ff, tt, dep, arr), dv) in vals
            base_ct[(ff, tt, dep, arr)] = dv
            i = something(findfirst(==(dep), p.deps), 0)
            j = something(findfirst(==(arr), p.arrs), 0)
            i > 0 && j > 0 && (p.values[i, j] = dv)
        end

        completed += 1

        # Checkpoint
        if completed >= next_save || completed == n_pairs
            elapsed = round(time() - t_start; digits=1)
            rate = elapsed > 0 ? completed / elapsed : 0.0
            @info "Checkpoint" completed total=n_pairs elapsed=elapsed rate="$(@sprintf("%.1f", rate)) pairs/s"
            grid_meta = [(pp.from_idx, pp.to_idx, pp.deps, pp.arrs) for pp in ag.pairs]
            jldsave(ADAPTIVE_GRID_PATH; d=base_ct, grid_meta=grid_meta)
            next_save += checkpoint_interval
        end

        # Launch next task (replace the completed slot)
        pair_idx += 1
        if pair_idx <= n_pairs
            (f, t), keys = sorted[pair_idx]
            active_tasks[done_idx] = Threads.@spawn build_cost_table(sim, keys)
            active_info[done_idx] = (f, t)
        elseif length(active_tasks) > 1
            deleteat!(active_tasks, done_idx)
            deleteat!(active_info, done_idx)
        end
    end

    total_elapsed = round(time() - t_start; digits=1)
    @info "All missing costs computed" total=total_keys elapsed=total_elapsed
    return total_keys
end

# ── Main entry: generate adaptive cost table ────────────────────────────────

function gen_adaptive_cost_table(sim, base_ct::Dict; target_dv::Float64=50.0,
                                 compute_costs::Bool=false)
    if isfile(ADAPTIVE_GRID_PATH)
        return load_adaptive_cost_table()
    end

    @info "Phase 1: grouping $(length(base_ct)) entries..."
    t0 = time()
    groups = _group_by_pair(base_ct)
    @info "  $(length(groups)) unique pairs ($(round(time()-t0; digits=2))s)"

    @info "Phase 2: building adaptive grid structures (target_dv=$target_dv)..."
    t1 = time()
    ag = AdaptiveGrid([], Dict{Tuple{Int,Int}, Int}())

    prog = Progress(length(groups); desc="Analysing gradients: ", barlen=40, showspeed=true)
    for ((f, t), entries) in groups
        deps_p = sort(unique(Float64[e[1] for e in entries]))
        arrs_p = sort(unique(Float64[e[2] for e in entries]))
        Z_init = _build_matrix(deps_p, arrs_p, entries)
        push!(ag.pairs, AdaptivePair(f, t, deps_p, arrs_p, Z_init))
        ag.lookup[(f,t)] = length(ag.pairs)
        refine_grid_structure!(ag, f, t, entries, target_dv)
        next!(prog)
    end
    finish!(prog)
    @info "  Adaptive grid structure built ($(round(time()-t1; digits=2))s, $(grid_size(ag)) cells total)"

    # Save refined grid coordinates + augmented Dict
    @info "Saving adaptive cost table to $ADAPTIVE_GRID_PATH"
    grid_meta = [(p.from_idx, p.to_idx, p.deps, p.arrs) for p in ag.pairs]
    jldsave(ADAPTIVE_GRID_PATH; d=base_ct, grid_meta=grid_meta)

    if compute_costs
        @info "Phase 3: computing missing costs..."
        compute_missing_costs!(ag, sim, base_ct)
        jldsave(ADAPTIVE_GRID_PATH; d=base_ct, grid_meta=grid_meta)
    end

    return ag
end

# ── Load saved adaptive table ──────────────────────────────────────────────

function load_adaptive_cost_table()
    isfile(ADAPTIVE_GRID_PATH) || error("No adaptive cost table at $ADAPTIVE_GRID_PATH")
    local d, grid_meta
    @load ADAPTIVE_GRID_PATH d grid_meta
    @info "Loaded adaptive cost table" n_entries=length(d) n_pairs=length(grid_meta)

    ag = AdaptiveGrid([], Dict{Tuple{Int,Int}, Int}())
    groups = _group_by_pair(d)

    for (f, t, deps, arrs) in grid_meta
        entries = get(groups, (f, t), Tuple{Float64,Float64,Float64}[])
        Z = _build_matrix(deps, arrs, entries)
        push!(ag.pairs, AdaptivePair(f, t, deps, arrs, Z))
        ag.lookup[(f,t)] = length(ag.pairs)
    end

    return ag
end

# ── Fast nearest-neighbour lookup ───────────────────────────────────────────

function snap_cost(ag::AdaptiveGrid, from_idx, to_idx, dep_epoch, arr_epoch)
    idx = get(ag.lookup, (from_idx, to_idx), nothing)
    idx === nothing && return Inf

    p = ag.pairs[idx]
    n, m = length(p.deps), length(p.arrs)

    i = clamp(searchsortedlast(p.deps, dep_epoch), 1, n-1)
    j = clamp(searchsortedlast(p.arrs, arr_epoch), 1, m-1)

    best = Inf
    best_d = Inf
    for di in (i, i+1), dj in (j, j+1)
        v = p.values[di, dj]
        isfinite(v) || continue
        d2 = (p.deps[di] - dep_epoch)^2 + (p.arrs[dj] - arr_epoch)^2
        if d2 < best_d
            best = v; best_d = d2
        end
    end

    # Fallback: expand search if none of 4 corners are finite
    if !isfinite(best)
        radius = 1
        while radius < max(n, m)
            for di in max(1, i-radius):min(n, i+radius)
                for dj in max(1, j-radius):min(m, j+radius)
                    v = p.values[di, dj]
                    isfinite(v) || continue
                    d2 = (p.deps[di] - dep_epoch)^2 + (p.arrs[dj] - arr_epoch)^2
                    if d2 < best_d
                        best = v; best_d = d2
                    end
                end
            end
            isfinite(best) && return best
            radius += 1
        end
    end

    return best
end

# ── Dict-style access ───────────────────────────────────────────────────────

Base.getindex(ag::AdaptiveGrid, key::Tuple{Int,Int,Float64,Float64}) = snap_cost(ag, key...)
Base.haskey(ag::AdaptiveGrid, key::Tuple{Int,Int,Float64,Float64}) = haskey(ag.lookup, (key[1], key[2]))
