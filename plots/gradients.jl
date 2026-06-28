# plots/gradients.jl
# plot_gradient — heatmap of |∇(ΔV)| over departure/arrival epochs

using CairoMakie

function _gradient_magnitude(Z, deps, arrs)
    n, m = size(Z)
    G = fill(NaN, n, m)

    for i in 2:n-1, j in 2:m-1
        isfinite(Z[i-1, j]) && isfinite(Z[i+1, j]) &&
        isfinite(Z[i, j-1]) && isfinite(Z[i, j+1]) || continue

        df_ddep = (Z[i+1, j] - Z[i-1, j]) / (deps[i+1] - deps[i-1])
        df_darr = (Z[i, j+1] - Z[i, j-1]) / (arrs[j+1] - arrs[j-1])
        G[i, j] = sqrt(df_ddep^2 + df_darr^2)
    end

    return G
end

function plot_gradient(name1::String, name2::String, cost_table::Dict, sim)

    haskey(sim.id_to_idx, name1) || error("\"$name1\" not found in simulation")
    haskey(sim.id_to_idx, name2) || error("\"$name2\" not found in simulation")

    idx1 = sim.id_to_idx[name1]
    idx2 = sim.id_to_idx[name2]

    matching = [(k, v) for (k, v) in cost_table if k[1] == idx1 && k[2] == idx2]
    isempty(matching) && error("No cost table entries for $name1 → $name2")

    deps = sort(unique(Float64[k[3] for (k, _) in matching]))
    arrs = sort(unique(Float64[k[4] for (k, _) in matching]))

    dep_idx = Dict(d => i for (i, d) in enumerate(deps))
    arr_idx = Dict(a => j for (j, a) in enumerate(arrs))

    Z = fill(NaN, length(deps), length(arrs))
    for (k, v) in matching
        v < 1e7 && (Z[dep_idx[k[3]], arr_idx[k[4]]] = v)
    end

    G = _gradient_magnitude(Z, deps, arrs)

    finite_vals = filter(isfinite, vec(G))
    clims = isempty(finite_vals) ? (0.0, 1.0) : (minimum(finite_vals), maximum(finite_vals))

    fig = Figure(size = (860, 620))
    ax  = Axis(fig[1, 1];
               xlabel = "Departure [days]",
               ylabel = "Arrival [days]",
               title  = "|∇(ΔV)| [m/s/day]  |  $name1 → $name2")
    hm  = heatmap!(ax, deps, arrs, G;
                   colormap   = :viridis,
                   nan_color  = :lightgray,
                   colorrange = clims)
    Colorbar(fig[1, 2], hm; label = "|∇(ΔV)| [m/s/day]")

    return fig
end

function plot_refinement_grid(name1::String, name2::String, ag, sim)

    haskey(sim.id_to_idx, name1) || error("\"$name1\" not found in simulation")
    haskey(sim.id_to_idx, name2) || error("\"$name2\" not found in simulation")

    idx1 = sim.id_to_idx[name1]
    idx2 = sim.id_to_idx[name2]

    haskey(ag.lookup, (idx1, idx2)) || error("No adaptive data for $name1 → $name2")
    p = ag.pairs[ag.lookup[(idx1, idx2)]]

    G = _gradient_magnitude(p.values, p.deps, p.arrs)

    finite_vals = filter(isfinite, vec(G))
    clims = isempty(finite_vals) ? (0.0, 1.0) : (minimum(finite_vals), maximum(finite_vals))

    fig = Figure(size = (860, 620))
    ax  = Axis(fig[1, 1];
               xlabel = "Departure [days]",
               ylabel = "Arrival [days]",
               title  = "Adaptive grid  |  $name1 → $name2")

    hm = heatmap!(ax, p.deps, p.arrs, G;
                  colormap   = :viridis,
                  nan_color  = :lightgray,
                  colorrange = clims)

    for dep in p.deps
        vlines!(ax, dep; color=:white, linewidth=0.5, alpha=0.3)
    end
    for arr in p.arrs
        hlines!(ax, arr; color=:white, linewidth=0.5, alpha=0.3)
    end

    Colorbar(fig[1, 2], hm; label = "|∇(ΔV)| [m/s/day]")

    return fig
end

function plot_refinement_comparison(name1::String, name2::String, base_ct::Dict,
                                    ag, sim)

    haskey(sim.id_to_idx, name1) || error("\"$name1\" not found in simulation")
    haskey(sim.id_to_idx, name2) || error("\"$name2\" not found in simulation")

    idx1 = sim.id_to_idx[name1]
    idx2 = sim.id_to_idx[name2]

    # ── Uniform panel ────────────────────────────────────────────────────────
    matching = [(k, v) for (k, v) in base_ct if k[1] == idx1 && k[2] == idx2]
    isempty(matching) && error("No cost table entries for $name1 → $name2")

    deps_u = sort(unique(Float64[k[3] for (k, _) in matching]))
    arrs_u = sort(unique(Float64[k[4] for (k, _) in matching]))
    dep_idx_u = Dict(d => i for (i, d) in enumerate(deps_u))
    arr_idx_u = Dict(a => j for (j, a) in enumerate(arrs_u))
    Z_u = fill(NaN, length(deps_u), length(arrs_u))
    for (k, v) in matching
        v < 1e7 && (Z_u[dep_idx_u[k[3]], arr_idx_u[k[4]]] = v)
    end
    G_u = _gradient_magnitude(Z_u, deps_u, arrs_u)

    # ── Adaptive panel ───────────────────────────────────────────────────────
    haskey(ag.lookup, (idx1, idx2)) || error("No adaptive data for $name1 → $name2")
    p = ag.pairs[ag.lookup[(idx1, idx2)]]
    G_a = _gradient_magnitude(p.values, p.deps, p.arrs)

    # ── Common colour range ──────────────────────────────────────────────────
    all_finite = filter(isfinite, vcat(vec(G_u), vec(G_a)))
    clims = isempty(all_finite) ? (0.0, 1.0) : (minimum(all_finite), maximum(all_finite))

    fig = Figure(size = (1400, 550))

    # Panel 1: uniform grid gradient
    ax1 = Axis(fig[1, 1];
               xlabel = "Departure [days]",
               ylabel = "Arrival [days]",
               title  = "|∇(ΔV)| — uniform 15d grid",
               titlesize = 12)
    hm1 = heatmap!(ax1, deps_u, arrs_u, G_u;
                   colormap = :viridis, nan_color = :lightgray, colorrange = clims)

    # Panel 2: adaptive grid gradient
    ax2 = Axis(fig[1, 2];
               xlabel = "Departure [days]",
               ylabel = "Arrival [days]",
               title  = "|∇(ΔV)| — adaptive grid  |  $name1 → $name2",
               titlesize = 12)
    hm2 = heatmap!(ax2, p.deps, p.arrs, G_a;
                   colormap = :viridis, nan_color = :lightgray, colorrange = clims)
    for dep in p.deps
        vlines!(ax2, dep; color=:white, linewidth=0.5, alpha=0.3)
    end
    for arr in p.arrs
        hlines!(ax2, arr; color=:white, linewidth=0.5, alpha=0.3)
    end

    cb = Colorbar(fig[1, 3], hm2; label = "|∇(ΔV)| [m/s/day]")

    return fig
end

function plot_grid_comparison(name1::String, name2::String,
                               base_ct::Dict, ag, sim)
    idx1 = sim.id_to_idx[name1]
    idx2 = sim.id_to_idx[name2]

    matching = [(k, v) for (k, v) in base_ct if k[1] == idx1 && k[2] == idx2]
    deps_u = sort(unique(Float64[k[3] for (k, _) in matching]))
    arrs_u = sort(unique(Float64[k[4] for (k, _) in matching]))

    p = ag.pairs[ag.lookup[(idx1, idx2)]]

    fig = Figure(size=(720, 660), backgroundcolor=:white)
    ax  = Axis(fig[1, 1];
               xlabel = "Departure [days]",
               ylabel = "Arrival [days]",
               title  = "Grid comparison: $name1 → $name2",
               backgroundcolor = :white,
               xgridvisible = false,
               ygridvisible = false)

    for d in deps_u
        vlines!(ax, d; color=(:gray, 0.4), linewidth=0.8, linestyle=:dash)
    end
    for a in arrs_u
        hlines!(ax, a; color=(:gray, 0.4), linewidth=0.8, linestyle=:dash)
    end

    for d in p.deps
        vlines!(ax, d; color=:black, linewidth=1.2)
    end
    for a in p.arrs
        hlines!(ax, a; color=:black, linewidth=1.2)
    end

    n_u = length(deps_u) * length(arrs_u)
    n_a = length(p.deps) * length(p.arrs)
    axislegend(ax,
        [LineElement(color=(:gray, 0.4), linestyle=:dash, linewidth=0.8),
         LineElement(color=:black, linewidth=1.2)],
        ["Uniform ($(length(deps_u))×$(length(arrs_u)) = $n_u)",
         "Adaptive ($(length(p.deps))×$(length(p.arrs)) = $n_a)"],
        position=:lt)

    return fig
end
