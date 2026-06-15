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
