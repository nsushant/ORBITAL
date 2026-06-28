# plots/visualise.jl
# plotdvtable  — heatmap of departure vs arrival epoch coloured by ΔV
# plotdemands  — scatter of deadline vs total service time, coloured by min ΔV to depot

using CairoMakie

"""
    _gauss_blur(Z, σ) → Matrix{Float64}

NaN-aware 2D Gaussian blur with standard deviation `σ` (in grid cells).
NaN cells are skipped in the kernel sum; output cell is NaN only if no
finite neighbour falls within the kernel radius.
"""
function _gauss_blur(Z::Matrix{Float64}, σ::Float64=1.5)
    n, m = size(Z)
    r    = ceil(Int, 3σ)
    out  = fill(NaN, n, m)
    for i in 1:n, j in 1:m
        wsum = 0.0; vsum = 0.0
        for di in -r:r, dj in -r:r
            ii = i + di; jj = j + dj
            1 <= ii <= n && 1 <= jj <= m || continue
            isnan(Z[ii, jj]) && continue
            w     = exp(-0.5 * (di^2 + dj^2) / σ^2)
            vsum += w * Z[ii, jj]
            wsum += w
        end
        wsum > 0 && (out[i, j] = vsum / wsum)
    end
    return out
end

"""
    plotdvtable(name1, name2, cost_table, sim; smooth=true, σ=1.5)

Heatmap of departure epoch [days] vs arrival epoch [days] coloured by ΔV [m/s]
for transfers from node `name1` to node `name2`.
Invalid / missing cells are shown in light grey.
`smooth=true` applies a NaN-aware Gaussian blur with std `σ` grid cells.
"""
function plotdvtable(name1::String, name2::String, cost_table::Dict, sim;
                     smooth::Bool    = true,
                     σ::Float64      = 1.5)

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

    smooth && (Z = _gauss_blur(Z, σ))

    finite_vals = filter(isfinite, vec(Z))
    clims = isempty(finite_vals) ? (0.0, 1.0) : (minimum(finite_vals), maximum(finite_vals))

    fig = Figure(size = (860, 620))
    ax  = Axis(fig[1, 1];
               xlabel = "Departure [days]",
               ylabel = "Arrival [days]",
               title  = "ΔV [m/s]  |  $name1 → $name2" * (smooth ? "  (smoothed)" : ""))
    hm  = heatmap!(ax, deps, arrs, Z;
                   colormap   = :viridis,
                   nan_color  = :lightgray,
                   colorrange = clims)
    Colorbar(fig[1, 2], hm; label = "ΔV [m/s]")

    return fig
end

"""
    plotdemands(sat_identifiers, demand_deadlines, service_times, cost_table, sim)

Scatter plot where each point is one demand:
  x     = deadline [days]
  y     = total service time accumulated for that satellite across all its demands [days]
  color = minimum ΔV from any depot to that satellite (across all cost table entries) [m/s]
"""
function plotdemands(demands::Dict, cost_table::Dict, sim)

    sat_identifiers = demands["sat_identifiers"]
    demand_deadlines = demands["demand_deadlines"]
    service_times    = demands["service_times"]

    depot_idxs = findall(n -> startswith(n, "depot"), sim.names)
    isempty(depot_idxs) && error("No depot nodes found in simulation")

    # ── per-satellite total service time ──────────────────────────────────────
    sat_total_svc = Dict{String, Float64}()
    for (i, sat) in enumerate(sat_identifiers)
        sat_total_svc[sat] = get(sat_total_svc, sat, 0.0) + service_times[i]
    end

    # ── per-satellite minimum ΔV from any depot ────────────────────────────────
    sat_min_dv = Dict{String, Float64}()
    for sat in keys(sat_total_svc)
        si     = sim.id_to_idx[sat]
        min_dv = Inf
        for di in depot_idxs
            for (k, v) in cost_table
                k[1] == di && k[2] == si && v < min_dv && (min_dv = v)
            end
        end
        sat_min_dv[sat] = isfinite(min_dv) ? min_dv : NaN
    end

    x = demand_deadlines
    y = Float64[sat_total_svc[s]  for s in sat_identifiers]
    c = Float64[sat_min_dv[s]     for s in sat_identifiers]

    finite_c = filter(isfinite, c)
    clims    = isempty(finite_c) ? (0.0, 1.0) : (minimum(finite_c), maximum(finite_c))

    fig = Figure(size = (860, 620))
    ax  = Axis(fig[1, 1];
               xlabel = "Deadline [days]",
               ylabel = "Total service time for satellite [days]",
               title  = "Demands  (colour = min ΔV depot→sat [m/s])")
    sc  = scatter!(ax, x, y;
                   color       = c,
                   colormap    = :plasma,
                   colorrange  = clims,
                   markersize  = 9,
                   nan_color   = :lightgray)
    Colorbar(fig[1, 2], sc; label = "Min ΔV [m/s]")

    return fig
end

function plot_pareto(archive::Archive)

    isempty(archive.solutions) && error("Archive is empty.")

    xs      = archive.total_deltaV
    ys      = archive.total_serv_time_unassigned
    n_vehs  = archive.total_vehicles_used

    unique_nv   = sort(unique(n_vehs))
    base_colors = Makie.wong_colors()
    palette     = [base_colors[mod1(i, length(base_colors))] for i in eachindex(unique_nv)]

    fig = Figure(size=(1200, 800))
    ax  = Axis(fig[1, 1];
               xlabel = "Total ΔV [m/s]",
               ylabel = "Unassigned service time [days]",
               title  = "MDLS solutions")

    @info "plot_pareto data summary" n_solutions=length(xs) unique_vehicle_counts=unique_nv ΔV_range=(minimum(xs), maximum(xs)) unassigned_range=(minimum(ys), maximum(ys))
    for nv in unique_nv
        mask = findall(==(nv), n_vehs)
        @info "  vehicle group" n_vehicles=nv n_solutions=length(mask) ΔV_range=(minimum(xs[mask]), maximum(xs[mask])) unassigned_range=(minimum(ys[mask]), maximum(ys[mask]))
    end

    for (i, nv) in enumerate(unique_nv)
        mask  = findall(==(nv), n_vehs)
        col   = palette[i]
        label = "$nv vehicle$(nv == 1 ? "" : "s")"

        order = sortperm(xs[mask])
        gx    = xs[mask][order]
        gy    = ys[mask][order]

        lines!(ax, gx, gy; color=(col, 0.5), linewidth=1.5)
        scatter!(ax, gx, gy; color=col, markersize=10, label=label)
    end

    Legend(fig[1, 2], ax; title="Vehicles used", framevisible=true)

    return fig
end

"""
    plot_comparison(fronts; title, budget_sec)

Overlay plot comparing multiple algorithms on the same 2D Pareto axes (ΔV vs
unassigned service time).  `fronts` is a `Vector` of `(label, Matrix{Float64})`
pairs where each matrix is `n_solutions × 3` with columns [f1_dv, f2_unassigned,
f3_vehicles].

MDLS points are drawn with per-vehicle-count lines (same style as `plot_pareto`).
GA points are drawn as plain scatter with a connecting Pareto-front line sorted
by f1.  Marker size scales with f3 (vehicles used) across all algorithms.
"""
function plot_comparison(fronts::Vector{<:Tuple};
                         title::String    = "Algorithm comparison",
                         budget_sec::Real = 0)

    isempty(fronts) && error("No fronts to plot.")

    alg_colors = [:royalblue, :crimson, :seagreen, :darkorange,
                  :purple,    :brown,   :teal,     :olive]

    title_str = budget_sec > 0 ?
                "$title  ($(Int(round(budget_sec)))s budget each)" : title

    fig = Figure(size = (1500, 900))
    ax  = Axis(fig[1, 1];
               xlabel = "Total ΔV [m/s]",
               ylabel = "Unassigned service time [days]",
               title  = title_str)

    # global vehicle range for consistent marker sizing
    all_f3   = vcat([m[:, 3] for (_, m) in fronts]...)
    f3_min   = minimum(all_f3)
    f3_max   = max(maximum(all_f3), f3_min + 1.0)
    _msz(f3) = @. 5.0 + 15.0 * (f3 - f3_min) / (f3_max - f3_min)

    for (ai, (lbl, m)) in enumerate(fronts)
        isempty(m) && continue
        col    = alg_colors[mod1(ai, length(alg_colors))]
        xs     = m[:, 1]
        ys     = m[:, 2]
        f3vals = m[:, 3]

        if lbl == "MDLS"
            # per-vehicle-count groups with connecting lines (same as plot_pareto)
            unique_nv = sort(unique(Int.(f3vals)))
            nv_colors = Makie.wong_colors()
            for (ni, nv) in enumerate(unique_nv)
                mask  = findall(==(nv), f3vals)
                ncol  = nv_colors[mod1(ni, length(nv_colors))]
                order = sortperm(xs[mask])
                gx    = xs[mask][order]
                gy    = ys[mask][order]
                lines!(ax, gx, gy;
                       color     = (ncol, 0.35),
                       linewidth = 1.2)
                scatter!(ax, gx, gy;
                         color      = (ncol, 0.7),
                         markersize = _msz(Float64(nv)),
                         label      = ni == 1 ? "MDLS" : nothing)
            end
        else
            # GA: sort by f1 and draw a single connecting line
            order = sortperm(xs)
            lines!(ax, xs[order], ys[order];
                   color     = (col, 0.45),
                   linewidth = 1.8)
            scatter!(ax, xs, ys;
                     color      = (col, 0.75),
                     markersize = _msz.(f3vals),
                     label      = lbl)
        end
    end

    Legend(fig[1, 2], ax;
           title        = "Algorithm\n(size ∝ vehicles)",
           framevisible = true,
           merge        = true)

    return fig
end

"""
    plot_spider_comparison(fronts; title)

Radar / spider chart comparing algorithms across all three objectives simultaneously.
`fronts` is a `Vector` of `(label, Matrix{Float64})` pairs (n_solutions × 3,
columns: f1_dv, f2_unassigned_time, f3_vehicles).

Each algorithm is summarised by its per-objective minimum (ideal point of its
Pareto front).  On each spoke, **outer edge = best**, centre = worst — so a
larger polygon means better overall performance.
"""
function plot_spider_comparison(fronts::Vector{<:Tuple};
                                title::String = "Algorithm comparison — radar") :: Figure

    isempty(fronts) && error("No fronts to plot.")

    obj_labels = ["Total ΔV [m/s]", "Unassigned\ntime [days]", "Vehicles\nused"]
    alg_colors = [:royalblue, :crimson, :seagreen, :darkorange,
                  :purple,    :brown,   :teal,     :olive]

    # ── Per-algorithm ideal point (per-objective minimum) ─────────────────────
    n_alg = length(fronts)
    bests = Matrix{Float64}(undef, n_alg, 3)
    for (ai, (_, m)) in enumerate(fronts)
        for j in 1:3
            bests[ai, j] = minimum(m[:, j])
        end
    end

    # ── Global range for normalization ────────────────────────────────────────
    gmin = [minimum(bests[:, j]) for j in 1:3]
    gmax = [maximum(bests[:, j]) for j in 1:3]

    # display value: 1 = best (lowest objective), 0 = worst (highest objective)
    _display(val, j) = 1.0 - (val - gmin[j]) / max(gmax[j] - gmin[j], 1e-10)

    # ── Spoke geometry (equilateral triangle, spoke 1 points straight up) ─────
    θ = [π/2 + (j - 1) * 2π/3 for j in 1:3]

    # Polygon vertices for algorithm ai
    function _polygon(ai)
        pts = Point2f[]
        for j in 1:3
            r = _display(bests[ai, j], j)
            push!(pts, Point2f(r * cos(θ[j]), r * sin(θ[j])))
        end
        push!(pts, pts[1])   # close the loop
        return pts
    end

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = Figure(size = (900, 720))
    ax  = Axis(fig[1, 1];
               title              = title,
               aspect             = DataAspect(),
               xgridvisible       = false,
               ygridvisible       = false,
               xticksvisible      = false,
               yticksvisible      = false,
               xticklabelsvisible = false,
               yticklabelsvisible = false,
               leftspinevisible   = false,
               rightspinevisible  = false,
               topspinevisible    = false,
               bottomspinevisible = false)

    # ── Background: concentric reference triangles + spoke lines ─────────────
    for r in (0.25, 0.5, 0.75, 1.0)
        ring_pts = [Point2f(r * cos(θ[j]), r * sin(θ[j])) for j in 1:3]
        push!(ring_pts, ring_pts[1])
        lines!(ax, ring_pts;
               color     = (:lightgray, 0.9),
               linewidth = r == 1.0 ? 1.2 : 0.7,
               linestyle = r == 1.0 ? :solid : :dash)
        # percentage label on the first spoke
        if r < 1.0
            text!(ax, r * cos(θ[1]) + 0.03, r * sin(θ[1]);
                  text     = string(round(Int, (1 - r) * 100), "%"),
                  fontsize = 8,
                  color    = :gray60,
                  align    = (:left, :center))
        end
    end
    for j in 1:3
        lines!(ax, [Point2f(0, 0), Point2f(cos(θ[j]), sin(θ[j]))];
               color     = (:gray, 0.5),
               linewidth = 1.0)
    end

    # ── Spoke axis labels ─────────────────────────────────────────────────────
    label_nudge = [(0.0, 0.20), (0.20, -0.14), (-0.20, -0.14)]
    for j in 1:3
        tx = 1.15 * cos(θ[j]) + label_nudge[j][1]
        ty = 1.15 * sin(θ[j]) + label_nudge[j][2]
        text!(ax, tx, ty;
              text     = "$(obj_labels[j])\n▲ $(round(gmin[j]; sigdigits=4))\n▽ $(round(gmax[j]; sigdigits=4))",
              fontsize = 9,
              align    = (:center, :center),
              color    = :black)
    end

    # ── Per-algorithm polygons ────────────────────────────────────────────────
    legend_elements = PolyElement[]
    legend_labels   = String[]

    for (ai, (lbl, _)) in enumerate(fronts)
        col = alg_colors[mod1(ai, length(alg_colors))]
        pts = _polygon(ai)
        poly!(ax, pts[1:end-1];
              color       = (col, 0.20),
              strokecolor = col,
              strokewidth = 2.2)
        lines!(ax, pts; color = col, linewidth = 2.2)
        scatter!(ax, [p[1] for p in pts[1:end-1]], [p[2] for p in pts[1:end-1]];
                 color = col, markersize = 9)
        push!(legend_elements, PolyElement(color=(col, 0.35), strokecolor=col, strokewidth=2))
        push!(legend_labels,   lbl)
    end

    Legend(fig[1, 2], legend_elements, legend_labels;
           title        = "Algorithm\n(outer = better)",
           framevisible = true)

    limits!(ax, -1.65, 1.65, -1.55, 1.65)
    return fig
end
