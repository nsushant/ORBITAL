# plot_results.jl — read numerical_experiments.csv and produce one spider plot
# per algorithm (4 spokes = 4 instances, value = median hypervolume).
# All spider plots share the same normalised axes so polygons are comparable.
#
# Run: julia --project=. plot_results.jl

using CSV, DataFrames, CairoMakie, Printf, Statistics

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = joinpath(@__DIR__, "outputs", "numerical_experiments.csv")
isfile(csv_path) || error("Results file not found: $csv_path\n" *
                           "Run numerical_experiments.jl first.")

df = CSV.read(csv_path, DataFrame)

alg_order  = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]
inst_order = unique(df.instance)   # preserve file order

# ── Compute median HV per (algorithm, instance) ───────────────────────────────
summary = combine(groupby(df, [:algorithm, :instance]),
                  :hypervolume => median => :med_hv,
                  :hypervolume => (x -> quantile(x, 0.75) - quantile(x, 0.25)) => :iqr_hv)

# Build matrix: rows = algorithms, cols = instances
n_alg  = length(alg_order)
n_inst = length(inst_order)
hv_mat = Matrix{Float64}(undef, n_alg, n_inst)

for (ai, alg) in enumerate(alg_order)
    for (ii, inst) in enumerate(inst_order)
        row = filter(r -> r.algorithm == alg && r.instance == inst, summary)
        hv_mat[ai, ii] = isempty(row) ? 0.0 : first(row.med_hv)
    end
end

# ── Normalize per instance: outer edge = best (highest HV) per spoke ──────────
col_min = [minimum(hv_mat[:, j]) for j in 1:n_inst]
col_max = [maximum(hv_mat[:, j]) for j in 1:n_inst]

_norm(val, j) = (col_max[j] - col_min[j]) < 1e-12 ? 1.0 :
                (val - col_min[j]) / (col_max[j] - col_min[j])

# ── Spider geometry: n_inst spokes, evenly spaced, first spoke at top ─────────
θ = [π/2 - (j - 1) * 2π / n_inst for j in 1:n_inst]   # clockwise from top

function _spider_pts(display_vals)
    pts = [Point2f(display_vals[j] * cos(θ[j]), display_vals[j] * sin(θ[j]))
           for j in 1:n_inst]
    push!(pts, pts[1])   # close polygon
    return pts
end

# ── Nice short instance labels ─────────────────────────────────────────────────
inst_labels = replace.(inst_order,
    "tight_normal"    => "Tight\nnormal",
    "loose_uniform"   => "Loose\nuniform",
    "dense_tight_dv"  => "Dense\ntight-ΔV",
    "sparse_wide_dv"  => "Sparse\nwide-ΔV")

# ── Color palette ──────────────────────────────────────────────────────────────
alg_colors = [:royalblue, :crimson, :seagreen, :darkorange]

# ── Build figure: 2×2 grid of spider plots ────────────────────────────────────
fig = Figure(size = (1100, 1000))

for (ai, alg) in enumerate(alg_order)
    row_i = (ai - 1) ÷ 2 + 1
    col_i = (ai - 1) % 2 + 1
    col   = alg_colors[mod1(ai, length(alg_colors))]

    ax = Axis(fig[row_i, col_i];
              title              = alg,
              titlesize          = 15,
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

    # Background: concentric reference polygons at 0.25, 0.5, 0.75, 1.0
    for r in (0.25, 0.5, 0.75, 1.0)
        ring = [Point2f(r * cos(θ[j]), r * sin(θ[j])) for j in 1:n_inst]
        push!(ring, ring[1])
        lines!(ax, ring;
               color     = (:lightgray, 0.9),
               linewidth = r == 1.0 ? 1.2 : 0.6,
               linestyle = r == 1.0 ? :solid : :dash)
    end
    # Percentage labels on first spoke
    for r in (0.25, 0.5, 0.75)
        text!(ax, r * cos(θ[1]) + 0.03, r * sin(θ[1]);
              text     = string(round(Int, r * 100), "%"),
              fontsize = 8,
              color    = :gray60,
              align    = (:left, :center))
    end

    # Spoke lines
    for j in 1:n_inst
        lines!(ax, [Point2f(0, 0), Point2f(cos(θ[j]), sin(θ[j]))];
               color     = (:gray, 0.45),
               linewidth = 0.9)
    end

    # Spoke labels (instance names + best/worst HV anchors)
    label_scale = 1.28
    for j in 1:n_inst
        tx = label_scale * cos(θ[j])
        ty = label_scale * sin(θ[j])
        best_str  = @sprintf("%.2e", col_max[j])
        worst_str = @sprintf("%.2e", col_min[j])
        text!(ax, tx, ty;
              text     = "$(inst_labels[j])\n▲$best_str\n▽$worst_str",
              fontsize = 8,
              align    = (:center, :center),
              color    = :black)
    end

    # Algorithm polygon
    disp  = [_norm(hv_mat[ai, j], j) for j in 1:n_inst]
    pts   = _spider_pts(disp)
    poly!(ax, pts[1:end-1];
          color       = (col, 0.22),
          strokecolor = col,
          strokewidth = 2.5)
    lines!(ax, pts; color = col, linewidth = 2.5)
    scatter!(ax, [p[1] for p in pts[1:end-1]], [p[2] for p in pts[1:end-1]];
             color = col, markersize = 10)

    # IQR error "whiskers" along each spoke
    for j in 1:n_inst
        row_s = filter(r -> r.algorithm == alg && r.instance == inst_order[j], summary)
        isempty(row_s) && continue
        iqr_val = first(row_s.iqr_hv)
        med_val = first(row_s.med_hv)
        rng     = max(col_max[j] - col_min[j], 1e-12)
        r_med   = _norm(med_val, j)
        r_lo    = clamp(_norm(med_val - iqr_val / 2, j), 0, 1)
        r_hi    = clamp(_norm(med_val + iqr_val / 2, j), 0, 1)
        cx, cy  = cos(θ[j]), sin(θ[j])
        lines!(ax, [Point2f(r_lo * cx, r_lo * cy), Point2f(r_hi * cx, r_hi * cy)];
               color     = (col, 0.70),
               linewidth = 4.0)
    end

    limits!(ax, -1.7, 1.7, -1.7, 1.7)
end

# ── Shared legend ─────────────────────────────────────────────────────────────
Legend(fig[3, 1:2],
       [PolyElement(color=(alg_colors[i], 0.30), strokecolor=alg_colors[i], strokewidth=2)
        for i in eachindex(alg_order)],
       alg_order;
       orientation  = :horizontal,
       tellwidth    = false,
       tellheight   = true,
       framevisible = false)

Label(fig[0, 1:2],
      "Median hypervolume by instance  (outer edge = best per instance, whiskers = IQR)";
      fontsize = 13, tellwidth = false)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = joinpath(@__DIR__, "outputs", "spider_by_algorithm.png")
save(out_path, fig; px_per_unit = 2)
@info "Spider plots saved" out_path
