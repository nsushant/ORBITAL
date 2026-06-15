using Dash
using DashBootstrapComponents
using PlotlyBase
using JLD2
using CSV
using DataFrames
using Statistics

@load joinpath(@__DIR__, "..", "outputs", "dashboard_data.jld2") schedule_copy schedule unassigned demands

deadlines = demands["demand_deadlines"]
n_veh     = length(schedule)

# ── load MDLS archive for click-through interactivity ────────────────────────
const ARCHIVE_PATH = joinpath(@__DIR__, "..", "outputs", "mdls_archive.jld2")
const HAS_ARCHIVE  = isfile(ARCHIVE_PATH)
HAS_ARCHIVE && @load ARCHIVE_PATH archive

const GANTT_MAX_TIME = begin
    base = maximum(maximum(veh.departures) for veh in schedule) + 20
    if HAS_ARCHIVE
        arch_max = maximum(
            maximum(veh.departures)
            for sol in archive.solutions if !isempty(sol)
            for veh in sol if !isempty(veh.departures)
        )
        max(base, arch_max + 20)
    else
        base
    end
end

# ── Gantt ─────────────────────────────────────────────────────────────────────
function gantt_figure(sched=schedule,
                      title="Optimised Schedule — click an MDLS point to inspect")
    depot_x = Float64[]; depot_base = Float64[]; depot_y = Int[]
    demand_x = Float64[]; demand_base = Float64[]; demand_y = Int[]
    slack_x  = Float64[]; slack_base  = Float64[]; slack_y  = Int[]

    for (v, veh) in enumerate(sched)
        for i in eachindex(veh.visitedUID)
            uid = veh.visitedUID[i]
            arr = veh.arrivals[i]
            dep = veh.departures[i]
            dep == arr && (dep = arr + 0.5)
            if uid < 0
                push!(depot_x, dep - arr); push!(depot_base, arr); push!(depot_y, v)
            else
                push!(demand_x, dep - arr); push!(demand_base, arr); push!(demand_y, v)
                sl = deadlines[uid] - dep
                sl > 0 && (push!(slack_x, sl); push!(slack_base, dep); push!(slack_y, v))
            end
        end
    end

    nv = length(sched)
    traces = GenericTrace[]
    !isempty(slack_x)  && push!(traces, bar(x=slack_x,  y=slack_y,  base=slack_base,
        orientation="h", marker_color="rgba(180,180,180,0.25)", showlegend=false,
        hovertemplate="Slack: %{x:.1f} days<extra></extra>"))
    !isempty(depot_x)  && push!(traces, bar(x=depot_x,  y=depot_y,  base=depot_base,
        orientation="h", marker_color="#4A7BB5", name="Depot",
        hovertemplate="Depot %{base:.1f}→+%{x:.1f}<extra></extra>"))
    !isempty(demand_x) && push!(traces, bar(x=demand_x, y=demand_y, base=demand_base,
        orientation="h", marker_color="#D4733B", name="Service",
        hovertemplate="Service %{base:.1f}→+%{x:.1f}<extra></extra>"))

    lay = Layout(
        title     = title,
        xaxis     = attr(title="Time [days]", range=[0, GANTT_MAX_TIME]),
        yaxis     = attr(title=nothing, range=[0, nv + 1], dtick=1, tickmode="linear"),
        barmode   = "overlay",
        height    = max(300, nv * 18 + 80),
        hovermode = "closest",
        legend    = attr(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin    = attr(l=40, r=20, t=50, b=40),
    )
    Plot(traces, lay, config=PlotConfig(scrollZoom=true))
end

# ── Pareto comparison ─────────────────────────────────────────────────────────
function pareto_comparison_figure()
    alg_configs = [
        ("mdls",     "MDLS",            "#1f77b4", "solid"),
        ("nsga2",    "NSGA-II",         "#d62728", "dot"),
        ("smsemoa",  "SMS-EMOA",        "#2ca02c", "dot"),
        ("moead",    "MOEA/D-DE",       "#ff7f0e", "dot"),
        ("memetic",  "Memetic NSGA-II", "#9467bd", "dashdot"),
    ]

    traces = GenericTrace[]
    for (fname, label, color, dash_style) in alg_configs
        if fname == "mdls" && HAS_ARCHIVE
            # build MDLS trace from archive so customdata carries 1-based archive index
            xs    = archive.total_deltaV
            ys    = archive.total_serv_time_unassigned
            f3    = Float64.(archive.total_vehicles_used)
            keep  = vec(isfinite.(xs) .& isfinite.(ys))
            xs, ys, f3 = xs[keep], ys[keep], f3[keep]
            idxs  = (1:length(archive.solutions))[keep]
            order = sortperm(xs)
            msz   = clamp.(f3[order] .* 0.5 .+ 5.0, 5.0, 18.0)
            push!(traces, scatter(
                x          = xs[order],
                y          = ys[order],
                customdata = idxs[order],
                mode       = "markers+lines",
                name       = label,
                marker     = attr(size=msz, color=color, opacity=0.75),
                line       = attr(color=color, width=1.8, dash=dash_style),
                hovertemplate = "MDLS<br>ΔV: %{x:.0f} m/s<br>Unassigned: %{y:.2f} days<br><i>click to view schedule</i><extra></extra>",
            ))
        else
            path = joinpath(@__DIR__, "..", "outputs", "ga_pareto_$fname.csv")
            isfile(path) || continue
            df = sort!(CSV.read(path, DataFrame), :f1_dv)
            isempty(df) && continue
            msz = clamp.(df.f3_vehicles .* 0.5 .+ 5.0, 5.0, 18.0)
            push!(traces, scatter(
                x    = df.f1_dv,
                y    = df.f2_unassigned_time,
                mode = "markers+lines",
                name = label,
                marker = attr(size=msz, color=color, opacity=0.75),
                line   = attr(color=color, width=1.8, dash=dash_style),
                hovertemplate = "$label<br>ΔV: %{x:.0f} m/s<br>Unassigned: %{y:.2f} days<extra></extra>",
            ))
        end
    end

    isempty(traces) && push!(traces, scatter(
        x=[0], y=[0], mode="text",
        text=["Run GATests.jl first to populate comparison data"],
        textposition="middle center", showlegend=false,
    ))

    lay = Layout(
        title     = "Pareto Front Comparison  (click MDLS point to inspect schedule)",
        xaxis     = attr(title="Total ΔV [m/s]"),
        yaxis     = attr(title="Unassigned service time [days]"),
        height    = 420,
        hovermode = "closest",
        legend    = attr(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin    = attr(l=60, r=20, t=60, b=50),
    )
    Plot(traces, lay, config=PlotConfig(scrollZoom=true))
end

# ── Unassigned table ──────────────────────────────────────────────────────────
function unassigned_panel()
    if unassigned === nothing
        return dbc_card([
            dbc_cardheader("Unassigned Demands"),
            dbc_cardbody(html_p("All demands routed ✓", className="text-center text-success")),
        ])
    end
    uids = unassigned["UIDs"]
    sats = unassigned["sat_identifiers"]
    dls  = unassigned["demand_deadlines"]
    svcs = unassigned["service_times"]
    header = html_tr([html_th(x, scope="col") for x in ["UID", "Satellite", "Deadline", "Service"]])
    rows   = [header; [html_tr([
        html_td(string(uids[j])), html_td(sats[j]),
        html_td(string(round(dls[j]; digits=1))),
        html_td(string(round(svcs[j]; digits=1))),
    ]) for j in eachindex(uids)]]
    dbc_card([
        dbc_cardheader("Unassigned Demands ($(length(uids)))"),
        dbc_cardbody(dbc_table(children=rows, bordered=true, striped=true, size="sm", hover=true)),
    ])
end

# ── Header metrics ────────────────────────────────────────────────────────────
function MetricCard(label, value; color="inherit")
    html_div([
        html_p(label, className="mb-0 small text-muted"),
        html_h4(value, className="mb-0", style=Dict("color" => color)),
    ], className="text-center")
end

function header_metrics()
    total_bf = sum(sum(veh.costs) for veh in schedule_copy)
    total_af = sum(sum(veh.costs) for veh in schedule)
    pct      = total_bf > 0 ? round((total_bf - total_af) / total_bf * 100; digits=1) : 0.0
    n_served = sum(count(uid -> uid > 0, veh.visitedUID) for veh in schedule)
    n_unas   = unassigned === nothing ? 0 : length(unassigned["UIDs"])

    dbc_card(dbc_cardbody(dbc_row([
        dbc_col(MetricCard("Total ΔV",       "$(round(Int, total_af)) m/s"),             width=3),
        dbc_col(MetricCard("ΔV Improvement", "$pct%", color=pct>0 ? "green" : "red"),    width=3),
        dbc_col(MetricCard("Vehicles Used",  string(n_veh)),                             width=3),
        dbc_col(MetricCard("Demands Routed", "$n_served served / $n_unas unassigned"),   width=3),
    ])), className="my-3")
end

# ── App layout ────────────────────────────────────────────────────────────────
app = dash(external_stylesheets=[dbc_themes.BOOTSTRAP])

app.layout = dbc_container([
    dbc_row(dbc_col(
        html_h2("OOS Schedule Dashboard", className="text-center my-3"),
    width=12)),

    dbc_row(dbc_col(header_metrics(), width=12)),

    dbc_row(dbc_col(dcc_graph(id="gantt", figure=gantt_figure()), width=12)),

    dbc_row([
        dbc_col(dcc_graph(id="pareto", figure=pareto_comparison_figure()), width=8),
        dbc_col(unassigned_panel(), width=4),
    ], className="mt-2"),

], fluid=true)

# ── Callback: Pareto click → Gantt update ────────────────────────────────────
callback!(app,
    Output("gantt", "figure"),
    Input("pareto",  "clickData"),
) do click_data
    click_data === nothing && return gantt_figure()

    pt = click_data["points"][1]

    # GA traces have no customdata — show a note and keep default schedule
    haskey(pt, "customdata") || return gantt_figure(schedule,
        "Schedule not stored for GA solutions — click an MDLS point instead")

    arch_idx = Int(pt["customdata"])
    sol      = archive.solutions[arch_idx]
    dv       = round(Int, archive.total_deltaV[arch_idx])
    nv       = archive.total_vehicles_used[arch_idx]
    us       = round(archive.total_serv_time_unassigned[arch_idx]; digits=1)
    return gantt_figure(sol,
        "MDLS #$arch_idx — ΔV: $dv m/s | $nv vehicles | $us days unassigned")
end

run_server(app, "0.0.0.0", 8050)
