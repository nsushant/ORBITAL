using Dash
using DashBootstrapComponents
using PlotlyBase
using JLD2
using Statistics

@load joinpath(@__DIR__, "..", "outputs", "dashboard_data.jld2") schedule_copy schedule unassigned demands

deadlines = demands["demand_deadlines"]
n_veh = max(length(schedule_copy), length(schedule))

max_time = maximum(maximum(veh.departures) for veh in [schedule_copy; schedule]) + 20

function gantt_figure(sched, title_text)
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
                push!(depot_x, dep - arr)
                push!(depot_base, arr)
                push!(depot_y, v)
            else
                push!(demand_x, dep - arr)
                push!(demand_base, arr)
                push!(demand_y, v)

                dl = deadlines[uid]
                slack = dl - dep
                if slack > 0
                    push!(slack_x, slack)
                    push!(slack_base, dep)
                    push!(slack_y, v)
                end
            end
        end
    end

    traces = GenericTrace[]
    isempty(slack_x) && isempty(depot_x) && isempty(demand_x) && return Plot(Layout(title_text))

    if !isempty(slack_x)
        push!(traces, bar(
            x=slack_x, y=slack_y, base=slack_base,
            orientation="h",
            marker_color="rgba(180,180,180,0.25)",
            showlegend=false,
            hovertemplate="Slack: %{x:.1f} days<extra></extra>"
        ))
    end
    if !isempty(depot_x)
        push!(traces, bar(
            x=depot_x, y=depot_y, base=depot_base,
            orientation="h",
            marker_color="#4A7BB5",
            name="Depot",
            hovertemplate="Depot<br>%{base:.1f}→%{base}+%{x:.1f}<extra></extra>"
        ))
    end
    if !isempty(demand_x)
        push!(traces, bar(
            x=demand_x, y=demand_y, base=demand_base,
            orientation="h",
            marker_color="#D4733B",
            name="Service",
            hovertemplate="Service<br>%{base:.1f}→%{base}+%{x:.1f}<extra></extra>"
        ))
    end

    lay = Layout(
        title=title_text,
        xaxis=attr(title="Time [days]", range=[0, max_time]),
        yaxis=attr(title=nothing, range=[0, n_veh + 1], dtick=1, tickmode="linear"),
        barmode="overlay",
        height=350,
        hovermode="closest",
        legend=attr(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=attr(l=40, r=20, t=40, b=40),
    )

    return Plot(traces, lay, config=PlotConfig(scrollZoom=true))
end

function dv_chart()
    bf = Float64[sum(veh.costs) for veh in schedule_copy]
    af = Float64[sum(veh.costs) for veh in schedule]
    xs = collect(1:n_veh)

    traces = GenericTrace[
        bar(x=xs, y=bf, name="Before",
            marker_color="#4A7BB5",
            hovertemplate="Vehicle %{x}<br>ΔV: %{y:.0f} m/s<extra>Before</extra>"),
        bar(x=xs, y=af, name="After",
            marker_color="#D4733B",
            hovertemplate="Vehicle %{x}<br>ΔV: %{y:.0f} m/s<extra>After</extra>"),
    ]

    lay = Layout(
        title="ΔV per Vehicle (Before vs After)",
        barmode="group",
        xaxis=attr(title="Vehicle", dtick=1, tickmode="linear"),
        yaxis=attr(title="ΔV [m/s]"),
        height=300,
        hovermode="x unified",
        margin=attr(l=40, r=20, t=40, b=40),
    )

    return Plot(traces, lay, config=PlotConfig(scrollZoom=true))
end

function avg_slack(sched)
    slacks = Float64[]
    for veh in sched
        for i in eachindex(veh.visitedUID)
            uid = veh.visitedUID[i]
            uid > 0 || continue
            push!(slacks, deadlines[uid] - veh.departures[i])
        end
    end
    isempty(slacks) ? 0.0 : mean(slacks)
end

function unassigned_panel()
    if unassigned === nothing
        return dbc_card([
            dbc_cardheader("Unassigned Demands"),
            dbc_cardbody(html_p("All demands routed", className="text-center text-success")),
        ])
    end

    uids = unassigned["UIDs"]
    sats = unassigned["sat_identifiers"]
    dls  = unassigned["demand_deadlines"]
    svcs = unassigned["service_times"]

    header = html_tr([html_th(x, scope="col") for x in ["UID", "Satellite", "Deadline", "Service"]])
    rows = [header]
    for j in eachindex(uids)
        push!(rows, html_tr([
            html_td(string(uids[j])),
            html_td(sats[j]),
            html_td(string(round(dls[j], digits=1))),
            html_td(string(round(svcs[j], digits=1))),
        ]))
    end

    dbc_card([
        dbc_cardheader("Unassigned Demands ($(length(uids)))"),
        dbc_cardbody(
            dbc_table(children=rows, bordered=true, striped=true, size="sm", hover=true)
        ),
    ])
end

function MetricCard(label, value; color="dark")
    color_map = Dict("dark" => "inherit", "success" => "green", "danger" => "red")
    html_div([
        html_p(label, className="mb-0 small text-muted"),
        html_h4(value, className="mb-0", style=Dict("color" => get(color_map, color, "inherit"))),
    ])
end

function stats_panel()
    total_bf = sum(sum(veh.costs) for veh in schedule_copy)
    total_af = sum(sum(veh.costs) for veh in schedule)
    pct_improve = total_bf > 0 ? round((total_bf - total_af) / total_bf * 100, digits=1) : 0.0

    n_dems_af = sum(count(uid -> uid > 0, veh.visitedUID) for veh in schedule)
    n_unassigned = unassigned === nothing ? 0 : length(unassigned["UIDs"])

    slack_bf = round(avg_slack(schedule_copy), digits=1)
    slack_af = round(avg_slack(schedule), digits=1)

    dbc_card([
        dbc_cardheader("Summary Statistics"),
        dbc_cardbody([
            dbc_row([
                dbc_col(MetricCard("Total ΔV Before", "$(round(Int, total_bf)) m/s"), width=4),
                dbc_col(MetricCard("Total ΔV After", "$(round(Int, total_af)) m/s"), width=4),
                dbc_col(MetricCard("Improvement", "$pct_improve%",
                    color=pct_improve > 0 ? "success" : "danger"), width=4),
            ]),
            html_hr(style=Dict("margin" => "8px 0")),
            dbc_row([
                dbc_col(MetricCard("Vehicles Used", string(n_veh)), width=4),
                dbc_col(MetricCard("Demands Routed", string(n_dems_af)), width=4),
                dbc_col(MetricCard("Unassigned", string(n_unassigned)), width=4),
            ]),
            html_hr(style=Dict("margin" => "8px 0")),
            dbc_row([
                dbc_col(MetricCard("Avg Slack Before", "$slack_bf days"), width=6),
                dbc_col(MetricCard("Avg Slack After", "$slack_af days"), width=6),
            ]),
        ]),
    ])
end

app = dash(external_stylesheets=[dbc_themes.BOOTSTRAP])

app.layout = dbc_container([
    dbc_row(dbc_col(
        html_h1("Schedule Comparison Dashboard", className="text-center my-4"),
        width=12
    )),

    dbc_row([
        dbc_col(dcc_graph(id="gantt-before", figure=gantt_figure(schedule_copy, "Before Optimization")), width=6),
        dbc_col(dcc_graph(id="gantt-after",  figure=gantt_figure(schedule,      "After Optimization")),  width=6),
    ]),

    dbc_row(dbc_col(dcc_graph(id="dv-chart", figure=dv_chart()), width=12)),

    dbc_row([
        dbc_col(unassigned_panel(), width=6),
        dbc_col(stats_panel(),      width=6),
    ]),
], fluid=true)

run_server(app, "0.0.0.0", 8050)
