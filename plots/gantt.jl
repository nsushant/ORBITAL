
using VegaLite
using DataFrames
using CairoMakie

const _GANTT_BLUE   = CairoMakie.RGBA(0.30, 0.55, 0.85, 0.85)
const _GANTT_ORANGE = CairoMakie.RGBA(0.85, 0.45, 0.25, 0.85)
const _GANTT_GREEN  = CairoMakie.RGBA(0.25, 0.70, 0.45, 0.85)
const _GANTT_PURPLE = CairoMakie.RGBA(0.70, 0.25, 0.55, 0.85)
const _BAR_MIN      = 0.5

function _gantt_cairo(schedule, save_path)
    n_vehicles = length(schedule)
    fig = Figure(size=(1000, max(350, n_vehicles * 30 + 50)))
    ax = Axis(fig[1, 1]; xlabel="Time [days]", ylabel="Vehicle",
              title="Schedule Gantt", yreversed=true)

    for (v, veh) in enumerate(schedule)
        y_top = v + 0.2
        for i in eachindex(veh.visitedUID)
            x1 = veh.arrivals[i]
            x2 = veh.departures[i]
            x2 == x1 && (x2 = x1 + _BAR_MIN)
            is_depot = veh.visitedUID[i] < 0
            pts = [Point2f(x1, y_top), Point2f(x2, y_top),
                   Point2f(x2, y_top - 0.4), Point2f(x1, y_top - 0.4)]
            poly!(ax, pts, color=is_depot ? _GANTT_BLUE : _GANTT_ORANGE,
                  strokecolor=:black, strokewidth=0.5)
        end
    end

    ax.yticks = (1:n_vehicles, ["Vehicle $v" for v in 1:n_vehicles])
    xlims!(ax, 0, maximum(maximum(veh.departures) for veh in schedule) + 10)
    ylims!(ax, 0.5, n_vehicles + 0.5)

    elements = [PolyElement(color=_GANTT_ORANGE),
                PolyElement(color=_GANTT_BLUE)]
    Legend(fig[1, 2], elements, ["Service", "Depot"], "Type")

    save(save_path, fig)
    @info "Gantt chart saved to $save_path"
    return fig
end

function _gantt_vega(schedule, save_path)
    rows = []
    for (v, veh) in enumerate(schedule)
        for i in eachindex(veh.visitedUID)
            x1 = veh.arrivals[i]
            x2 = veh.departures[i]
            x2 == x1 && (x2 = x1 + _BAR_MIN)
            push!(rows, (
                vehicle = "Vehicle $v",
                task    = veh.visitedSAT[i],
                start   = x1, finish = x2,
                type    = startswith(veh.visitedSAT[i], "depot") ? "depot" : "service",
            ))
        end
    end

    chart = DataFrame(rows) |>
        @vlplot(
            mark={:bar, tooltip=true, cornerRadius=3},
            width=900, height=500,
            x={:start, title="Time [days]", axis={grid=true}},
            x2=:finish,
            y={:vehicle, title=nothing, axis={labelFontSize=12}},
            color={:type, legend={title="Type"}},
        )

    VegaLite.save(save_path, chart)
    @info "Gantt chart saved to $save_path"
    return chart
end

"""
    plot_schedule_gantt(schedule; save_to="outputs/schedule.html")

Gantt chart of the vehicle schedule.  Extension determines output format:
  - .html  → interactive VegaLite (zoom, pan, tooltip)
  - .png   → static CairoMakie image
"""
function plot_schedule_gantt(schedule; save_to="outputs/schedule.html")
    if endswith(lowercase(save_to), ".png")
        return _gantt_cairo(schedule, save_to)
    else
        return _gantt_vega(schedule, save_to)
    end
end

"""
    plot_schedule_comparison(sched1, sched2; label1, label2, save_to)

Overlay two schedules in the same frame, split per row for side-by-side
comparison.  Saves as PNG.
"""
function plot_schedule_comparison(sched1, sched2;
    label1="Schedule 1", label2="Schedule 2",
    save_to="outputs/comparison.png")

    n_veh = max(length(sched1), length(sched2))
    fig = Figure(size=(1000, max(350, n_veh * 40 + 50)))
    ax = Axis(fig[1, 1]; xlabel="Time [days]", ylabel="Vehicle",
              title="Schedule Comparison ($label1 vs $label2)", yreversed=true)

    max_time = 0.0
    for v in 1:n_veh
        y_mid = v
        # top half — schedule 1
        if v <= length(sched1)
            for i in eachindex(sched1[v].visitedUID)
                x1 = sched1[v].arrivals[i]
                x2 = sched1[v].departures[i]
                x2 == x1 && (x2 = x1 + _BAR_MIN)
                max_time = max(max_time, x2)
                is_depot = sched1[v].visitedUID[i] < 0
                pts = [Point2f(x1, y_mid + 0.02), Point2f(x2, y_mid + 0.02),
                       Point2f(x2, y_mid - 0.18), Point2f(x1, y_mid - 0.18)]
                poly!(ax, pts, color=is_depot ? _GANTT_BLUE : _GANTT_ORANGE,
                      strokecolor=:black, strokewidth=0.5)
            end
        end
        # bottom half — schedule 2
        if v <= length(sched2)
            for i in eachindex(sched2[v].visitedUID)
                x1 = sched2[v].arrivals[i]
                x2 = sched2[v].departures[i]
                x2 == x1 && (x2 = x1 + _BAR_MIN)
                max_time = max(max_time, x2)
                is_depot = sched2[v].visitedUID[i] < 0
                pts = [Point2f(x1, y_mid - 0.02), Point2f(x2, y_mid - 0.02),
                       Point2f(x2, y_mid - 0.38), Point2f(x1, y_mid - 0.38)]
                poly!(ax, pts, color=is_depot ? _GANTT_GREEN : _GANTT_PURPLE,
                      strokecolor=:black, strokewidth=0.5)
            end
        end
    end

    ax.yticks = (1:n_veh, ["Vehicle $v" for v in 1:n_veh])
    xlims!(ax, 0, max_time + 10)
    ylims!(ax, 0.5, n_veh + 0.5)

    elements = [PolyElement(color=_GANTT_ORANGE),
                PolyElement(color=_GANTT_BLUE),
                PolyElement(color=_GANTT_PURPLE),
                PolyElement(color=_GANTT_GREEN)]
    Legend(fig[1, 2], elements,
           ["Service $label1", "Depot $label1", "Service $label2", "Depot $label2"],
           "Type")

    save(save_to, fig)
    @info "Comparison Gantt saved to $save_to"
    return fig
end
