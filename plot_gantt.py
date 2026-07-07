"""
plot_gantt.py
Reads the 3 knee-point schedule JSONs produced by export_knee_schedules.jl
and generates an interactive HTML Gantt chart (one tab per algorithm).

Run: python plot_gantt.py
Output: outputs/knee_gantt.html
"""

import json
import os
import math

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
ALGS    = ["MDLS", "NSGA-III", "MOPSO-CD"]
COLORS  = {
    "MDLS":      "#1f77b4",
    "NSGA-III":  "#2ca02c",
    "MOPSO-CD":  "#d62728",
}

# ── load schedules ────────────────────────────────────────────────────────────
def load_schedule(alg):
    path = os.path.join(OUT_DIR, f"knee_schedule_{alg}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

# ── build Plotly trace data as Python dicts (serialised to JSON inline) ───────
def build_traces(schedule_data, alg_color):
    """
    Returns a list of Plotly bar trace dicts for one algorithm.
    Each vehicle gets one trace (so legend groups work cleanly).
    """
    routes = schedule_data.get("schedule", [])
    traces = []

    for vi, route in enumerate(routes):
        sats  = route["visitedSAT"]
        uids  = route["visitedUID"]
        arrs  = route["arrivals"]
        deps  = route["departures"]

        for i, (sat, uid, arr, dep) in enumerate(zip(sats, uids, arrs, deps)):
            if dep <= arr:
                dep = arr + 0.3
            is_depot  = uid < 0
            color     = "#aec7e8" if is_depot else alg_color
            opacity   = 0.6 if is_depot else 0.9
            label     = "depot" if is_depot else sat
            hover     = (
                f"Vehicle {vi+1}<br>"
                f"{'Depot return' if is_depot else 'Service: ' + sat}<br>"
                f"Arrive: {arr:.2f} d<br>"
                f"Depart: {dep:.2f} d<br>"
                f"Duration: {dep-arr:.2f} d"
            )
            traces.append({
                "type": "bar",
                "name": f"V{vi+1}",
                "x": [dep - arr],
                "y": [f"V{vi+1:02d}"],
                "base": [arr],
                "orientation": "h",
                "marker": {"color": color, "opacity": opacity,
                           "line": {"color": "#333", "width": 0.4}},
                "hovertemplate": hover + "<extra></extra>",
                "showlegend": False,
            })

    return traces

# ── compute summary stats ──────────────────────────────────────────────────────
def schedule_stats(data):
    if data is None:
        return None
    routes    = data.get("schedule", [])
    unassigned = data.get("unassigned")

    n_veh  = len(routes)
    total_dv = 0.0
    for route in routes:
        total_dv += sum(route.get("costs", []))

    n_unassigned = 0
    if unassigned and isinstance(unassigned, dict):
        uids = unassigned.get("UIDs", [])
        n_unassigned = len(uids)

    span = 0.0
    for route in routes:
        deps = route.get("departures", [])
        if deps:
            span = max(span, max(deps))

    return {
        "n_vehicles": n_veh,
        "total_dv_ms": total_dv,
        "n_unassigned": n_unassigned,
        "mission_span_days": span,
    }

# ── HTML template ──────────────────────────────────────────────────────────────
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Knee-Point Schedule Gantt — {scenario}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Helvetica Neue", Arial, sans-serif;
          background: #f4f6f9; color: #222; }}
  header {{ background: #1a2a4a; color: #fff;
             padding: 18px 32px; }}
  header h1 {{ font-size: 1.4em; font-weight: 600; }}
  header p  {{ font-size: 0.85em; opacity: 0.75; margin-top: 4px; }}

  /* tabs */
  .tabs {{ display: flex; background: #223358; gap: 2px; padding: 0 32px; }}
  .tab  {{ padding: 10px 22px; cursor: pointer; color: #aac; font-size: 0.92em;
           border-radius: 4px 4px 0 0; transition: background 0.15s; }}
  .tab:hover  {{ background: #2e4475; color: #fff; }}
  .tab.active {{ background: #f4f6f9; color: #1a2a4a; font-weight: 700; }}

  /* panels */
  .panel {{ display: none; padding: 24px 32px; }}
  .panel.active {{ display: block; }}

  .stats-row {{ display: flex; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }}
  .stat-card  {{ background: #fff; border-radius: 8px; padding: 14px 20px;
                 box-shadow: 0 1px 4px rgba(0,0,0,.1); min-width: 160px; }}
  .stat-card .label {{ font-size: 0.72em; color: #666; text-transform: uppercase;
                       letter-spacing: .05em; }}
  .stat-card .value {{ font-size: 1.5em; font-weight: 700; margin-top: 2px; }}

  .chart-wrap {{ background: #fff; border-radius: 8px;
                 box-shadow: 0 1px 4px rgba(0,0,0,.1); padding: 8px; }}

  .missing {{ color: #a00; padding: 24px;
              background: #fff3f3; border-radius: 8px;
              border: 1px solid #f0b0b0; }}
  .legend-row {{ display: flex; gap: 20px; align-items: center;
                  margin-bottom: 10px; font-size: 0.82em; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 3px;
              display: inline-block; margin-right: 5px; }}
</style>
</head>
<body>
<header>
  <h1>Knee-Point Schedule Gantt Charts</h1>
  <p>Scenario: <strong>{scenario}</strong> &nbsp;|&nbsp;
     Trial: <strong>{trial}</strong> &nbsp;|&nbsp;
     Budget: <strong>{budget:,} evals</strong></p>
</header>

<div class="tabs" id="tabs">
{tab_buttons}
</div>

{panels}

<script>
const plotData = {plot_data_json};

function activateTab(idx) {{
  document.querySelectorAll('.tab').forEach((t,i) =>
    t.classList.toggle('active', i===idx));
  document.querySelectorAll('.panel').forEach((p,i) =>
    p.classList.toggle('active', i===idx));

  const key = Object.keys(plotData)[idx];
  const d   = plotData[key];
  if (d && !d.rendered) {{
    const layout = {{
      barmode: 'overlay',
      xaxis: {{ title: 'Time [days]', gridcolor: '#ddd', zeroline: false }},
      yaxis: {{ title: 'Vehicle', autorange: 'reversed',
                tickfont: {{ size: 11 }} }},
      margin: {{ l: 70, r: 30, t: 20, b: 50 }},
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#fff',
      height: Math.max(350, d.traces.length > 0 ?
              [...new Set(d.traces.map(t => t.y[0]))].length * 28 + 80 : 350),
      showlegend: false,
    }};
    Plotly.newPlot('chart-' + key, d.traces, layout,
                   {{responsive: true, displayModeBar: true}});
    d.rendered = true;
  }}
}}

// activate first tab on load
document.addEventListener('DOMContentLoaded', () => activateTab(0));
</script>
</body>
</html>
"""

def fmt_dv(v):
    if v >= 1e6:
        return f"{v/1e3:.0f} km/s"
    return f"{v:.0f} m/s"

def build_html(scenario="tight_normal", trial=1, budget=10_000):
    tab_buttons = ""
    panels_html = ""
    plot_data   = {}

    for ti, alg in enumerate(ALGS):
        data  = load_schedule(alg)
        stats = schedule_stats(data)
        color = COLORS[alg]
        active_cls = " active" if ti == 0 else ""

        tab_buttons += (
            f'  <div class="tab{active_cls}" onclick="activateTab({ti})">'
            f'{alg}</div>\n'
        )

        if data is None:
            panels_html += (
                f'<div class="panel{active_cls}" id="panel-{alg}">\n'
                f'  <div class="missing">⚠ Schedule file not found for {alg}.<br>'
                f'  Run: <code>julia --project=. -t auto export_knee_schedules.jl</code></div>\n'
                f'</div>\n'
            )
            plot_data[alg] = {"traces": [], "rendered": False}
            continue

        traces = build_traces(data, color)

        # stat cards
        cards = [
            ("Vehicles",        str(stats["n_vehicles"])),
            ("Total ΔV",        fmt_dv(stats["total_dv_ms"])),
            ("Unserviced",      str(stats["n_unassigned"])),
            ("Mission span",    f"{stats['mission_span_days']:.0f} days"),
        ]
        cards_html = ""
        for lbl, val in cards:
            cards_html += (
                f'<div class="stat-card">'
                f'<div class="label">{lbl}</div>'
                f'<div class="value">{val}</div>'
                f'</div>\n'
            )

        legend_html = (
            f'<div class="legend-row">'
            f'  <span><span class="swatch" style="background:{color};opacity:0.9"></span>'
            f'Service visit</span>'
            f'  <span><span class="swatch" style="background:#aec7e8;opacity:0.8"></span>'
            f'Depot return</span>'
            f'</div>'
        )

        panels_html += (
            f'<div class="panel{active_cls}" id="panel-{alg}">\n'
            f'  <div class="stats-row">{cards_html}</div>\n'
            f'  {legend_html}\n'
            f'  <div class="chart-wrap"><div id="chart-{alg}"></div></div>\n'
            f'</div>\n'
        )
        plot_data[alg] = {"traces": traces, "rendered": False}

    html = HTML_TEMPLATE.format(
        scenario=scenario,
        trial=trial,
        budget=budget,
        tab_buttons=tab_buttons,
        panels=panels_html,
        plot_data_json=json.dumps(plot_data),
    )
    return html


if __name__ == "__main__":
    html = build_html(scenario="tight_normal", trial=1, budget=10_000)
    out  = os.path.join(OUT_DIR, "knee_gantt.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved: {out}")
    # quick check: how many schedule files found
    found = [alg for alg in ALGS
             if os.path.isfile(os.path.join(OUT_DIR, f"knee_schedule_{alg}.json"))]
    missing = [a for a in ALGS if a not in found]
    if found:
        print(f"Schedules found: {found}")
    if missing:
        print(f"Missing (run export_knee_schedules.jl first): {missing}")
