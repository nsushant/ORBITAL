"""
radar_plots.py
==============
Renders a 2×2 grid of radar (spider) charts — one per algorithm — showing
the min / median / max of the three Pareto-front objectives:

    f1  ΔV (m/s)                       → minimise
    f2  Unassigned service time (days)  → minimise
    f3  Vehicles used                   → minimise

All axes are normalised to [0, 1] using global bounds across all four
algorithms, so inner = better, outer = worse.

Output: outputs/radar_plots.html  (interactive Plotly)
Run:    python3 radar_plots.py
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────────
ALGORITHMS = {
    "MDLS":     "outputs/ga_pareto_mdls.csv",
    "NSGA-III": "outputs/ga_pareto_nsga3.csv",
    "MOEA/D":   "outputs/ga_pareto_moead.csv",
    "PSO":      "outputs/ga_pareto_pso.csv",
}

COLS         = ["f1_dv", "f2_unassigned_time", "f3_vehicles"]
THETA_LABELS = ["ΔV (m/s)", "Unassigned Time (days)", "Vehicles"]

COLOURS = {
    "MDLS":     "#1f77b4",
    "NSGA-III": "#ff7f0e",
    "MOEA/D":   "#2ca02c",
    "PSO":      "#d62728",
}

# Stat layers: outermost first so fills stack correctly
STAT_NAMES  = ["Max (worst)", "Median", "Min (best)"]
STAT_ALPHAS = [0.12,           0.30,     0.60]
STAT_WIDTHS = [1,               2,        2.5]

OUT_PATH = os.path.join("outputs", "radar_plots.html")

# ── Utilities ──────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_stats(norm_df: pd.DataFrame) -> dict[str, list[float]]:
    """Return {stat_name: [v_f1, v_f2, v_f3]} for normalised objectives."""
    return {
        "Max (worst)": norm_df[COLS].max().tolist(),
        "Median":      norm_df[COLS].median().tolist(),
        "Min (best)":  norm_df[COLS].min().tolist(),
    }


# ── Step 1: load & deduplicate ─────────────────────────────────────────────────
frames: dict[str, pd.DataFrame] = {}
for alg, path in ALGORITHMS.items():
    df = (
        pd.read_csv(path, usecols=COLS)
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .drop_duplicates()
    )
    frames[alg] = df.reset_index(drop=True)
    print(f"{alg}: {len(frames[alg])} unique solutions loaded")

# ── Step 2: global normalisation bounds (shared across all algorithms) ─────────
all_pts = pd.concat(frames.values(), ignore_index=True)
g_min   = all_pts[COLS].min()
g_max   = all_pts[COLS].max()
g_range = (g_max - g_min).replace(0, 1)  # avoid division by zero

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to [0, 1] — 0 = best (lowest), 1 = worst (highest)."""
    return (df[COLS] - g_min) / g_range


# ── Step 3: build figure ───────────────────────────────────────────────────────
positions  = [(1, 1), (1, 2), (2, 1), (2, 2)]
polar_keys = ["polar", "polar2", "polar3", "polar4"]

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "polar"}] * 2, [{"type": "polar"}] * 2],
    subplot_titles=list(ALGORITHMS.keys()),
    horizontal_spacing=0.10,
    vertical_spacing=0.14,
)

for idx, (alg, (row, col)) in enumerate(zip(ALGORITHMS, positions)):
    norm_df = normalise(frames[alg])
    stats   = compute_stats(norm_df)
    colour  = COLOURS[alg]
    polar_n = "polar" if idx == 0 else f"polar{idx + 1}"
    theta   = THETA_LABELS + [THETA_LABELS[0]]   # close the polygon

    for stat_name, alpha, width in zip(STAT_NAMES, STAT_ALPHAS, STAT_WIDTHS):
        r_vals = stats[stat_name] + [stats[stat_name][0]]  # close polygon

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta,
                fill="toself",
                fillcolor=hex_to_rgba(colour, alpha),
                line=dict(color=colour, width=width),
                name=stat_name,
                legendgroup=stat_name,
                showlegend=(idx == 0),   # show legend items once
                subplot=polar_n,
            ),
            row=row, col=col,
        )

# ── Step 4: uniform radial axis across all subplots ───────────────────────────
radial_cfg = dict(
    range=[0, 1],
    tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
    ticktext=["0\n(best)", "0.25", "0.5", "0.75", "1\n(worst)"],
    showticklabels=True,
    tickfont=dict(size=9),
    gridcolor="lightgrey",
    linecolor="lightgrey",
)
angular_cfg = dict(tickfont=dict(size=11))

for key in polar_keys:
    fig.update_layout(**{key: dict(radialaxis=radial_cfg, angularaxis=angular_cfg)})

fig.update_layout(
    title=dict(
        text=(
            "Pareto Front Objectives — min / median / max per algorithm<br>"
            "<sup>Axes normalised to global bounds  ·  0 = best, 1 = worst</sup>"
        ),
        x=0.5,
        xanchor="center",
        font=dict(size=15),
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.04,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    height=800,
    width=900,
    paper_bgcolor="white",
)

# ── Step 5: save & show ────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
fig.write_html(OUT_PATH)
print(f"\nSaved → {OUT_PATH}")
fig.show()
