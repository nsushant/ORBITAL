"""
plot_propulsion_trade.py — Architecture trade study using full Pareto front.

Produces three figures in outputs/:
  architecture_coverage_cost.pdf   — coverage % vs mission cost for each propulsion system
  architecture_breakeven.pdf       — break-even dry mass vs coverage for each propulsion system
  architecture_fleet_coverage.pdf  — fleet size vs coverage coloured by algorithm

Run: python plot_propulsion_trade.py
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

OUT       = "outputs"
FRONT_CSV = os.path.join(OUT, "starlink_pareto_fronts.csv")

# ── Propulsion systems ────────────────────────────────────────────────────────

# Propellant costs from Tirila et al. (2023), Acta Astronautica 212, 284-306.
# Xe/Kr: Table 2 USD/l converted via liquid density at boiling point
#   Xe: $1050/L ÷ 3.084 kg/L = $340/kg  (ρ_liq = 3.084 kg/L at 165 K)
#   Kr: $204/L  ÷ 2.413 kg/L = $85/kg   (ρ_liq = 2.413 kg/L at 120 K)
# I₂, Bi, Zn: Table 2 USD/kg direct (2021-2022 market data).
# Isp: practical values for a ~1 kW, ~300 V Hall thruster.
PROPULSION = {
    "Hall / Xe" : dict(Isp=2141, cost_kg=340),   # $340/kg  — Table 2; Isp — Table 5
    "Hall / Kr" : dict(Isp=2680, cost_kg=85),    # $85/kg   — Table 2; Isp — Table 5
    "Hall / I₂" : dict(Isp=2178, cost_kg=32),    # $32/kg   — Table 2; Isp — Table 5
    "Hall / Bi" : dict(Isp=1697, cost_kg=8),     # $8/kg    — Table 2; Isp — Table 5
    "Hall / Zn" : dict(Isp=3034, cost_kg=3),     # $3/kg    — Table 2; Isp — Table 5
}
G0            = 9.80665
DV_BUDGET     = 5000.0   # m/s per depot leg
N_DEMANDS     = 200
SERVICE_TIME  = 3.0      # days (constant service time per satellite)

# Falcon 9 launch model (SpaceX website)
F9_PAYLOAD_KG    = 22_000.0      # kg to LEO
F9_LAUNCH_COST   = 74_000_000.0  # $ for a dedicated Falcon 9
RIDESHARE_MARKET = F9_LAUNCH_COST / F9_PAYLOAD_KG   # $3,364/kg

# Starlink asset value (weighted average: 114 V1 × $770K + 86 V2mini × $1.77M) / 200
ASSET_VALUE     = 1_200_000                              # $ per satellite
LAUNCH_COST_SAT = round(RIDESHARE_MARKET * 0.5) * 200   # 50% of rideshare rate × 200 kg (~$336K)
ASSET_TOTAL     = ASSET_VALUE + LAUNCH_COST_SAT          # ~$1.54M per satellite

ALG_NAMES  = ["MDLS", "NSGA-III", "MOEA/D", "PSO"]
COLOURS    = {"MDLS": "#1f77b4", "NSGA-III": "#ff7f0e",
              "MOEA/D": "#2ca02c", "PSO": "#d62728"}
PROP_COLOURS = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7", "#e7ba52"]

FS       = 14
FS_TICK  = 12
FS_TITLE = 15

DRY_MASS_BASELINE = 300.0   # kg — servicer dry mass

# ── Physics helpers ───────────────────────────────────────────────────────────

def v_exhaust(Isp):
    return Isp * G0

def prop_per_leg(m_dry, dv, ve):
    return m_dry * (math.exp(dv / ve) - 1)

def wet_per_leg(m_dry, dv, ve):
    return m_dry * math.exp(dv / ve)

def _vehicle_masses(m_dry, f1_total, f3, system):
    """Shared propulsion computation. Returns (m_prop_per_veh, m_wet, dv_per_sortie)."""
    ve             = v_exhaust(system["Isp"])
    dv_per_vehicle = f1_total / max(f3, 1)
    n_sorties      = max(1, math.ceil(dv_per_vehicle / DV_BUDGET))
    dv_per_sortie  = dv_per_vehicle / n_sorties
    m_prop_per_veh = n_sorties * prop_per_leg(m_dry, dv_per_sortie, ve)
    m_wet          = wet_per_leg(m_dry, dv_per_sortie, ve)
    return m_prop_per_veh, m_wet, dv_per_sortie


def total_cost(m_dry, f1_total, f3, system):
    """Mission cost: propellant + dedicated Falcon 9 launch (no rideshare revenue)."""
    m_prop_per_veh, m_wet, _ = _vehicle_masses(m_dry, f1_total, f3, system)
    c_prop   = f3 * m_prop_per_veh * system["cost_kg"]
    c_launch = F9_LAUNCH_COST
    return c_prop + c_launch, f3 * m_prop_per_veh, m_wet


def breakeven_rideshare(m_dry, f1_total, f3, n_serviced, system):
    """$/kg that remaining Falcon 9 payload must sell for to achieve BCR = 1.

    Negative  → already profitable with no rideshare revenue.
    > RIDESHARE_MARKET → cannot break even even at current market rates.
    inf → fleet too heavy to fit on a single F9.
    """
    m_prop_per_veh, m_wet, _ = _vehicle_masses(m_dry, f1_total, f3, system)
    total_mass     = f3 * m_wet
    remaining_mass = F9_PAYLOAD_KG - total_mass
    if remaining_mass <= 0:
        return float("inf")
    value  = n_serviced * ASSET_TOTAL
    c_prop = f3 * m_prop_per_veh * system["cost_kg"]
    # BCR=1: value + r*remaining = F9_LAUNCH_COST + c_prop
    return (F9_LAUNCH_COST + c_prop - value) / remaining_mass

# ── New helpers ───────────────────────────────────────────────────────────────

def coverage_pct(f2):
    return max(0.0, (N_DEMANDS - f2 / SERVICE_TIME) / N_DEMANDS * 100.0)

def breakeven_drymass(f1, f2, f3, system):
    """Dry mass at which BCR = 1 for this Pareto point and propulsion system."""
    ve = v_exhaust(system["Isp"])
    n_serviced = max(0.0, N_DEMANDS - f2 / SERVICE_TIME)
    value = n_serviced * ASSET_TOTAL
    if value <= 0 or f1 <= 0:
        return float("inf")
    n_ret         = math.ceil(f1 / DV_BUDGET)
    prop_factor   = n_ret * (math.exp(DV_BUDGET / ve) - 1) * system["cost_kg"]
    launch_factor = f3 * math.exp(DV_BUDGET / ve) * LAUNCH_COST
    denom = prop_factor + launch_factor
    return value / denom if denom > 0 else float("inf")

# ── KDE contour helper (borrowed from plot_shadow.py) ────────────────────────

MIN_KDE_POINTS = 30   # lower threshold — Starlink fronts are sparser

def plot_kde(ax, x, y, colour):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 5:
        return
    if len(x) < MIN_KDE_POINTS:
        ax.scatter(x, y, c=colour, s=20, alpha=0.5, linewidths=0)
        return
    try:
        kde    = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xg     = np.linspace(x.min(), x.max(), 120)
        yg     = np.linspace(y.min(), y.max(), 120)
        X, Y   = np.meshgrid(xg, yg)
        Z      = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        z_flat = Z[Z > Z.max() * 0.05]
        levels = np.unique(np.percentile(z_flat, [40, 65, 82, 93, 98]))
        if len(levels) < 2:
            ax.scatter(x, y, c=colour, s=20, alpha=0.5, linewidths=0)
            return
        ax.contour(X, Y, Z, levels=levels, colors=[colour], alpha=0.85, linewidths=1.5)
    except Exception:
        ax.scatter(x, y, c=colour, s=20, alpha=0.5, linewidths=0)

# ── Load data ─────────────────────────────────────────────────────────────────

if not os.path.exists(FRONT_CSV):
    print(f"[error] {FRONT_CSV} not found — run starlink/run_starlink.jl first")
    exit(1)

pf = pd.read_csv(FRONT_CSV)
pf["coverage"] = pf["f2_unassigned_time"].apply(coverage_pct)

# ── BCR curve: BCR vs coverage for each propulsion system ────────────────────
# For every Pareto point compute BCR, then plot vs coverage.
# One subplot per propellant; one line per algorithm (thin per trial, thick median).

from scipy.stats import binned_statistic

def point_bcr(row, sys):
    n_serv = max(0.0, N_DEMANDS - row["f2_unassigned_time"] / SERVICE_TIME)
    value  = n_serv * ASSET_TOTAL
    cost, _, _ = total_cost(DRY_MASS_BASELINE, row["f1_dv"], row["f3_vehicles"], sys)
    return value / cost if cost > 0 else np.nan

COV_GRID   = np.linspace(0, 100, 60)
N_COLS_BCR = 3
N_ROWS_BCR = math.ceil(len(PROPULSION) / N_COLS_BCR)
fig_bcr, axes_bcr = plt.subplots(N_ROWS_BCR, N_COLS_BCR,
                                  figsize=(6 * N_COLS_BCR, 5 * N_ROWS_BCR))

for ax, (sname, sys) in zip(np.array(axes_bcr).flat, PROPULSION.items()):
    pf["_bcr"] = pf.apply(lambda r: point_bcr(r, sys), axis=1)

    for alg in ALG_NAMES:
        sub = pf[pf["algorithm"] == alg].dropna(subset=["_bcr"])
        col = COLOURS[alg]

        # thin lines per trial
        for _, tgrp in sub.groupby("trial"):
            tgrp_s = tgrp.sort_values("coverage")
            ax.plot(tgrp_s["coverage"], tgrp_s["_bcr"],
                    color=col, alpha=0.2, lw=0.8)

        # median line across all trials binned by coverage
        med, edges, _ = binned_statistic(sub["coverage"], sub["_bcr"],
                                          statistic="median", bins=COV_GRID)
        cen  = 0.5 * (edges[:-1] + edges[1:])
        mask = ~np.isnan(med)
        ax.plot(cen[mask], med[mask], color=col, lw=2, label=alg)

    ax.axhline(1.0, color="grey", lw=1.2, ls="--")
    ax.set_xlabel("Coverage (%)", fontsize=FS)
    ax.set_ylabel("BCR", fontsize=FS)
    ax.set_title(sname, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=FS_TICK - 1)

# hide unused subplots
for ax in list(np.array(axes_bcr).flat)[len(PROPULSION):]:
    ax.set_visible(False)

fig_bcr.suptitle(f"BCR vs Coverage  (dry mass = {int(DRY_MASS_BASELINE)} kg)",
                  fontsize=FS_TITLE + 1)
fig_bcr.tight_layout()
fig_bcr.savefig(os.path.join(OUT, "architecture_bcr_curve.pdf"), bbox_inches="tight")
plt.close(fig_bcr)
print("Saved architecture_bcr_curve.pdf")

# ── BCR summary table (LaTeX) ─────────────────────────────────────────────────
# For each (algorithm, propellant): mean ± std of the best BCR per trial.
# "Best BCR per trial" = max BCR across all Pareto points in that trial.

prop_short = {
    "Hall / Xe": r"\ce{Xe}",
    "Hall / Kr": r"\ce{Kr}",
    "Hall / I₂": r"\ce{I_2}",
    "Hall / Bi": r"\ce{Bi}",
    "Hall / Zn": r"\ce{Zn}",
}

lines = []
lines.append(r"\begin{tabular}{l" + "c" * len(PROPULSION) + r"}")
lines.append(r"\toprule")
header = "Algorithm & " + " & ".join(prop_short[s] for s in PROPULSION) + r" \\"
lines.append(header)
lines.append(r"\midrule")

for alg in ALG_NAMES:
    row_cells = [alg]
    for sname, sys in PROPULSION.items():
        pf["_bcr"] = pf.apply(lambda r: point_bcr(r, sys), axis=1)
        sub = pf[pf["algorithm"] == alg].dropna(subset=["_bcr"])
        # best (max) BCR per trial
        trial_best = sub.groupby("trial")["_bcr"].max()
        if len(trial_best) == 0:
            row_cells.append("---")
        else:
            mu  = trial_best.mean()
            sig = trial_best.std(ddof=1) if len(trial_best) > 1 else 0.0
            row_cells.append(f"{mu:.2f} $\\pm$ {sig:.2f}")
    lines.append(" & ".join(row_cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

table_str = "\n".join(lines)
table_path = os.path.join(OUT, "bcr_table.tex")
with open(table_path, "w") as f:
    f.write(table_str)
print("Saved bcr_table.tex")
print("\nBCR table (best per trial, mean ± std across trials):")
print(table_str)

# ── Breakeven rideshare price vs coverage ────────────────────────────────────
# For each Pareto point: what $/kg must the remaining F9 payload sell for so
# that the servicer operator breaks even (BCR = 1)?
# Horizontal bands: < 0 → already profitable; > RIDESHARE_MARKET → infeasible.

def point_rideshare(row, sys):
    n_serv = max(0.0, N_DEMANDS - row["f2_unassigned_time"] / SERVICE_TIME)
    return breakeven_rideshare(DRY_MASS_BASELINE, row["f1_dv"],
                               row["f3_vehicles"], n_serv, sys)

fig_rs, axes_rs = plt.subplots(N_ROWS_BCR, N_COLS_BCR,
                                figsize=(6 * N_COLS_BCR, 5 * N_ROWS_BCR))

for ax, (sname, sys) in zip(np.array(axes_rs).flat, PROPULSION.items()):
    pf["_rs"] = pf.apply(lambda r: point_rideshare(r, sys), axis=1)
    pf_fin    = pf[np.isfinite(pf["_rs"])]

    for alg in ALG_NAMES:
        sub = pf_fin[pf_fin["algorithm"] == alg]
        col = COLOURS[alg]
        for _, tgrp in sub.groupby("trial"):
            tgrp_s = tgrp.sort_values("coverage")
            ax.plot(tgrp_s["coverage"], tgrp_s["_rs"] / 1e3,
                    color=col, alpha=0.2, lw=0.8)
        med, edges, _ = binned_statistic(sub["coverage"], sub["_rs"] / 1e3,
                                          statistic="median", bins=COV_GRID)
        cen  = 0.5 * (edges[:-1] + edges[1:])
        mask = ~np.isnan(med)
        ax.plot(cen[mask], med[mask], color=col, lw=2, label=alg)

    ax.axhline(0,                       color="green", lw=1.4, ls="--",
               label="No rideshare needed (r* ≤ 0)")
    ax.axhline(RIDESHARE_MARKET / 1e3, color="red",   lw=1.4, ls="--",
               label=f"Market rate (${RIDESHARE_MARKET:,.0f}/kg)")
    ax.fill_between([0, 100], 0, RIDESHARE_MARKET / 1e3,
                    color="green", alpha=0.08, label="Viable rideshare window")
    ax.set_xlabel("Coverage (%)", fontsize=FS)
    ax.set_ylabel("Breakeven rideshare price (k$/kg)", fontsize=FS)
    ax.set_title(sname, fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=FS_TICK - 1)

for ax in list(np.array(axes_rs).flat)[len(PROPULSION):]:
    ax.set_visible(False)

fig_rs.suptitle(
    f"Breakeven rideshare price vs Coverage  "
    f"(dry mass = {int(DRY_MASS_BASELINE)} kg, F9 = \${int(F9_LAUNCH_COST/1e6)}M / {int(F9_PAYLOAD_KG)} kg)\n"
    f"Negative r* = already profitable without rideshare revenue",
    fontsize=FS_TITLE)
fig_rs.tight_layout()
fig_rs.savefig(os.path.join(OUT, "architecture_rideshare_breakeven.pdf"), bbox_inches="tight")
plt.close(fig_rs)
print("Saved architecture_rideshare_breakeven.pdf")

# ── Plot 1: Coverage–cost efficient frontier ──────────────────────────────────
# For each propulsion system: scatter all Pareto points in (coverage%, mission cost $M)
# at DRY_MASS_BASELINE. Overlay break-even diagonal (cost = value recovered).

fig, axes = plt.subplots(N_ROWS_BCR, N_COLS_BCR, figsize=(6 * N_COLS_BCR, 5 * N_ROWS_BCR))

for ax, (sname, sys) in zip(np.array(axes).flat, PROPULSION.items()):
    for alg in ALG_NAMES:
        sub  = pf[pf["algorithm"] == alg]
        cov  = sub["coverage"]
        cost = sub.apply(
            lambda r: total_cost(DRY_MASS_BASELINE, r.f1_dv, r.f3_vehicles, sys)[0] / 1e6,
            axis=1)
        plot_kde(ax, cov.values, cost.values, COLOURS[alg])

    # Break-even line: cost = value_recovered
    cvs = np.linspace(0, 100, 300)
    be  = cvs / 100.0 * N_DEMANDS * ASSET_TOTAL / 1e6
    ax.plot(cvs, be, "k--", lw=1.4, label="Break-even", zorder=4)

    ax.set_title(sname.replace("\n", " "), fontsize=FS_TITLE)
    ax.set_xlabel("Coverage [%]", fontsize=FS)
    ax.set_ylabel("Mission cost [M$]", fontsize=FS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linewidth=0.4, alpha=0.5)

# hide unused subplots
for ax in list(np.array(axes).flat)[len(PROPULSION):]:
    ax.set_visible(False)

# Shared legend
handles = [Line2D([0], [0], color=COLOURS[a], lw=2, label=a) for a in ALG_NAMES]
handles += [Line2D([0], [0], color="black", linestyle="--", lw=1.4, label="Break-even")]
fig.legend(handles=handles, loc="lower center", ncol=len(ALG_NAMES) + 1,
           fontsize=FS_TICK, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_coverage_cost.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: architecture_coverage_cost.pdf")

# ── Plot 2: Break-even dry mass vs coverage ───────────────────────────────────
# For each Pareto point and propulsion system, compute the dry mass at which
# BCR = 1. Points below DRY_MASS_BASELINE are already profitable at 300 kg.

MAX_BDM = 2000.0   # cap for display (infinite break-even filtered out)

fig, ax = plt.subplots(figsize=(9, 6))

for (sname, sys), col in zip(PROPULSION.items(), PROP_COLOURS):
    bdm = pf.apply(
        lambda r: breakeven_drymass(r.f1_dv, r.f2_unassigned_time, r.f3_vehicles, sys),
        axis=1)
    mask = bdm < MAX_BDM
    ax.scatter(pf.loc[mask, "coverage"], bdm[mask],
               color=col, alpha=0.35, s=10,
               label=sname.replace("\n", " "))

ax.axhline(DRY_MASS_BASELINE, color="grey", linestyle="--", lw=1.4,
           label=f"Baseline ({int(DRY_MASS_BASELINE)} kg)")
ax.fill_between([0, 100], 0, DRY_MASS_BASELINE,
                color="grey", alpha=0.08, label="Profitable at baseline")

ax.set_xlabel("Coverage [%]", fontsize=FS)
ax.set_ylabel("Break-even dry mass [kg]", fontsize=FS)
ax.set_xlim(0, 100)
ax.set_ylim(0, MAX_BDM)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.legend(fontsize=FS_TICK, loc="upper left")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_breakeven.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: architecture_breakeven.pdf")

# ── Plot 3: Fleet size vs coverage ────────────────────────────────────────────
# Shows how many vehicles are needed to achieve a given coverage level.

fig, ax = plt.subplots(figsize=(9, 6))

for alg in ALG_NAMES:
    sub = pf[pf["algorithm"] == alg]
    plot_kde(ax, sub["coverage"].values, sub["f3_vehicles"].values.astype(float), COLOURS[alg])

ax.set_xlabel("Coverage [%]", fontsize=FS)
ax.set_ylabel("Number of servicer vehicles", fontsize=FS)
ax.set_xlim(0, 100)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, linewidth=0.4, alpha=0.5)
handles = [Line2D([0], [0], color=COLOURS[a], lw=2, label=a) for a in ALG_NAMES]
ax.legend(handles=handles, fontsize=FS_TICK)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_fleet_coverage.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: architecture_fleet_coverage.pdf")

# ── Plot 4: Coverage vs ΔV ────────────────────────────────────────────────────
# Shows where each algorithm's solutions sit in (coverage %, total ΔV) space.
# MDLS concentrated at high coverage + low ΔV explains its larger hypervolume.

fig, ax = plt.subplots(figsize=(9, 6))

for alg in ALG_NAMES:
    sub = pf[pf["algorithm"] == alg]
    plot_kde(ax, sub["coverage"].values, sub["f1_dv"].values.astype(float), COLOURS[alg])

ax.set_xlabel("Coverage [%]", fontsize=FS)
ax.set_ylabel("Total ΔV [m/s]", fontsize=FS)
ax.set_xlim(0, 100)
ax.tick_params(labelsize=FS_TICK)
ax.grid(True, linewidth=0.4, alpha=0.5)
handles = [Line2D([0], [0], color=COLOURS[a], lw=2, label=a) for a in ALG_NAMES]
ax.legend(handles=handles, fontsize=FS_TICK)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_coverage_dv.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: architecture_coverage_dv.pdf")

# ── Plot 5: Per-algorithm filled KDE — coverage vs ΔV ─────────────────────────
# 2×2 grid, one panel per algorithm. Filled contours (contourf) show solution
# density in (coverage %, ΔV) space. Reveals that MDLS clusters at the
# high-coverage / low-ΔV corner — the dominant region — explaining larger HV.

from scipy.stats import binned_statistic_2d

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

cov_all = pf["coverage"].values
veh_all = pf["f3_vehicles"].values.astype(float)
dv_all  = pf["f1_dv"].values.astype(float)
xmin, xmax = cov_all.min(), cov_all.max()
ymin, ymax = veh_all.min(), veh_all.max()
NBINS = 20

for ax, alg in zip(axes.flat, ALG_NAMES):
    sub  = pf[pf["algorithm"] == alg]
    x    = sub["coverage"].values
    y    = sub["f3_vehicles"].values.astype(float)
    dv   = sub["f1_dv"].values.astype(float)

    ax.set_title(alg, fontsize=FS_TITLE)
    ax.set_xlabel("Coverage [%]", fontsize=FS)
    ax.set_ylabel("Number of servicer vehicles", fontsize=FS)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    stat, xe, ye, _ = binned_statistic_2d(
        x, y, dv, statistic="median", bins=NBINS,
        range=[[xmin, xmax], [ymin, ymax]])

    # mask empty bins
    stat = np.ma.masked_invalid(stat)

    Xc = 0.5 * (xe[:-1] + xe[1:])
    Yc = 0.5 * (ye[:-1] + ye[1:])
    Xg, Yg = np.meshgrid(Xc, Yc, indexing="ij")

    cf = ax.contourf(Xg, Yg, stat, levels=12, cmap="plasma_r")
    fig.colorbar(cf, ax=ax, label="Median ΔV [m/s]")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "architecture_coverage_dv_filled.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: architecture_coverage_dv_filled.pdf")

print("\nAll architecture trade plots complete.")
