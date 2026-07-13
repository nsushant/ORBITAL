"""
plot_bcr_mass_trade.py — BCR vs servicer dry mass trade study.

For each candidate servicer dry mass, compute BCR for the best-f2 solution
at each dv_budget level (using fixed MDLS results, post-hoc cost model).
Plot: x = servicer dry mass [kg], y = number of budget levels with BCR > 1.

Alpha fixed at 0.6 (moderate complexity), N=3 vehicles in production run.

Run: python plot_bcr_mass_trade.py --h5 outputs/long_horizon_results.h5
"""

import os, math, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--h5",      default="outputs/long_horizon_results.h5")
parser.add_argument("--out-dir", default="outputs")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G0          = 9.80665
ISP_XE      = 2800.0
VE_XE       = ISP_XE * G0
XE_COST     = 340.0          # $/kg xenon
F9_COST     = 74_000_000.0   # $ per Falcon 9 launch
F9_CAPACITY = 22_000.0       # kg to LEO
MARKET_RATE = F9_COST / F9_CAPACITY
INFLATION   = 2.005          # FY2000 -> FY2026 (NASA NNSI 2026)
ALPHA        = 0.6    # payload complexity multiplier
OPERATOR_FEE = 0.10   # operator receives 10% of recovered satellite value as revenue
# N in learning curve = f3 (fleet size) — computed per solution

BUDGETS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000]

# ---------------------------------------------------------------------------
def sscm_unit_cost(m_dry, alpha, N):
    c_bus_fy00k = 781.0 + 26.1 * (m_dry ** 1.261)
    c_total     = c_bus_fy00k * 1000.0 * INFLATION * (1.0 + alpha)
    nre         = 0.6 * c_total
    rc1         = 0.4 * c_total
    rc_avg      = rc1 * (N ** (-0.578))
    return nre / N + rc_avg

def mission_costs(f1, f2, f3, dv_budget, m_dry):
    """Operator mission costs: launch + manufacturing + propellant."""
    n      = max(1, int(round(f3)))
    m_prop = m_dry * (math.exp(dv_budget / VE_XE) - 1.0)
    m_wet  = m_dry + m_prop
    c_veh  = sscm_unit_cost(m_dry, ALPHA, n)
    return (f3 * m_wet * MARKET_RATE
            + (f1 / dv_budget) * m_prop * XE_COST
            + f3 * c_veh)

def compute_bcr(f1, f2, f3, dv_budget, total_demand_value, m_dry):
    """Operator BCR = (10% of recovered value) / mission costs."""
    recovered = max(0.0, total_demand_value - f2)
    revenue   = OPERATOR_FEE * recovered
    costs     = mission_costs(f1, f2, f3, dv_budget, m_dry)
    return revenue / costs if costs > 0 else np.nan

def compute_client_bcr(f1, f2, f3, dv_budget, total_demand_value, m_dry):
    """Client BCR = replacement cost saved / (OOS fee + remaining replacements)."""
    recovered     = max(0.0, total_demand_value - f2)
    oos_fee       = OPERATOR_FEE * recovered   # what client pays operator
    remaining     = max(0.0, f2)               # unserviced satellites still need replacing
    client_costs  = oos_fee + remaining
    client_benefit = recovered
    return client_benefit / client_costs if client_costs > 0 else np.nan

def compute_bcr_old(f1, f2, f3, dv_budget, total_demand_value, m_dry):
    """Legacy: BCR = recovered / costs (no fee split) — kept for reference."""
    n         = max(1, int(round(f3)))
    m_prop    = m_dry * (math.exp(dv_budget / VE_XE) - 1.0)
    m_wet     = m_dry + m_prop
    recovered = max(0.0, total_demand_value - f2)
    return recovered / costs if costs > 0 else np.nan  # legacy fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_best_f2(data):
    """Minimum unrecovered value (most demands served)."""
    return data[np.argmin(data[:, 1])]

def select_knee(data):
    """Closest to ideal [0,0,0] in within-front normalised space."""
    lo  = data.min(axis=0)
    hi  = data.max(axis=0)
    rng = np.maximum(hi - lo, 1e-10)
    norm = (data - lo) / rng
    return data[np.argmin((norm ** 2).sum(axis=1))]

# ---------------------------------------------------------------------------
# Load solutions per budget (both selectors)
# ---------------------------------------------------------------------------
best_f2_solutions  = {}   # budget -> (f1, f2, f3, tdv)
knee_solutions     = {}

with h5py.File(args.h5, "r") as f:
    for b in BUDGETS:
        path = f"mdls/bcr_mixed_{b}"
        if path not in f:
            continue
        grp = f[path]
        for tk in sorted(grp.keys()):
            ds   = grp[tk]
            data = ds[:]
            if data.shape[0] == 3 and data.shape[1] != 3:
                data = data.T
            tdv  = float(ds.attrs.get("total_demand_value", 0.0))
            data = data[data[:, 0] < 1e6]
            if len(data) == 0:
                continue
            f1, f2, f3 = select_best_f2(data)
            best_f2_solutions[b] = (f1, f2, f3, tdv)
            f1k, f2k, f3k = select_knee(data)
            knee_solutions[b] = (f1k, f2k, f3k, tdv)

print(f"Loaded {len(best_f2_solutions)} budget levels")
for b in sorted(best_f2_solutions):
    f1,f2,f3,tdv = best_f2_solutions[b]
    f1k,f2k,f3k,_ = knee_solutions[b]
    print(f"  dv={b:5d}  best-f2: f3={f3:.0f} f2=${f2/1e6:.0f}M  |  knee: f3={f3k:.0f} f2=${f2k/1e6:.0f}M")

# ---------------------------------------------------------------------------
# BCR=1 crossover stats for best-f2 solutions
# ---------------------------------------------------------------------------
from scipy.optimize import brentq

def bcr_minus1(m, f1, f2, f3, dv, tdv):
    n   = max(1, int(round(f3)))
    mp  = m * (math.exp(dv / VE_XE) - 1)
    mw  = m + mp
    rec = max(0.0, tdv - f2)
    cost = f3*mw*MARKET_RATE + (f1/dv)*mp*XE_COST + f3*sscm_unit_cost(m, ALPHA, n)
    return rec / cost - 1.0

crossover_rows = []
print("\nBCR=1 crossover (max-coverage solutions):")
for b in BUDGETS:
    if b not in best_f2_solutions:
        continue
    f1, f2, f3, tdv = best_f2_solutions[b]
    try:
        m_star = brentq(bcr_minus1, 10, 2000, args=(f1, f2, f3, b, tdv), xtol=0.5)
        mp     = m_star * (math.exp(b / VE_XE) - 1)
        crossover_rows.append((b, m_star, f3, mp))
        print(f"  dv={b:5d}  M_DRY={m_star:.0f}kg  f3={f3:.0f}  m_prop={mp:.1f}kg")
    except Exception:
        print(f"  dv={b:5d}  no BCR=1 crossover in range")

if crossover_rows:
    cr = np.array(crossover_rows)
    print(f"\n  M_DRY:  {cr[:,1].min():.0f}–{cr[:,1].max():.0f} kg  "
          f"mean={cr[:,1].mean():.0f}  std={cr[:,1].std():.0f}")
    print(f"  f3:     {cr[:,2].min():.0f}–{cr[:,2].max():.0f}  mean={cr[:,2].mean():.0f}")
    print(f"  m_prop: {cr[:,3].min():.0f}–{cr[:,3].max():.0f} kg  mean={cr[:,3].mean():.0f}")

# ---------------------------------------------------------------------------
# Sweep dry mass for both selectors
# ---------------------------------------------------------------------------
mass_range = np.concatenate([
    np.arange(50,  500,  25),
    np.arange(500, 2001, 50),
]).astype(int)

def sweep(solutions, bcr_fn):
    bcr_by_mass = {}
    for m in mass_range:
        bcrs = []
        for b in BUDGETS:
            if b not in solutions:
                continue
            f1, f2, f3, tdv = solutions[b]
            bcrs.append(bcr_fn(f1, f2, f3, b, tdv, float(m)))
        bcr_by_mass[m] = bcrs
    return bcr_by_mass

# operator and client BCRs for best-f2 solutions
op_f2     = sweep(best_f2_solutions, compute_bcr)
client_f2 = sweep(best_f2_solutions, compute_client_bcr)

# ---------------------------------------------------------------------------
# Plot: operator BCR and client BCR as two panels
# ---------------------------------------------------------------------------
FS     = 22
masses = np.array(mass_range, dtype=float)

def stats(bcr_by_mass):
    mean = np.array([np.mean(bcr_by_mass[m]) for m in mass_range])
    lo   = np.array([np.min(bcr_by_mass[m])  for m in mass_range])
    hi   = np.array([np.max(bcr_by_mass[m])  for m in mass_range])
    return mean, lo, hi

op_mean, op_lo, op_hi = stats(op_f2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(masses, op_lo, op_hi, alpha=0.2, color="#1f77b4")
ax.plot(masses, op_mean, color="#1f77b4", linewidth=2.5)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Servicer dry mass [kg]", fontsize=FS)
ax.set_ylabel("BCR", fontsize=FS)
ax.tick_params(labelsize=FS - 2)
ax.set_xlim(masses[0], masses[-1])
ax.set_ylim(bottom=0)

fig.tight_layout()
outpath = os.path.join(args.out_dir, "bcr_mass_trade.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
