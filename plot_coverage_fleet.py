"""
plot_coverage_fleet.py — Fleet size vs coverage, client BCR, and operator BCR (three panels).

Left panel:   fleet size vs demand coverage %
Middle panel: fleet size vs client BCR
Right panel:  fleet size vs operator BCR
              band = range over dry masses 50–1000 kg (BCR>1 regime from mass trade study)

Client BCR   = recovered / (0.1 * recovered + f2)
Operator BCR = 0.1 * recovered / mission_costs(m_dry)

Run: python plot_coverage_fleet.py --h5 outputs/long_horizon_results.h5
"""

import os, math, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--h5",       default="outputs/long_horizon_results.h5")
parser.add_argument("--out-dir",  default="outputs")
parser.add_argument("--scenario", default="bcr_mixed")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

BUDGETS      = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000]
OPERATOR_FEE = 0.10

# ---------------------------------------------------------------------------
# BCR constants and helpers (operator, mass-dependent)
# ---------------------------------------------------------------------------
G0          = 9.80665
ISP_XE      = 2800.0
VE_XE       = ISP_XE * G0
XE_COST     = 340.0
F9_COST     = 74_000_000.0
F9_CAPACITY = 22_000.0
MARKET_RATE = F9_COST / F9_CAPACITY
INFLATION   = 2.005
ALPHA       = 0.6

M_DRY = 150.0   # kg — representative servicer dry mass

def sscm_unit_cost(m_dry, N):
    c_bus   = 781.0 + 26.1 * (m_dry ** 1.261)
    c_total = c_bus * 1000.0 * INFLATION * (1.0 + ALPHA)
    nre     = 0.6 * c_total
    rc1     = 0.4 * c_total
    return nre / N + rc1 * (N ** (-0.578))

def operator_bcr(f1, f2, f3, dv, tdv, m_dry):
    n      = max(1, int(round(f3)))
    m_prop = m_dry * (math.exp(dv / VE_XE) - 1.0)
    m_wet  = m_dry + m_prop
    rec    = max(0.0, tdv - f2)
    costs  = (f3 * m_wet * MARKET_RATE
              + (f1 / dv) * m_prop * XE_COST
              + f3 * sscm_unit_cost(m_dry, n))
    return OPERATOR_FEE * rec / costs if costs > 0 else np.nan

# ---------------------------------------------------------------------------
# Load all Pareto solutions — group by f3
# ---------------------------------------------------------------------------
coverage_by_f3    = defaultdict(list)
client_bcr_by_f3  = defaultdict(list)
op_bcr_raw_by_f3  = defaultdict(list)   # f3 -> list of (f1, f2, tdv, dv)

with h5py.File(args.h5, "r") as f:
    for b in BUDGETS:
        path = f"mdls/{args.scenario}_{b}"
        if path not in f:
            continue
        grp = f[path]
        for tk in sorted(grp.keys()):
            ds   = grp[tk]
            data = ds[:]
            if data.shape[0] == 3 and data.shape[1] != 3:
                data = data.T
            tdv  = float(ds.attrs.get("total_demand_value", 0.0))
            if tdv <= 0:
                continue
            data = data[data[:, 0] < 1e6]
            for row in data:
                f1, f2, f3 = row
                recovered  = max(0.0, tdv - f2)
                cov        = recovered / tdv * 100.0
                oos_fee    = OPERATOR_FEE * recovered
                remaining  = max(0.0, f2)
                denom      = oos_fee + remaining
                client_bcr = recovered / denom if denom > 0 else np.nan

                key = int(round(f3))
                coverage_by_f3[key].append(cov)
                if not np.isnan(client_bcr):
                    client_bcr_by_f3[key].append(client_bcr)
                op_bcr_raw_by_f3[key].append((f1, f2, tdv, float(b)))

f3_vals = sorted(set(coverage_by_f3.keys()) & set(client_bcr_by_f3.keys()))

# ---------------------------------------------------------------------------
# Compute operator BCR band by sweeping M_DRY_RANGE for each f3
# ---------------------------------------------------------------------------
op_bcr_by_f3 = {}
for f3 in f3_vals:
    all_bcrs = []
    for (f1, f2, tdv, dv) in op_bcr_raw_by_f3[f3]:
        b = operator_bcr(f1, f2, f3, dv, tdv, M_DRY)
        if not np.isnan(b):
            all_bcrs.append(b)
    op_bcr_by_f3[f3] = all_bcrs if all_bcrs else [0.0]

# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------
def band_stats(by_f3):
    mean = np.array([np.mean(by_f3[f3])           for f3 in f3_vals])
    lo   = np.array([np.min(by_f3[f3])            for f3 in f3_vals])
    hi   = np.array([np.max(by_f3[f3])            for f3 in f3_vals])
    p25  = np.array([np.percentile(by_f3[f3], 25) for f3 in f3_vals])
    p75  = np.array([np.percentile(by_f3[f3], 75) for f3 in f3_vals])
    return mean, lo, hi, p25, p75

cov_mean, cov_lo, cov_hi, cov_p25, cov_p75 = band_stats(coverage_by_f3)
bcr_mean, bcr_lo, bcr_hi, bcr_p25, bcr_p75 = band_stats(client_bcr_by_f3)
op_mean = np.array([np.mean(op_bcr_by_f3[f3]) for f3 in f3_vals])
op_lo   = np.array([np.min(op_bcr_by_f3[f3])  for f3 in f3_vals])
op_hi   = np.array([np.max(op_bcr_by_f3[f3])  for f3 in f3_vals])

print(f"Fleet sizes: {min(f3_vals)} – {max(f3_vals)}")
print(f"Total solutions: {sum(len(v) for v in coverage_by_f3.values())}")

# ---------------------------------------------------------------------------
# Plot: three panels
# ---------------------------------------------------------------------------
FS     = 20
f3_arr = np.array(f3_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: coverage vs fleet size ---
ax1.fill_between(f3_arr, cov_lo,  cov_hi,  alpha=0.15, color="#1f77b4", label="Min–Max")
ax1.fill_between(f3_arr, cov_p25, cov_p75, alpha=0.35, color="#1f77b4", label="IQR")
ax1.plot(f3_arr, cov_mean, color="#1f77b4", linewidth=2.5, label="Mean")
ax1.set_xlabel("Fleet size (number of servicers)", fontsize=FS)
ax1.set_ylabel("Demand coverage [%]", fontsize=FS)
ax1.tick_params(labelsize=FS - 2)
ax1.set_xlim(f3_arr[0], f3_arr[-1])
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=FS - 2)

# --- Right: client BCR and operator BCR vs fleet size ---
ax2.fill_between(f3_arr, bcr_lo,  bcr_hi,  alpha=0.15, color="#2ca02c")
ax2.fill_between(f3_arr, bcr_p25, bcr_p75, alpha=0.35, color="#2ca02c")
ax2.plot(f3_arr, bcr_mean, color="#2ca02c", linewidth=2.5, label="Client BCR")

ax2.fill_between(f3_arr, op_lo, op_hi, alpha=0.15, color="#d62728")
ax2.plot(f3_arr, op_mean, color="#d62728", linewidth=2.5, label="Operator BCR (150 kg)")

ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Fleet size (number of servicers)", fontsize=FS)
ax2.set_ylabel("BCR", fontsize=FS)
ax2.tick_params(labelsize=FS - 2)
ax2.set_xlim(f3_arr[0], f3_arr[-1])
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=FS - 2)

fig.tight_layout()
outpath = os.path.join(args.out_dir, "coverage_vs_fleet.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
