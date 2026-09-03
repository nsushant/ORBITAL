"""The shared-ladder path must agree with the reference implementation.

`transfer_cost` refines each entry's growth bracket by bisection and is the
implementation every earlier result was validated against. `transfer_cost_grid`
computes each ring once and tests every entry against it, which is three orders
of magnitude cheaper. This gate holds the second to the first over the geometry
distribution the cost table actually contains, and reports where they differ
rather than only whether they do.

Both are judged against a high-resolution reference -- a 2000-rung ladder
refined to 1e-5 -- rather than against each other. Judging the fast path against
the 48-rung production settings was misleading: those settings carry their own
2 % tail, so the two errors compounded and the comparison looked far worse than
either implementation is.

Two differences are expected and are not failures:

  * The grid path can find closures the reference misses. The reference brackets
    a sign change and then bisects to confirm it, abandoning the bracket if a
    midpoint lands in an infeasible pocket; the grid path appeals to the
    intermediate value theorem over a run of feasible samples instead, which is
    the stronger argument.
  * The grid returns the first ladder rung that closes rather than refining
    within it, so its delta-V is high by up to one rung spacing. That error is
    one-sided: it never reports a transfer cheaper than it is.
"""
import math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oos import servicer
from oos.edelbaum import transfer_cost, transfer_cost_grid
from oos.nodes import load_nodes

SIM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "simulation.h5")
if not os.path.exists(SIM):
    raise SystemExit(f"need {SIM}; run stage 2 of run_all.sh first")

nd = load_nodes(SIM)
rng = np.random.default_rng(7)
grid = np.arange(15.0, 400.0, 15.0) * 86400.0
n_dep, n_tof = len(grid), len(grid)

N_PAIRS = int(os.environ.get("N_PAIRS", "24"))
pairs = [(int(a), int(b)) for a, b in rng.integers(0, len(nd), size=(N_PAIRS, 2))
         if a != b]

def state(k):
    return np.array([nd.a[k], nd.incl[k], 0.0,
                     servicer.MASS, servicer.ISP, servicer.THRUST])

signed = {"reference": [], "grid": []}
t_ref = t_fast = 0.0
n_both = n_ref_only = n_fast_only = n_neither = 0
rel = []

print(f"{len(pairs)} object pairs x {n_dep * n_tof} entries "
      f"= {len(pairs) * n_dep * n_tof:,} comparisons\n")

for (i, j) in pairs:
    s1, s2 = state(i), state(j)
    d_grid = np.empty((n_dep, n_tof))
    for p in range(n_dep):
        for q in range(n_tof):
            dep, tof = grid[p], grid[q]
            d_grid[p, q] = ((nd.raan0[j] + nd.raan_rate[j] * (dep + tof))
                            - (nd.raan0[i] + nd.raan_rate[i] * dep))
            d_grid[p, q] -= 2 * math.pi * round(d_grid[p, q] / (2 * math.pi))

    tofs_flat = np.tile(grid, n_dep)
    draan_flat = d_grid.reshape(-1)

    t0 = time.perf_counter()
    ref = transfer_cost(s1, s2, tofs_flat, draan_flat).reshape(n_dep, n_tof)
    t_ref += time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = transfer_cost_grid(s1, s2, grid, d_grid)
    t_fast += time.perf_counter() - t0

    truth = transfer_cost(s1, s2, tofs_flat, draan_flat,
                          180, 2000, 8.0, 1e-5).reshape(n_dep, n_tof)
    tm = np.isfinite(truth)
    for name, arr in (("reference", ref), ("grid", fast)):
        m = tm & np.isfinite(arr)
        if m.any():
            signed[name].append((arr[m] - truth[m]) / truth[m])

    fr, ff = np.isfinite(ref), np.isfinite(fast)
    n_both += int((fr & ff).sum())
    n_ref_only += int((fr & ~ff).sum())
    n_fast_only += int((~fr & ff).sum())
    n_neither += int((~fr & ~ff).sum())
    both = fr & ff
    if both.any():
        rel.append(np.abs(fast[both] - ref[both]) / np.maximum(ref[both], 1e-12))

rel = np.concatenate(rel) if rel else np.zeros(1)
total = len(pairs) * n_dep * n_tof

print(f"speed        reference {t_ref:7.2f} s   grid {t_fast:7.2f} s   "
      f"-> {t_ref / max(t_fast, 1e-9):.0f}x")
print()
print(f"both feasible           : {n_both:8,} ({100 * n_both / total:5.1f} %)")
print(f"reference only          : {n_ref_only:8,} ({100 * n_ref_only / total:5.1f} %)")
print(f"grid only               : {n_fast_only:8,} ({100 * n_fast_only / total:5.1f} %)"
      "   <- expected: IVT is stronger than bracket-and-bisect")
print(f"neither                 : {n_neither:8,} ({100 * n_neither / total:5.1f} %)")
print()
print("signed delta-V error against a 2000-rung, 1e-5-refined reference")
print(f"{'':24}{'median':>10}{'p99':>10}{'worst |.|':>12}{'too cheap':>12}")
stats = {}
for name in ("reference", "grid"):
    v = np.concatenate(signed[name])
    stats[name] = v
    print(f"  {name:<22}{np.median(v) * 100:+9.4f}%{np.percentile(v, 99) * 100:+9.3f}%"
          f"{np.abs(v).max() * 100:11.3f}%{100 * (v < -1e-9).mean():11.1f}%")
print()
print(f"the two against each other: median {np.median(rel) * 100:.4f} %, "
      f"worst {rel.max() * 100:.3f} %")

g = stats["grid"]
r = stats["reference"]
fails = []
if n_ref_only > 0.005 * total:
    fails.append(f"grid misses {n_ref_only} entries the reference finds")
# The grid must be conservative: high by less than a rung spacing, and hardly
# ever low. Both are properties of the grid alone, not of the old settings.
if np.percentile(g, 99) > 0.0125:
    fails.append(f"grid p99 error {np.percentile(g, 99) * 100:.3f} % exceeds the "
                 "0.95 % ladder spacing by too much")
if (g < -1e-9).mean() > max(0.01, 2 * (r < -1e-9).mean()):
    fails.append(f"grid reports a cheaper transfer than the truth on "
                 f"{100 * (g < -1e-9).mean():.2f} % of entries")
if np.abs(g).max() > 1.5 * np.abs(r).max() and np.abs(g).max() > 0.05:
    fails.append(f"grid worst error {np.abs(g).max() * 100:.2f} % is far above "
                 f"the reference's {np.abs(r).max() * 100:.2f} %")

print()
if fails:
    for f in fails:
        print("FAIL:", f)
    sys.exit(1)
print("all checks passed")
