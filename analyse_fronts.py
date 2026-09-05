"""Section 4.1: hypervolume, knee distance, and what each method actually found.

Three measures per (scenario, algorithm, trial):

  hypervolume    coverage of the objective space, normalised so the three
                 objectives -- m/s, a vehicle count and dollars -- are
                 comparable. Without normalisation f3 is six orders of
                 magnitude larger than f2 and the measure is really just f3.

  knee distance  L2 distance from the normalised ideal to the nearest front
                 point: the best available compromise.

  fraction new   the share of a front that is not already matched or beaten by
                 one of the common starting schedules. Both methods start from
                 the same 18 greedy restarts (D26), so a front can look large
                 while consisting mostly of its own starting set. This
                 separates what a method found from what it was given, and no
                 standard metric does that.

Every trial is a *different problem instance*: `generate_demands.py` draws a
fresh demand realisation per trial, so trial 7's objectives live on a different
scale from trial 1's. Two consequences run through this script.

  Normalisation is per (scenario, trial), over the union of both algorithms'
  fronts on that trial. Pooling trials would fold instance difficulty into the
  scale and make an easy instance look like a good front. The hypervolume
  reference point is the normalised nadir, (1, 1, 1).

  The seed set is rebuilt from *that trial's* demands. Scoring a trial-7 front
  against trial-1's starting schedules compares numbers from two unrelated
  instances; it was doing exactly that, and reported 26 of 41 points as
  "given" when the true count was 0 (F33).

Because both algorithms run on the identical instance, the trials are paired,
so the comparison is a Wilcoxon *signed-rank* test rather than rank-sum. That
is both the correct test and a considerably more sensitive one: at 15 pairs the
smallest attainable two-sided p is 6.1e-5, against 0.0079 for an unpaired
5-v-5.

Run:  python analyse_fronts.py [--trials 15] [--out outputs/section4_metrics.csv]
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

COLS = ["f1_dv", "f2_unrecovered_value", "f3_vehicles"]

# Objectives travel through CSV, so a value can come back changed in its last
# bit. Comparisons here are therefore tolerant: a seed that reappears in the
# front as 3695.118160000001 against 3695.11816 is the same solution, not a
# solution that beats it. Untolerated, that single ulp was reported as an
# archive-invariant violation (F33).
RTOL, ATOL = 1e-9, 1e-6


def _tol(a, b):
    return RTOL * np.maximum(np.abs(a), np.abs(b)) + ATOL


def weakly_dominates(a, b):
    """`a` is at least as good as `b` on every objective, within tolerance."""
    return bool(np.all(a <= b + _tol(a, b)))


def strictly_dominates(a, b):
    """`a` is at least as good everywhere and better somewhere, beyond noise."""
    t = _tol(a, b)
    return bool(np.all(a <= b + t) and np.any(a < b - t))


# ---------------------------------------------------------------------------
# Hypervolume. moocore when it is installed (what the paper cites), otherwise
# an exact sweep, which is cheap at these front sizes and lets the script run
# anywhere. When both are available they are cross-checked against each other.
# ---------------------------------------------------------------------------


def _hv2(P, r):
    """Exact 2-D hypervolume, minimisation, reference r."""
    P = P[P[:, 0].argsort()]
    area, ybest = 0.0, r[1]
    for i in range(len(P)):
        x_next = P[i + 1, 0] if i + 1 < len(P) else r[0]
        ybest = min(ybest, P[i, 1])
        area += max(0.0, x_next - P[i, 0]) * max(0.0, r[1] - ybest)
    return area


def _hv3(P, r):
    """Exact 3-D hypervolume by slab decomposition along the third objective."""
    P = np.asarray(P, float)
    P = P[np.all(P < np.asarray(r), axis=1)]         # points beyond r add nothing
    if len(P) == 0:
        return 0.0
    P = P[P[:, 2].argsort()]
    vol = 0.0
    for k in range(len(P)):
        z_next = P[k + 1, 2] if k + 1 < len(P) else r[2]
        vol += _hv2(P[:k + 1, :2].copy(), r[:2]) * max(0.0, z_next - P[k, 2])
    return vol


def hypervolume(P, r=(1.0, 1.0, 1.0)):
    mine = _hv3(P, r)
    try:
        import moocore
    except ImportError:
        return mine
    theirs = float(moocore.hypervolume(np.asarray(P, float), ref=list(r)))
    if abs(theirs - mine) > 1e-6 * max(1.0, abs(theirs)):
        print(f"  ! hypervolume disagreement: moocore {theirs:.6f} vs "
              f"internal {mine:.6f} -- investigate before reporting either",
              file=sys.stderr)
    return theirs


# ---------------------------------------------------------------------------
# The starting set, per instance.
# ---------------------------------------------------------------------------


def seed_front(ct, dem, depot, max_vehicles, refuel_time, cache=None):
    """Objectives of the 18 greedy restarts both methods start from, in CSV
    column order [dv, unrecovered, vehicles].

    Cached to disk: this is deterministic given the instance (verified across
    separate processes), and rebuilding 45 of them on every run costs minutes
    for nothing.
    """
    if cache and os.path.exists(cache):
        return np.load(cache)
    from oos.greedy import greedy_restarts
    from oos.schedule import evaluate
    seeds = greedy_restarts(ct, dem, depot, max_vehicles, refuel_time,
                            dv_budget=ct.dv_budget)
    S = np.array([evaluate(s, dem, ct.dv_budget)[0] for s in seeds])
    S = S[:, [0, 2, 1]]                  # -> CSV order [dv, unrecovered, vehicles]
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, S)
    return S


def seed_dominated(F, S):
    """Front points strictly beaten by a starting schedule.

    For MDLS this must be zero, and the reason is a proof rather than a
    convention. Every seed is offered to the archive before the search begins.
    If a seed s beats a final point f, then when f was offered either s was
    still held -- and f was rejected -- or s had been evicted by some p beating
    s and therefore beating f, and the argument repeats. Domination is
    transitive and the archive is unbounded, so something always beats f and f
    can never enter.

    A non-zero count is thus not a bad search; it is a broken pipeline. It
    caught the superseded Sec. 4 fronts (F32), and it caught this script
    scoring every trial against trial 1's seed set (F33).

    No equivalent invariant exists for NSGA-II -- its final population is not
    an archive and may legitimately carry dominated individuals -- so the check
    runs on the MDLS fronts only. The two are produced in the same pass, so
    that is enough to catch a bad set.
    """
    return sum(1 for f in F if any(strictly_dominates(s, f) for s in S))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="outputs/exp_results")
    p.add_argument("--cost-table", default="outputs/cost_table.h5")
    p.add_argument("--demand-dir", default="outputs/exp_demands")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--max-vehicles", type=int, default=25)
    p.add_argument("--refuel-time", type=float, default=0.5)
    p.add_argument("--out", default="outputs/section4_metrics.csv")
    p.add_argument("--cache-dir", default="outputs/.seed_cache")
    p.add_argument("--allow-stale", action="store_true",
                   help="report fronts the invariant check flags as not "
                        "reproducible; never appropriate for the paper")
    args = p.parse_args()

    from oos.demands import load_demands
    from oos.schedule import load_cost_table

    ct = load_cost_table(args.cost_table)
    depot = ct.names.index("depot_1")

    scenarios = sorted({os.path.basename(f).split("_", 1)[1].rsplit("_", 1)[0]
                        for f in glob.glob(f"{args.result_dir}/*_S*_*.csv")})
    algos = ["mdls", "nsga2"]
    rows, suspect = [], []

    def path(algo, scen, t):
        return f"{args.result_dir}/{algo}_{scen}_{t:02d}.csv"

    for scen in scenarios:
        for t in range(1, args.trials + 1):
            fronts = {a: pd.read_csv(path(a, scen, t))[COLS].values
                      for a in algos if os.path.exists(path(a, scen, t))}
            fronts = {a: F for a, F in fronts.items() if len(F)}
            if not fronts:
                continue

            # One scale per instance, over both algorithms' fronts on it.
            U = np.vstack(list(fronts.values()))
            ideal, nadir = U.min(axis=0), U.max(axis=0)
            span = np.maximum(nadir - ideal, 1e-12)

            dem = load_demands(f"{args.demand_dir}/{scen}_{t:02d}.h5")
            S = seed_front(ct, dem, depot, args.max_vehicles, args.refuel_time,
                           cache=f"{args.cache_dir}/{scen}_{t:02d}.npy")

            for algo, F in fronts.items():
                if algo == "mdls":
                    nsd = seed_dominated(F, S)
                    if nsd:
                        suspect.append((os.path.basename(path(algo, scen, t)),
                                        nsd, len(F)))
                N = (F - ideal) / span
                given = sum(1 for f in F
                            if any(weakly_dominates(s, f) for s in S))
                rows.append(dict(
                    scenario=scen, algo=algo, trial=t, points=len(F),
                    hypervolume=hypervolume(N),
                    knee=float(np.linalg.norm(N, axis=1).min()),
                    frac_new=1.0 - given / len(F),
                    best_dv=F[:, 0].min(), min_fleet=F[:, 2].min(),
                    best_unrecovered_M=F[:, 1].min() / 1e6))

    if not rows:
        sys.exit(f"no result CSVs found in {args.result_dir}")
    if suspect and not args.allow_stale:
        sys.exit(
            "these MDLS fronts contain points beaten by a starting schedule,\n"
            "which a non-dominated archive seeded from those schedules cannot\n"
            "produce, so the pipeline is inconsistent somewhere:\n  "
            + "\n  ".join(f"{n}: {d} of {m} points" for n, d, m in suspect[:8])
            + ("\n  ..." if len(suspect) > 8 else "")
            + "\n\nRe-run (bash run_experiments.sh) before reporting anything.\n"
              "--allow-stale overrides this, and should not be used for the paper.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    found = df.groupby(["scenario", "algo"]).size()
    if found.min() < args.trials:
        print(f"! asked for {args.trials} trials; found {found.min()}-"
              f"{found.max()} per cell. --trials only says how many files to\n"
              f"  look for -- it does not run anything. The medians below are\n"
              f"  over what exists.\n")

    print("median over trials (IQR in brackets)\n")
    hdr = (f"{'scenario':12s} {'algo':6s} {'hypervolume':>22s} {'knee':>20s} "
           f"{'% new':>7s} {'pts':>5s} {'best dV':>8s} {'fleet':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for scen in scenarios:
        for algo in algos:
            d = df[(df.scenario == scen) & (df.algo == algo)]
            if not len(d):
                continue

            def med(c, d=d):
                q1, q3 = d[c].quantile([.25, .75])
                return f"{d[c].median():.3f} [{q1:.3f},{q3:.3f}]"

            print(f"{scen:12s} {algo:6s} {med('hypervolume'):>22s} "
                  f"{med('knee'):>20s} {100*d.frac_new.median():6.0f}% "
                  f"{d.points.median():5.0f} {d.best_dv.median():8.0f} "
                  f"{d.min_fleet.median():6.0f}")
        print()

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("scipy not available; skipping the tests")
        return

    print("Wilcoxon signed-rank on trials paired by instance, MDLS against")
    print("NSGA-II (higher hypervolume and lower knee distance are better):\n")
    print(f"{'scenario':12s} {'measure':14s} {'pairs':>6s} {'MDLS':>9s} "
          f"{'NSGA-II':>9s} {'wins':>6s} {'p':>9s}  better")
    n_pairs_min = None
    for scen in scenarios:
        a = df[(df.scenario == scen) & (df.algo == "mdls")].set_index("trial")
        b = df[(df.scenario == scen) & (df.algo == "nsga2")].set_index("trial")
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 2:
            continue
        n_pairs_min = len(common) if n_pairs_min is None else min(n_pairs_min,
                                                                  len(common))
        for measure, better_is in (("hypervolume", "high"), ("knee", "low")):
            x, y = a.loc[common, measure].values, b.loc[common, measure].values
            d = x - y
            if np.allclose(d, 0):
                print(f"{scen:12s} {measure:14s} {len(common):6d} "
                      f"{'identical':>9s} {'':>9s} {'':>6s} {'':>9s}  tie")
                continue
            stat, pv = wilcoxon(x, y)
            wins = int((d > 0).sum() if better_is == "high" else (d < 0).sum())
            am, bm = np.median(x), np.median(y)
            win = "MDLS" if ((am > bm) == (better_is == "high")) else "NSGA-II"
            print(f"{scen:12s} {measure:14s} {len(common):6d} {am:9.3f} "
                  f"{bm:9.3f} {wins:3d}/{len(common):<2d} {pv:9.5f}  {win}")

    if n_pairs_min:
        print(f"\nAt {n_pairs_min} pairs the smallest attainable two-sided p is "
              f"{2 / 2**n_pairs_min:.2g}. Pairing is legitimate here because "
              f"both\nmethods run on the identical demand realisation in each "
              f"trial.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
