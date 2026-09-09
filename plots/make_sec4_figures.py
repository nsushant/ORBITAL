#!/usr/bin/env python3
"""Section 4 figures, drawn from the fronts in outputs/exp_results, plus the
tuning figure drawn from the irace logs.

The three front figures:

  sec4_hypervolume_knee.pdf   the paired comparison, two measures x three
                              scenarios. Each panel draws both distributions and
                              the fifteen trials as connecting lines, because
                              the trials are paired -- both algorithms run on the
                              identical demand realisation -- and a plain pair of
                              box plots hides exactly the thing the Wilcoxon
                              signed-rank test is testing. Fifteen lines all
                              sloping the same way is what "15 of 15" looks like.

  sec4_pareto.pdf             one representative trial per scenario, as two
                              projections of the three objectives. Pooling
                              trials would put dollars from different demand
                              realisations on one axis, so a single trial is
                              shown and named in the caption.

  sec4_fleet_coverage.pdf     unrecovered value against fleet size, the
                              trade-off the fleet operator acts on and the one
                              place the two methods visibly differ.

  sec4_tuning_parallel.pdf    irace's tuning in parallel coordinates, both
                              algorithms: every sampled configuration as thin
                              grey lines and the final elites (5 for MDLS,
                              3 for NSGA-II) in bold, so a parameter whose
                              elite lines converge is identified by the tuning
                              and one whose lines fan across the range is not.
                              Drawn only with --irace.

Colours are Okabe-Ito blue and vermillion, which separate under protanopia and
deuteranopia (worst adjacent pair Delta E 21.9) as well as in normal vision
(31.2). Marker shape repeats the distinction so the figures survive greyscale
printing.

Run from basic_project/:  python plots/make_sec4_figures.py [--irace]

The --irace flag draws sec4_tuning_parallel.pdf from the two CSVs written by
plots/irace_export.R and does not touch the fronts in outputs/exp_results.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

COLS = ["f1_dv", "f2_unrecovered_value", "f3_vehicles"]
SCEN = [("S1_repair", "S1 reactive repair"),
        ("S2_refuel", "S2 planned refuelling"),
        ("S3_deorbit", "S3 end-of-life deorbit")]
ALGO = [("mdls", "MDLS", "#0072B2", "o"),
        ("nsga2", "NSGA-II", "#D55E00", "^")]

INK, MUTED = "#1a1a1a", "#5c5c5c"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def tidy(ax):
    """Recessive frame: the data carries the figure, not the furniture."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)


def paired_panel(ax, a, b, ylabel, lower_is_better):
    """Two distributions and the pairing between them."""
    xs = [0.0, 1.0]
    for i in range(len(a)):                      # the pairing, drawn first
        ax.plot(xs, [a[i], b[i]], color=MUTED, linewidth=0.4, alpha=0.35,
                zorder=1, solid_capstyle="round")
    for x, v, (_k, _lab, colour, marker) in zip(xs, (a, b), ALGO):
        bp = ax.boxplot([v], positions=[x], widths=0.42, showfliers=False,
                        patch_artist=True, zorder=2, medianprops=dict(
                            color=INK, linewidth=1.4))
        for box in bp["boxes"]:
            box.set(facecolor=colour, alpha=0.20, edgecolor=colour,
                    linewidth=1.0)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=colour, linewidth=0.8)
        ax.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(
            -0.10, 0.10, len(v)), v, s=9, color=colour, marker=marker,
            linewidths=0, alpha=0.85, zorder=3)
    wins = int(np.sum(a < b) if lower_is_better else np.sum(a > b))
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _k, lab, _c, _m in ALGO])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, 1.55)
    tidy(ax)
    return wins


def load(result_dir, algo, scen, trials):
    out = []
    for t in range(1, trials + 1):
        p = f"{result_dir}/{algo}_{scen}_{t:02d}.csv"
        if os.path.exists(p):
            out.append(pd.read_csv(p)[COLS].values)
    return out


def figure_measures(df, outdir):
    from analyse_fronts import hypervolume                       # noqa: F401
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
    for col, (key, title) in enumerate(SCEN):
        for row, (measure, ylab, lower) in enumerate(
                [("hypervolume", "Hypervolume", False),
                 ("knee", "Knee distance", True)]):
            d = df[df.scenario == key]
            a = d[d.algo == "mdls"].sort_values("trial")[measure].values
            b = d[d.algo == "nsga2"].sort_values("trial")[measure].values
            ax = axes[row, col]
            wins = paired_panel(ax, a, b, ylab if col == 0 else "", lower)
            if row == 0:
                ax.set_title(title, pad=6)
            ax.annotate(f"{wins}/{len(a)}", xy=(0.5, 0.02),
                        xycoords="axes fraction", ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
            if col:
                ax.set_ylabel("")
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none",
                          markersize=5, label=lab)
               for _k, lab, c, m in ALGO]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    p = os.path.join(outdir, "sec4_hypervolume_knee.pdf")
    fig.savefig(p)
    plt.close(fig)
    return p


def figure_fronts(result_dir, outdir, trial):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for col, (key, title) in enumerate(SCEN):
        ax = axes[col]
        for algo, lab, colour, marker in ALGO:
            p = f"{result_dir}/{algo}_{key}_{trial:02d}.csv"
            if not os.path.exists(p):
                continue
            F = pd.read_csv(p)[COLS].values
            ax.scatter(F[:, 0], F[:, 1] / 1e6, s=14, facecolor="none",
                       edgecolor=colour, marker=marker, linewidths=0.9,
                       label=lab, alpha=0.9)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"cumulative $\Delta V$  [m s$^{-1}$]")
        if col == 0:
            ax.set_ylabel("unrecovered value  [\\$M]")
        tidy(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    p = os.path.join(outdir, "sec4_pareto.pdf")
    fig.savefig(p)
    plt.close(fig)
    return p


def figure_fleet(result_dir, outdir, trial):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for col, (key, title) in enumerate(SCEN):
        ax = axes[col]
        for algo, lab, colour, marker in ALGO:
            p = f"{result_dir}/{algo}_{key}_{trial:02d}.csv"
            if not os.path.exists(p):
                continue
            F = pd.read_csv(p)[COLS].values
            ax.scatter(F[:, 2], F[:, 1] / 1e6, s=14, facecolor="none",
                       edgecolor=colour, marker=marker, linewidths=0.9,
                       label=lab, alpha=0.9)
        ax.set_title(title, pad=6)
        ax.set_xlabel("active vehicles")
        if col == 0:
            ax.set_ylabel("unrecovered value  [\\$M]")
        tidy(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    p = os.path.join(outdir, "sec4_fleet_coverage.pdf")
    fig.savefig(p)
    plt.close(fig)
    return p


def param_ranges(paramfile):
    """Ranges from a tune/parameters-*.txt file: (name, "r", lo, hi) for
    numeric parameters and (name, "c") for categorical ones. The line the
    file marks conditional (its value side carries a '|') is skipped, which
    drops swap_top_n: the exchange operator's top-N only exists when the
    operator is on, and no irace elite turned it on."""
    params = []
    for line in open(paramfile, "r"):
        m = re.match(r"^\s*(\w+)\s+(.*?)\s+([ric])\s+(.*)$", line.strip())
        if not m or "(" not in m.group(4):
            continue
        tail = m.group(4)
        if "|" in tail:
            continue
        if m.group(3) in ("r", "i"):
            lo, hi = (float(x)
                      for x in re.findall(r"[-+]?\d*\.?\d+", tail)[:2])
            params.append((m.group(1), "r", lo, hi))
        else:
            params.append((m.group(1), "c"))
    return params


def _irace_panel(ax, cfg, params, colour):
    """One algorithm's race in parallel coordinates. Every configuration is a
    thin grey polyline; the final elites are the bold lines in `colour`."""
    norm, intr = [], []
    for p in params:
        name = p[0]
        if p[1] == "r":
            lo, hi = p[2], p[3]
            v = (cfg[name].astype(float) - lo) / (hi - lo)
        else:
            s = cfg[name].fillna("").astype(str)
            codes = {x: i for i, x in
                     enumerate(sorted(s.unique(), key=str))}
            nk = len(codes)
            v = s.map(codes)
            v = v / (nk - 1) if nk > 1 else v * 0
        norm.append(np.asarray(v, dtype=float))
        intr.append(name)
    N = np.clip(np.column_stack(norm), 0.0, 1.0)
    xs = np.arange(len(intr))

    elite = cfg["final_elite"].fillna(False).to_numpy().astype(bool)

    def segs(vals):
        return [tuple(zip(xs[i:i + 2], vals[i:i + 2]))
                for i in range(len(xs) - 1)]

    grey = [s for v in N[~elite] for s in segs(v)]
    if len(grey):
        ax.add_collection(LineCollection(grey, colors=(0.6, 0.6, 0.6, 0.30),
                                         linewidths=0.3, zorder=1))
    for v in N[elite]:
        ax.add_collection(LineCollection(segs(v), colors=colour,
                                         linewidths=1.5, alpha=0.95, zorder=3))
        ax.scatter(xs, v, s=15, color=colour, linewidths=0, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(intr, rotation=20, ha="right", fontsize=7.5)
    ax.set_xlim(-0.6, len(intr) - 0.4)
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.annotate(f"{len(cfg)} sampled · {int(elite.sum())} elites",
                xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, -6), textcoords="offset points",
                ha="left", va="top", fontsize=7, color=MUTED)


def figure_irace(mdls_csv, nsga2_csv, mdls_params, nsga2_params, outdir):
    specs = [(mdls_csv, mdls_params, "#0072B2", "MDLS"),
             (nsga2_csv, nsga2_params, "#D55E00", "NSGA-II")]
    for csv, _p, _c, _t in specs:
        if not os.path.exists(csv):
            sys.exit(f"{csv} not found -- run plots/irace_export.R first")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, (csv, pfile, colour, title) in zip(axes, specs):
        _irace_panel(ax, pd.read_csv(csv), param_ranges(pfile), colour)
        ax.set_title(title, pad=8)
    handles = [plt.Line2D([], [], color=(0.6, 0.6, 0.6), linewidth=0.8,
                          label="configurations sampled"),
               plt.Line2D([], [], color=INK, linewidth=1.6,
                          label="final elites")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "sec4_tuning_parallel.pdf")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default=os.path.join(ROOT, "outputs/exp_results"))
    ap.add_argument("--metrics", default=os.path.join(ROOT, "outputs/section4_metrics.csv"))
    # The paper's figures live beside MAIN_DOCUMENT_V0.tex, one level up from
    # basic_project, because that is where \includegraphics{figures/...}
    # resolves from. Writing them into basic_project/figures would leave LaTeX
    # looking at an empty directory and reporting a missing file.
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(ROOT), "figures"))
    ap.add_argument("--trial", type=int, default=1,
                    help="which trial the front figures show; each trial is a "
                         "different demand realisation, so one is shown rather "
                         "than several pooled onto one dollar axis")
    ap.add_argument("--irace", action="store_true",
                    help="draw sec4_tuning_parallel.pdf from the CSVs written "
                         "by plots/irace_export.R and do nothing else")
    ap.add_argument("--irace-mdls-csv",
                    default=os.path.join(ROOT, "outputs/irace-mdls-configurations.csv"))
    ap.add_argument("--irace-nsga2-csv",
                    default=os.path.join(ROOT, "outputs/irace-nsga2-configurations.csv"))
    ap.add_argument("--irace-params-mdls",
                    default=os.path.join(ROOT, "tune/parameters-mdls.txt"))
    ap.add_argument("--irace-params-nsga2",
                    default=os.path.join(ROOT, "tune/parameters-nsga2.txt"))
    args = ap.parse_args()

    if args.irace:
        made = [figure_irace(args.irace_mdls_csv, args.irace_nsga2_csv,
                             args.irace_params_mdls, args.irace_params_nsga2,
                             args.outdir)]
        for p in made:
            print(f"  wrote {os.path.relpath(p, ROOT)}")
        return

    if not os.path.exists(args.metrics):
        sys.exit(f"{args.metrics} not found -- run analyse_fronts.py first")
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.metrics)

    made = [figure_measures(df, args.outdir),
            figure_fronts(args.result_dir, args.outdir, args.trial),
            figure_fleet(args.result_dir, args.outdir, args.trial)]
    for p in made:
        print(f"  wrote {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
