"""Bi-objective 0/1 knapsack solved with mdls.MDLS; run `python examples/knapsack.py -h`."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numba import njit

from mdls import MDLS, move

COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # one per compared run; grey is reserved
GREY = "#9e9e9e"


def make_instance(n_items, cap_frac, seed):
    """Random weights and two profit rows; capacity is a fraction of total weight."""
    rng = np.random.default_rng(seed)
    w = rng.integers(10, 100, n_items).astype(float)
    p = rng.integers(10, 100, (2, n_items)).astype(float)
    return dict(w=w, p=p, cap=cap_frac * w.sum())


def evaluate(x, inst):
    """f = negated profits (MDLS minimises), g = [load - capacity] (feasible when <= 0)."""
    return -(inst["p"] @ x), [inst["w"] @ x - inst["cap"]]


@njit
def greedy_fill(x, w, cap, order):
    """Add the items of x in `order` while they fit; x is modified in place."""
    load = 0.0
    for i in range(x.size):
        if x[i]:
            load += w[i]
    for t in range(order.size):
        i = order[t]
        if not x[i] and load + w[i] <= cap:
            x[i] = True
            load += w[i]
    return x


def _drop_add(x, inst, rng, k, n_drop, noise):
    """One attempt: drop the weakest held items on profit k, refill greedily.

    Returns a copy that strictly improves profit k, or None; x is left untouched
    so the decorator can retry from the same base.
    """
    p, w = inst["p"][k], inst["w"]
    held = np.flatnonzero(x)
    if held.size == 0:
        return None
    y = x.copy()
    ratio = p[held] / w[held]
    noisy = ratio * rng.uniform(1 - noise, 1 + noise, held.size)  # avoids stalling
    y[held[np.argsort(noisy)[:n_drop]]] = False
    greedy_fill(y, w, inst["cap"], np.argsort(-p / w))
    return y if p @ y > p @ x else None


@move(improves=0, attempts=4)
def drop_add_0(x, inst, rng, n_drop=3, noise=0.2):
    """Move for profit 0: drop the weakest held items, refill greedily.

    The drop is noisy, so the decorator retries up to four times.
    """
    return _drop_add(x, inst, rng, 0, n_drop, noise)


@move(improves=1, attempts=4)
def drop_add_1(x, inst, rng, n_drop=3, noise=0.2):
    """Move for profit 1: drop the weakest held items, refill greedily.

    The drop is noisy, so the decorator retries up to four times.
    """
    return _drop_add(x, inst, rng, 1, n_drop, noise)


@njit
def find_swap(w, p, slack, i_draw, j_draw):
    """First drawn pair that raises profit 0 and still fits; returns (found, i, j)."""
    for t in range(i_draw.size):
        i, j = i_draw[t], j_draw[t]
        if p[j] > p[i] and w[j] - w[i] <= slack:
            return True, i, j
    return False, 0, 0


@move(improves=0, attempts=1)
def swap(x, inst, rng, tries=50):
    """Move for profit 0: first-improvement 1-1 swap over randomly drawn pairs.

    `tries` is the batch size inside the numba kernel, so no decorator retries.
    """
    w, p = inst["w"], inst["p"][0]
    held, free = np.flatnonzero(x), np.flatnonzero(~x)
    if held.size == 0 or free.size == 0:
        return None
    slack = inst["cap"] - w @ x
    i_draw = held[rng.integers(0, held.size, tries)]
    j_draw = free[rng.integers(0, free.size, tries)]
    found, i, j = find_swap(w, p, slack, i_draw, j_draw)
    if not found:
        return None
    x[i], x[j] = False, True
    return x


def build(cfg):
    """Instance, MDLS object and init function from a config dict."""
    inst = make_instance(cfg["n_items"], cfg["capacity_fraction"], cfg["instance_seed"])
    moves = [drop_add_0(n_drop=cfg["n_drop"], noise=cfg["noise"]),
             drop_add_1(n_drop=cfg["n_drop"], noise=cfg["noise"])]
    if cfg["use_swap"]:
        moves.append(swap(tries=cfg["swap_tries"]))
    n = len(inst["w"])

    def init(rng):
        empty = np.zeros(n, dtype=bool)
        starts = [greedy_fill(empty.copy(), inst["w"], inst["cap"],
                              np.argsort(-inst["p"][k] / inst["w"])) for k in range(2)]
        return starts + [greedy_fill(empty.copy(), inst["w"], inst["cap"], rng.permutation(n))
                         for _ in range(cfg["n_random_starts"])]

    search = MDLS(evaluate, moves=moves, instance=inst, verbose=cfg["verbose"])
    return inst, search, init


def cmd_run(cfg):
    """One search; prints the summary and writes the front as CSV."""
    _, search, init = build(cfg)
    res = search.run(init, max_eval=cfg["max_eval"], seed=cfg["seeds"][0])
    print(res.summary())
    f = -res.objectives
    f = f[np.argsort(f[:, 0])]
    np.savetxt(cfg["front_csv"], f, delimiter=",", header="profit_0,profit_1", comments="", fmt="%.0f")
    k = res.archive.knee()
    print(f"knee: {-res.objectives[k, 0]:.0f}, {-res.objectives[k, 1]:.0f}; front written to {cfg['front_csv']}")


def cmd_plot(cfg):
    """Fronts and archive growth for up to 4 seeds, one colour each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = cfg["seeds"][:4]
    _, search, init = build(cfg)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for c, s in zip(COLOURS, seeds):
        res = search.run(init, max_eval=cfg["max_eval"], seed=s, record_every=cfg["record_every"])
        f = -res.objectives
        f = f[np.argsort(f[:, 0])]
        a1.plot(f[:, 0], f[:, 1], color=GREY, lw=0.8, zorder=1)
        a1.scatter(f[:, 0], f[:, 1], color=c, s=14, zorder=2, label=f"seed {s}")
        if res.history:
            h = np.array(res.history)
            a2.plot(h[:, 1], h[:, 2], color=GREY, lw=0.8, zorder=1)
            a2.scatter(h[:, 1], h[:, 2], color=c, s=10, zorder=2, label=f"seed {s}")
    a1.set_xlabel("Profit 1"); a1.set_ylabel("Profit 2")
    a2.set_xlabel("Evaluations"); a2.set_ylabel("Front size")
    a1.legend(); a2.legend()
    fig.tight_layout()
    fig.savefig(cfg["plot_png"], dpi=150)
    print(f"plot written to {cfg['plot_png']}")


def main(argv=None):
    """Parse the command line, load the config, apply overrides, dispatch."""
    ap = argparse.ArgumentParser(description="MDLS on a bi-objective knapsack")
    ap.add_argument("command", choices=["run", "plot"])
    ap.add_argument("--config", default=None,
                    help="parameter file (JSON); defaults to params.json next to this script")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. --set max_eval=2000 seeds=[1,2]")
    args = ap.parse_args(argv)
    config = Path(args.config) if args.config else Path(__file__).with_name("params.json")
    with config.open() as fh:
        cfg = json.load(fh)
    for kv in args.set:
        key, val = kv.split("=", 1)
        if key not in cfg:
            ap.error(f"unknown config key {key!r}")
        try:
            cfg[key] = json.loads(val)
        except json.JSONDecodeError:
            cfg[key] = val  # not a JSON literal: keep as string
    logging.basicConfig(level=logging.INFO if cfg["verbose"] else logging.WARNING,
                        format="%(message)s")
    {"run": cmd_run, "plot": cmd_plot}[args.command](cfg)


if __name__ == "__main__":
    main()
