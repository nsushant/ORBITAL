"""
plot_bcr_vs_fee.py — Operator and client BCR vs contract fee F.

For the cheapest mutually viable equal-deal architecture (min F* with
BCR* >= 1), plot:

    operator BCR = F / C
    client BCR   = R / (F + U)

They meet at F*. Optional --f3 restricts the pick to that fleet size.

Run: python plot_bcr_vs_fee.py --h5 outputs/long_horizon_results_edelbaum.h5
"""

import os, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bcr_model import (
    unpack_front, min_viable_contract, client_label, DV_BUDGET, M_DRY,
)

parser = argparse.ArgumentParser()
parser.add_argument("--h5",       default="outputs/long_horizon_results_edelbaum.h5")
parser.add_argument("--out-dir",  default="outputs")
parser.add_argument("--scenario", default="bcr_mixed")
parser.add_argument("--f3",       type=int, default=None,
                    help="Restrict the equal-deal pick to this fleet size")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

BUDGET = int(DV_BUDGET)
FS = 16
COLORS = {"operator": "#1f77b4", "client": "#d62728"}

fronts = []
with h5py.File(args.h5, "r") as f:
    path = f"mdls/{args.scenario}_{BUDGET}"
    if path not in f:
        raise SystemExit(f"No {path} in {args.h5}")
    for tk in sorted(f[path].keys()):
        front = unpack_front(f[path][tk])
        data = front["data"]
        data = data[data[:, 0] < 1e6]
        if len(data) == 0 or not front["clients"]:
            continue
        fronts.append(front)

if not fronts:
    raise SystemExit("No per-client fronts")

clients = list(fronts[0]["clients"])


def pick_for_client(client):
    """Cheapest viable equal-deal for `client` across loaded trials."""
    best = None
    for front in fronts:
        data = front["data"]
        if args.f3 is not None:
            data = data[np.round(data[:, 2]) == args.f3]
            if len(data) == 0:
                continue
        s = min_viable_contract(front, BUDGET, M_DRY, data=data).get(client)
        if s is None or not s["viable"]:
            continue
        if best is None or s["F"] < best["F"]:
            best = s
    return best


n = len(clients)
fig, axes = plt.subplots(1, n, figsize=(7.2 * n, 5.5), squeeze=False)
axes = axes[0]

for ax, c in zip(axes, clients):
    s = pick_for_client(c)
    if s is None:
        ax.set_title(f"{client_label(c)} — no BCR* ≥ 1")
        ax.set_xlabel("Contract fee F [$M]", fontsize=FS)
        ax.set_ylabel("BCR", fontsize=FS)
        continue
    C, R, U = s["C"], s["recovered"], s["unserved"]
    Fstar, bcr = s["F"], s["bcr"]
    # Zoom around the equal-deal, not the client's distant BCR=1 (R-U can be >> C).
    F_max = 4.0 * max(Fstar, C, 1.0)
    F = np.linspace(0.0, F_max, 400)
    bcr_op = F / C
    bcr_cl = R / np.maximum(F + U, 1e-12)
    y_top = 1.25 * max(bcr, float(bcr_op[-1]), float(bcr_cl[0]), 1.0)

    ax.plot(F / 1e6, bcr_op, color=COLORS["operator"], linewidth=2.5,
            label="Operator BCR  (F / C)")
    ax.plot(F / 1e6, bcr_cl, color=COLORS["client"], linewidth=2.5,
            label=f"{client_label(c)} BCR  (R / (F+U))")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(Fstar / 1e6, color="0.45", linestyle=":", linewidth=1.2)
    ax.plot([Fstar / 1e6], [bcr], marker="o", color="black", markersize=8, zorder=5)
    ax.annotate(
        f"F* = ${Fstar/1e6:.1f} M\nBCR* = {bcr:.2f}",
        xy=(Fstar / 1e6, bcr),
        xytext=(12, 12), textcoords="offset points",
        fontsize=FS - 4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"),
    )
    f3_txt = f", {int(round(s['f3']))} servicers"
    ax.set_title(client_label(c) + f3_txt, fontsize=FS)
    ax.set_xlabel("Contract fee F [$M]", fontsize=FS)
    ax.set_ylabel("BCR", fontsize=FS)
    ax.tick_params(labelsize=FS - 2)
    ax.set_xlim(0, F_max / 1e6)
    ax.set_ylim(0, y_top)
    ax.legend(fontsize=FS - 4, loc="best")
    print(f"{c}: F*=${Fstar/1e6:.2f}M  BCR*={bcr:.3f}  "
          f"C=${C/1e6:.2f}M  R=${R/1e6:.2f}M  U=${U/1e6:.2f}M  "
          f"f3={s['f3']:.0f}  f1={s['f1']:.0f}")

fig.tight_layout()
outpath = os.path.join(args.out_dir, "bcr_vs_fee.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
