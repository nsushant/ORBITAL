"""
plot_contract_budget.py — Min F* at the BCR=1 crossover.

Client BCR  = R / (U + F)
Operator BCR = F / (C + C_dv)
Both = 1 ⇒ F* = R - U and C = F* - C_dv. Xenon is negligible, so C ≈ F*
and a C map would duplicate F*. Right panel is coverage of the same point.

Run: python plot_contract_budget.py --h5 outputs/depot_location_sweep.h5
"""

import os, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bcr_model import unpack_front, min_crossover_deal, client_label
from plot_depot_bcr_map import paper_cmap, draw_map, INC_LO, INC_HI, FS


def load_fronts(h5_path, scenario):
    recs = []
    with h5py.File(h5_path, "r") as f:
        if "loc" not in f:
            raise SystemExit(f"No 'loc' group in {h5_path}")
        for name in sorted(f["loc"].keys()):
            g = f["loc"][name]
            a_km = float(g.attrs.get("a_km", np.nan))
            inc_deg = float(g.attrs.get("inc_deg", np.nan))
            path = f"mdls/{scenario}/trial_01"
            if path not in g:
                recs.append({"a_km": a_km, "inc_deg": inc_deg, "front": None})
                continue
            recs.append({
                "a_km": a_km, "inc_deg": inc_deg,
                "front": unpack_front(g[path]),
            })
    return recs


def deal_of(front, client):
    if front is None or client not in (front["clients"] if front else []):
        return None
    s = min_crossover_deal(front).get(client)
    if s is None or not s["viable"]:
        return None
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",       default="outputs/depot_location_sweep.h5")
    parser.add_argument("--out-dir",  default="outputs/depot_location_sweep")
    parser.add_argument("--scenario", default="bcr_mixed_7000")
    parser.add_argument("--client",   default="")
    parser.add_argument("--out",      default="contract_budget.pdf")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f"{args.h5} not found — run sweep_depot_location.jl first")

    recs = load_fronts(args.h5, args.scenario)
    clients = []
    for r in recs:
        if r["front"] is not None and r["front"]["clients"]:
            clients = list(r["front"]["clients"])
            break
    if not clients:
        raise SystemExit("No per-client columns in this H5 — re-run the depot sweep.")
    client = args.client or clients[0]
    if client not in clients:
        raise SystemExit(f"client {client!r} not in {clients}")

    a = np.array([r["a_km"] for r in recs])
    inc = np.array([r["inc_deg"] for r in recs])
    F = np.full(len(recs), np.nan)
    C = np.full(len(recs), np.nan)
    cov = np.full(len(recs), np.nan)
    for i, r in enumerate(recs):
        s = deal_of(r["front"], client)
        if s is None:
            continue
        F[i] = s["F"]
        C[i] = s["C"]
        tot = s["recovered"] + s["unserved"]
        cov[i] = 100.0 * s["recovered"] / tot if tot > 0 else np.nan

    ok = np.isfinite(F) & (inc >= INC_LO) & (inc <= INC_HI)
    print(f"{client}: {ok.sum()}/{len(F)} with BCR=1 crossover")
    if ok.any():
        i_lo = int(np.nanargmin(np.where(ok, F, np.nan)))
        i_hi = int(np.nanargmax(np.where(ok, F, np.nan)))
        print(f"  min F* = ${F[i_lo]/1e6:.1f} M  C = ${C[i_lo]/1e6:.1f} M  "
              f"cov={cov[i_lo]:.1f}%  at a={a[i_lo]:.0f} km, i={inc[i_lo]:.1f}°")
        print(f"  max F* = ${F[i_hi]/1e6:.1f} M  C = ${C[i_hi]/1e6:.1f} M  "
              f"cov={cov[i_hi]:.1f}%  at a={a[i_hi]:.0f} km, i={inc[i_hi]:.1f}°")
        rel = np.abs(F[ok] - C[ok]) / np.maximum(F[ok], 1.0)
        print(f"  median |F*-C|/F* = {100*np.median(rel):.3f}%  (xenon; C map omitted)")

    cmap = paper_cmap()
    lab = client_label(client)
    F_M = F / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(24.0, 10.0), sharey=True,
                             layout="constrained")
    ax_F, ax_cov = axes

    sc_F, _ = draw_map(ax_F, a, inc, F_M, title=f"{lab}:  min F*  (= C + C_dv)",
                       cmap=cmap)
    cbar_F = fig.colorbar(sc_F, ax=ax_F, pad=0.02, fraction=0.046)
    cbar_F.set_label("F* [$M]", fontsize=FS)
    cbar_F.ax.tick_params(labelsize=FS)

    sc_c, _ = draw_map(ax_cov, a, inc, cov, title=f"{lab}:  coverage",
                       cmap=cmap)
    ax_cov.set_ylabel("")
    cbar_c = fig.colorbar(sc_c, ax=ax_cov, pad=0.02, fraction=0.046)
    cbar_c.set_label("Coverage [%]", fontsize=FS)
    cbar_c.ax.tick_params(labelsize=FS)

    out = os.path.join(args.out_dir, args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
