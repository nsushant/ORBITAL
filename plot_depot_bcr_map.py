"""
plot_depot_bcr_map.py — Depot (a, i) surface coloured by implied-contract F*.

At each location, pick the cheapest mutually viable equal-BCR contract:
argmin F* among Pareto points with BCR* >= 1. Colour is that F* [$M].
Black = no such point.

Run: python3 plot_depot_bcr_map.py --h5 outputs/depot_location_sweep.h5
"""

import os, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from bcr_model import (
    unpack_front, min_viable_contract, client_label, DV_BUDGET,
)

BUDGET    = float(DV_BUDGET)
INC_LO    = 76.0              # map window (clips black edge bands)
INC_HI    = 84.0
FS        = 34
CMAP_NAME = "plasma"


def paper_cmap(name=CMAP_NAME):
    """Plasma, trimmed off the near-black end so BCR*<1 stays distinct."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.08, 1.0, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_paper", colors)
    cmap.set_bad("black")
    return cmap


def viable_scores(front, budget=BUDGET):
    """Per client: (F_M or nan if not viable, bcr, f3, coverage_pct)."""
    out = {}
    for c, s in min_viable_contract(front, budget).items():
        F_M = s["F"] / 1e6 if s["viable"] else float("nan")
        tot = s["recovered"] + s["unserved"]
        cov = (s["recovered"] / tot * 100.0) if s["viable"] and tot > 0 else float("nan")
        out[c] = (F_M, s["bcr"], s["f3"], cov)
    return out


def load_locations(h5_path, scenario, client=None):
    """Rows (a_km, inc_deg, F_M, bcr, f3, cov). F_M is nan if BCR* < 1."""
    rows = []
    clients = []
    with h5py.File(h5_path, "r") as f:
        if "loc" not in f:
            raise SystemExit(f"No 'loc' group in {h5_path}")
        for name in sorted(f["loc"].keys()):
            g = f["loc"][name]
            a_km    = float(g.attrs.get("a_km", np.nan))
            inc_deg = float(g.attrs.get("inc_deg", np.nan))
            path = f"mdls/{scenario}/trial_01"
            if path not in g:
                rows.append((a_km, inc_deg, np.nan, np.nan, np.nan, np.nan))
                continue
            front = unpack_front(g[path])
            if not clients:
                clients = list(front["clients"])
            scores = viable_scores(front)
            use = client if client is not None else (clients[0] if clients else None)
            if use is None or use not in scores:
                rows.append((a_km, inc_deg, np.nan, np.nan, np.nan, np.nan))
            else:
                F_M, bcr, f3, cov = scores[use]
                rows.append((a_km, inc_deg, F_M, bcr, f3, cov))
    return rows, clients


def _regular_grid(a, inc, values):
    """Pack scattered (a, i, z) onto a unique sorted mesh (n_i, n_a)."""
    a, inc, values = np.asarray(a, float), np.asarray(inc, float), np.asarray(values, float)
    ua = np.unique(np.round(a, 6))
    ui = np.unique(np.round(inc, 6))
    Z = np.full((len(ui), len(ua)), np.nan)
    ia = {v: k for k, v in enumerate(ua)}
    ii = {v: k for k, v in enumerate(ui)}
    for ak, ik, zk in zip(np.round(a, 6), np.round(inc, 6), values):
        if ak in ia and ik in ii:
            Z[ii[ik], ia[ak]] = zk
    return ua, ui, Z


def _bilinear_upsample(ua, ui, Z, fa=12, fi=12):
    """Upsample a regular mesh; a 2×2 cell is filled only if all four corners are finite."""
    ua, ui, Z = np.asarray(ua, float), np.asarray(ui, float), np.asarray(Z, float)
    ny, nx = Z.shape
    if ny < 2 or nx < 2:
        return ua, ui, Z
    ai = np.linspace(ua[0], ua[-1], (nx - 1) * fa + 1)
    ii = np.linspace(ui[0], ui[-1], (ny - 1) * fi + 1)
    Zd = np.full((len(ii), len(ai)), np.nan)
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = Z[j:j + 2, i:i + 2]
            if not np.all(np.isfinite(corners)):
                continue
            nx_s = fa + 1 if i == nx - 2 else fa
            ny_s = fi + 1 if j == ny - 2 else fi
            u = np.linspace(0.0, 1.0, nx_s)
            v = np.linspace(0.0, 1.0, ny_s)
            U, V = np.meshgrid(u, v)
            z00, z01 = corners[0, 0], corners[0, 1]
            z10, z11 = corners[1, 0], corners[1, 1]
            Zd[j * fi:j * fi + ny_s, i * fa:i * fa + nx_s] = (
                (1 - U) * (1 - V) * z00 + U * (1 - V) * z01
                + (1 - U) * V * z10 + U * V * z11)
    return ai, ii, Zd


def draw_map(ax, a, inc, values, title=None, show_legend=True, norm=None, cmap=None):
    """Filled (a, i) surface of viable F* [$M]. Black = BCR* < 1."""
    a, inc, values = np.asarray(a, float), np.asarray(inc, float), np.asarray(values, float)
    keep = (inc >= INC_LO) & (inc <= INC_HI)
    a, inc, values = a[keep], inc[keep], values[keep]
    ok = np.isfinite(values)
    if cmap is None:
        cmap = paper_cmap()
    else:
        cmap = cmap.copy()
        cmap.set_bad("black")
    if norm is None:
        vmax = float(np.nanmax(values[ok])) if ok.any() else 1.0
        vmin = float(np.nanmin(values[ok])) if ok.any() else 0.0
        if not np.isfinite(vmin) or vmax <= vmin:
            vmin, vmax = 0.0, max(vmax, 1e-3)
        norm = Normalize(vmin=vmin, vmax=vmax)

    ax.set_facecolor("black")
    ua, ui, Z = _regular_grid(a, inc, values)
    if len(ua) >= 2 and len(ui) >= 2:
        ai, ii, Zd = _bilinear_upsample(ua, ui, Z)
        Zm = np.ma.masked_invalid(Zd)
        sc = ax.pcolormesh(ai, ii, Zm, cmap=cmap, norm=norm, shading="auto", zorder=2)
    else:
        sc = ax.scatter(a[ok], inc[ok], c=values[ok], cmap=cmap, norm=norm,
                        s=70, zorder=3, edgecolors="0.9", linewidths=0.4)

    ax.set_xlabel("Semi-major axis a [km]", fontsize=FS)
    ax.set_ylabel("Inclination i [deg]", fontsize=FS)
    ax.set_ylim(INC_LO, INC_HI)
    ax.tick_params(labelsize=FS)
    if title:
        ax.set_title(title, fontsize=FS)
    return sc, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",       default="outputs/depot_location_sweep.h5")
    parser.add_argument("--out-dir",  default="outputs/depot_location_sweep")
    parser.add_argument("--scenario", default="bcr_mixed_7000")
    parser.add_argument("--title",    default="")
    parser.add_argument("--out",      default="depot_bcr_map.pdf")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f"{args.h5} not found — run sweep_depot_location.jl first")

    _, clients = load_locations(args.h5, args.scenario)
    if not clients:
        print("No per-client columns in this H5 — re-run the depot sweep.")
        clients = ["mix"]

    panels = []
    all_F = []
    for c in clients:
        rows, _ = load_locations(args.h5, args.scenario, client=c)
        a   = np.array([r[0] for r in rows])
        inc = np.array([r[1] for r in rows])
        F_M = np.array([r[2] for r in rows])
        bcr = np.array([r[3] for r in rows])
        f3  = np.array([r[4] for r in rows])
        cov = np.array([r[5] for r in rows])
        panels.append((c, a, inc, F_M, bcr, f3, cov, rows))
        ok = np.isfinite(F_M) & (inc >= INC_LO) & (inc <= INC_HI)
        all_F.append(F_M[ok])
        print(f"{c}: {ok.sum()}/{len(F_M)} with BCR* ≥ 1")
        if ok.any():
            i_best = int(np.nanargmin(F_M))
            print(f"  min F* = ${F_M[i_best]:.1f} M  at a={a[i_best]:.0f} km, "
                  f"i={inc[i_best]:.1f}°, f3={f3[i_best]:.0f}, BCR*={bcr[i_best]:.2f}")

    csv_path = os.path.join(args.out_dir, "crossover.csv")
    with open(csv_path, "w") as io:
        hdr = "a_km,inc_deg"
        for c in clients:
            hdr += f",F_M_{c},bcr_{c},f3_{c},cov_{c}"
        io.write(hdr + "\n")
        n = len(panels[0][7])
        for i in range(n):
            a_km, inc_deg = panels[0][7][i][0], panels[0][7][i][1]
            line = f"{a_km:.3f},{inc_deg:.3f}"
            for _, _, _, _, _, _, _, rows in panels:
                line += f",{rows[i][2]},{rows[i][3]},{rows[i][4]},{rows[i][5]}"
            io.write(line + "\n")
    print(f"Wrote {csv_path}")

    def _win(F_M, inc, z):
        keep = (inc >= INC_LO) & (inc <= INC_HI) & np.isfinite(F_M) & np.isfinite(z)
        return z[keep]

    def _norm(vals, lo=0.0, hi=1.0):
        vals = np.asarray(vals, float)
        if len(vals) == 0:
            return Normalize(vmin=lo, vmax=hi), lo, hi
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or vmax <= vmin:
            vmax = vmin + 1.0 if np.isfinite(vmin) else hi
            vmin = lo if not np.isfinite(vmin) else vmin
        return Normalize(vmin=vmin, vmax=vmax), vmin, vmax

    cmap = paper_cmap()
    metrics = [
        ("Depot Location Vs. Contract Size", "Contract Size [$M]",
         lambda p: p[3], None),
        ("Depot Location Vs. Fleet Size", "Fleet size (number of servicers)",
         lambda p: p[5], "int"),
        ("Depot Location Vs. Coverage", "Coverage [%]",
         lambda p: p[6], None),
    ]
    n_c = len(panels)
    n_m = len(metrics)
    fig, axes = plt.subplots(n_c, n_m, figsize=(12.0 * n_m, 10.0 * n_c),
                             sharey=True, sharex=True, layout="constrained")
    axes = np.atleast_2d(axes)
    for i, p in enumerate(panels):
        _, a, inc, F_M = p[0], p[1], p[2], p[3]
        for j, (title, clabel, getter, ticks) in enumerate(metrics):
            ax = axes[i, j]
            z = getter(p)
            norm, vmin, vmax = _norm(_win(F_M, inc, z))
            sc, _ = draw_map(ax, a, inc, z, title=title if i == 0 else None,
                             norm=norm, cmap=cmap)
            if j:
                ax.set_ylabel("")
            if i < n_c - 1:
                ax.set_xlabel("")
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label(clabel, fontsize=FS)
            if ticks == "int":
                tk = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)) + 1)
                if 1 < len(tk) <= 16:
                    cbar.set_ticks(tk)
            cbar.ax.tick_params(labelsize=FS)
    out = os.path.join(args.out_dir, args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
