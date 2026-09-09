"""plot_depot_coverage_map.py — (a, i) recovered-value surface per scenario.

For each grid location the surface shows the *max-coverage* operating point
from `outputs/depot_sweep_{scenario}.h5`: recovered request value [$M].
Locations where no client is reachable within the dv budget are black. The
depots chosen by the facility-location MILP at one (B, theta) contract cell
under its per-depot Delta-V cap (from `depot_milp_surface_{scenario}.h5`) are
overlaid as white-ringed markers, each labelled with its mission Delta-V.

Run: python plot_depot_coverage_map.py
"""

import argparse
import os

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from plots import map_style as pbm

pbm.INC_LO = 50.0
pbm.INC_HI = 100.0
pbm.FS     = 20
FS_TICK    = 18

SCENARIOS = ["S1_repair", "S2_refuel", "S3_deorbit"]


def load_sweep(path):
    with h5py.File(path, "r") as f:
        return dict(a_km=f["a_km"][()], incl_deg=f["incl_deg"][()],
                    served=f["served_request_value"][()],
                    dv=f["dv_m_s"][()],
                    total_value=float(f.attrs["total_value_usd"]),
                    scenario=f.attrs["scenario"])


def load_depots(path, B, theta, starlink_loss, planet_loss):
    with h5py.File(path, "r") as f:
        mask = np.isclose(f["B_usd"][()], B)
        if "starlink_loss_frac" in f and np.isfinite(f["starlink_loss_frac"][()]).any():
            mask &= np.isclose(f["starlink_loss_frac"][()], starlink_loss)
            mask &= np.isclose(f["planet_loss_frac"][()], planet_loss)
        else:
            mask &= np.isclose(f["theta"][()], theta)
        if not mask.any():
            return None
        return (f["a_km"][()][mask], f["incl_deg"][()][mask],
                f["dv_m_s"][()][mask])


def masked(live, values):
    values = np.asarray(values, float).copy()
    values[~np.asarray(live, bool)] = np.nan
    return values


def _keep_finite_upsample(ua, ui, Z, fa=12, fi=12):
    """Upsample a mesh, filling each sub-cell from the corners that are finite.

    pbm._bilinear_upsample fills a 2x2 base cell only when all four corners are
    finite, which erases isolated live cells (both neighbours dead). Here every
    sub-cell whose base cell touches any finite corner is filled, so each live
    location is visible; regions with no finite corner anywhere stay NaN/black.
    """
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
            fin = np.isfinite(corners)
            if not fin.any():
                continue
            nx_s = fa + 1 if i == nx - 2 else fa
            ny_s = fi + 1 if j == ny - 2 else fi
            u = np.linspace(0.0, 1.0, nx_s)
            v = np.linspace(0.0, 1.0, ny_s)
            U, V = np.meshgrid(u, v)
            if fin.all():
                z00, z01 = corners[0, 0], corners[0, 1]
                z10, z11 = corners[1, 0], corners[1, 1]
                fill = ((1 - U) * (1 - V) * z00 + U * (1 - V) * z01
                        + (1 - U) * V * z10 + U * V * z11)
            else:
                fill = np.full((ny_s, nx_s), float(corners[fin].mean()))
            Zd[j * fi:j * fi + ny_s, i * fa:i * fa + nx_s] = fill
    return ai, ii, Zd


def draw_map(ax, a, inc, values, norm=None, cmap=None):
    """Filled (a, i) surface; black cells = no viable location there."""
    a, inc, values = np.asarray(a, float), np.asarray(inc, float), np.asarray(values, float)
    keep = (inc >= pbm.INC_LO) & (inc <= pbm.INC_HI)
    a, inc, values = a[keep], inc[keep], values[keep]
    ok = np.isfinite(values)
    cmap = cmap.copy()
    cmap.set_bad("black")
    if norm is None:
        vmax = float(np.nanmax(values[ok])) if ok.any() else 1.0
        vmin = float(np.nanmin(values[ok])) if ok.any() else 0.0
        if not np.isfinite(vmin) or vmax <= vmin:
            vmin, vmax = 0.0, max(vmax, 1e-3)
        norm = Normalize(vmin=vmin, vmax=vmax)

    ax.set_facecolor("black")
    ua, ui, Z = pbm._regular_grid(a, inc, values)
    if len(ua) >= 2 and len(ui) >= 2:
        ai, ii, Zd = _keep_finite_upsample(ua, ui, Z)
        Zm = np.ma.masked_invalid(Zd)
        sc = ax.pcolormesh(ai, ii, Zm, cmap=cmap, norm=norm, shading="auto", zorder=2)
        x0, x1 = float(ua[0]), float(ua[-1])
        y0, y1 = float(ui[0]), float(ui[-1])
    else:
        sc = ax.scatter(a[ok], inc[ok], c=values[ok], cmap=cmap, norm=norm,
                        s=70, zorder=3, edgecolors="0.9", linewidths=0.4)
        x0, x1 = float(a.min()), float(a.max())
        y0, y1 = float(inc.min()), float(inc.max())
    pad_a = 150.0
    pad_i = 3.0
    ax.set_xlim(x0 - pad_a, x1 + pad_a)
    ax.set_ylim(y0 - pad_i, y1 + pad_i)
    return sc, norm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--sweep-dir", default="outputs")
    p.add_argument("--surface-dir", default="outputs")
    p.add_argument("--B", type=float, default=2.0e9)
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--starlink-loss", type=float, default=0.4)
    p.add_argument("--planet-loss", type=float, default=0.4)
    p.add_argument("--surface-prefix", default="depot_milp_grouped")
    p.add_argument("--out", default="figures/depot_coverage_map.pdf")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    scenarios = [s for s in args.scenarios.split(",") if s]
    sweeps = [load_sweep(os.path.join(args.sweep_dir,
                                      f"depot_sweep_{s}.h5")) for s in scenarios]

    live = [sw["served"] > 0 for sw in sweeps]
    rec_all = np.concatenate([sw["served"][m] for sw, m in zip(sweeps, live)]) / 1e6
    rec_lo, rec_hi = float(rec_all.min()), float(rec_all.max())
    if rec_hi <= rec_lo:
        rec_hi = rec_lo + 1.0

    cmap = pbm.paper_cmap()
    metrics = [
        ("recovered value", "Recovered value [\$M]",
         lambda sw: sw["served"] / 1e6,
         Normalize(vmin=rec_lo, vmax=rec_hi)),
    ]
    n_row, n_col = 1, len(sweeps)
    fig, axes = plt.subplots(n_row, n_col, figsize=(6.4 * n_col, 5.6 * n_row),
                             sharey=True, sharex=True, layout="constrained")
    axes = np.atleast_1d(axes).reshape(n_row, n_col)

    sci = []
    axes = np.atleast_2d(axes)
    for i, sw in enumerate(sweeps):
        dep = None
        surf = os.path.join(args.surface_dir,
                            f"{args.surface_prefix}_{sw['scenario']}.h5")
        if os.path.exists(surf):
            got = load_depots(surf, args.B, args.theta,
                              args.starlink_loss, args.planet_loss)
            if got is not None:
                dep = got
                print(f"{sw['scenario']} @ B=${args.B/1e9:.1f}B, "
                      f"SL loss={args.starlink_loss}, "
                      f"PL loss={args.planet_loss}: "
                      f"{len(dep[0])} depot(s)")
                for aq, iq, dv in zip(*dep):
                    print(f"  a={aq:.0f} km i={iq:.1f}°  mission dv={dv:,.0f} m/s")
        for j, (label, clabel, getter, norm) in enumerate(metrics):
            ax = axes[i % n_row, i // n_row]
            vals = masked(live[i], getter(sw))
            sc, _ = draw_map(ax, sw["a_km"], sw["incl_deg"], vals,
                             norm=norm, cmap=cmap)
            if dep is not None:
                ax.scatter(dep[0], dep[1], s=120, marker="o",
                           facecolor="none", edgecolors="white",
                           linewidths=2.2, zorder=5, clip_on=False)
                for aq, iq, dv in zip(*dep):
                    ax.annotate(f"{dv:,.0f}", (aq, iq),
                                textcoords="offset points", xytext=(7, 7),
                                fontsize=13, color="white", zorder=6,
                                clip_on=False)
            ax.set_title(f"{sw['scenario'].replace('_', ' ')} · {label}",
                         fontsize=pbm.FS)
            if j:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Inclination i [deg]", fontsize=pbm.FS)
            if i < n_row - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Semi-major axis a [km]", fontsize=pbm.FS)
            ax.tick_params(axis="both", which="major", labelsize=FS_TICK,
                           length=6, width=1.2)
            sci.append(sc)

    cbar = fig.colorbar(sci[0], ax=list(axes.ravel()), pad=0.02, fraction=0.02)
    cbar.set_label("Recovered value [\$M]", fontsize=pbm.FS)
    cbar.ax.tick_params(labelsize=pbm.FS)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
