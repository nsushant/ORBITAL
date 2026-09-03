"""
plot_depot_bcr_maps_thrust.py — depot BCR* maps at several thrust levels.

Rows = clients, columns = thrust tags.

  python3 plot_depot_bcr_maps_thrust.py --tags 10mN 100mN 1N
"""

import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from plot_depot_bcr_map import load_locations, draw_map, FS, paper_cmap
from bcr_model import client_label


def pretty_tag(tag):
    if tag.endswith("mN"):
        return f"{tag[:-2]} mN"
    if tag.endswith("N"):
        return f"{tag[:-1]} N"
    return tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["10mN", "100mN", "1N"])
    parser.add_argument("--h5-fmt", default="outputs/depot_location_sweep_{tag}.h5")
    parser.add_argument("--scenario", default="bcr_mixed_7000")
    parser.add_argument("--out", default="outputs/depot_location_sweep/depot_bcr_maps_thrust.pdf")
    args = parser.parse_args()

    # Discover clients from the first available H5.
    clients = []
    tagged = []
    for tag in args.tags:
        path = args.h5_fmt.format(tag=tag)
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        _, cids = load_locations(path, args.scenario)
        if not clients:
            clients = list(cids) or ["mix"]
        tagged.append((tag, path))
    if not tagged:
        raise SystemExit("No sweep H5 files found. Run bash run_depot_thrust_maps.sh")

    cells = []
    all_ok = []
    for tag, path in tagged:
        for c in clients:
            rows, _ = load_locations(path, args.scenario, client=c)
            a   = np.array([r[0] for r in rows])
            inc = np.array([r[1] for r in rows])
            F_M = np.array([r[2] for r in rows])
            cells.append((tag, c, a, inc, F_M))
            all_ok.append(F_M[np.isfinite(F_M)])
            print(f"{tag} {c}: {np.isfinite(F_M).sum()}/{len(F_M)} with BCR* ≥ 1")

    concat = np.concatenate([x for x in all_ok if len(x)]) if any(len(x) for x in all_ok) \
             else np.array([1.0])
    vmax = max(float(np.nanmax(concat)) if len(concat) else 1.0, 1e-3)
    vmin = float(np.nanmin(concat)) if len(concat) else 0.0
    if not np.isfinite(vmin) or vmax <= vmin:
        vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = paper_cmap()

    n_t, n_c = len(tagged), len(clients)
    fig, axes = plt.subplots(n_c, n_t, figsize=(7.2 * n_t, 5.8 * n_c),
                             sharey=True, sharex=True, layout="constrained")
    axes = np.atleast_2d(axes)
    sc = None
    for i, c in enumerate(clients):
        for j, (tag, _) in enumerate(tagged):
            ax = axes[i, j]
            _, _, a, inc, F_M = cells[j * n_c + i]
            title = pretty_tag(tag) if i == 0 else None
            sc, _ = draw_map(ax, a, inc, F_M, title=title,
                             show_legend=(i == 0 and j == 0),
                             norm=norm, cmap=cmap)
            if j == 0:
                ax.set_ylabel(f"{client_label(c)}\nInclination i [deg]", fontsize=FS)
            else:
                ax.set_ylabel("")
            if i < n_c - 1:
                ax.set_xlabel("")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02, fraction=0.03)
        cbar.set_label("Implied contract F* [$M]  (knee, BCR* ≥ 1)", fontsize=FS)
        cbar.ax.tick_params(labelsize=FS - 2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
