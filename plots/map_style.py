"""Plotting helpers shared by the paper's depot-location map."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

INC_LO = 50.0
INC_HI = 100.0
FS = 20


def paper_cmap(name="plasma"):
    base = plt.get_cmap(name)
    colors = base(np.linspace(0.08, 1.0, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_paper", colors)
    cmap.set_bad("black")
    return cmap


def _regular_grid(a, inc, values):
    a = np.asarray(a, float)
    inc = np.asarray(inc, float)
    values = np.asarray(values, float)
    ua = np.unique(np.round(a, 6))
    ui = np.unique(np.round(inc, 6))
    grid = np.full((len(ui), len(ua)), np.nan)
    ia = {value: index for index, value in enumerate(ua)}
    ii = {value: index for index, value in enumerate(ui)}
    for a_km, inc_deg, value in zip(np.round(a, 6), np.round(inc, 6), values):
        if a_km in ia and inc_deg in ii:
            grid[ii[inc_deg], ia[a_km]] = value
    return ua, ui, grid


def _bilinear_upsample(ua, ui, grid, fa=12, fi=12):
    ua = np.asarray(ua, float)
    ui = np.asarray(ui, float)
    grid = np.asarray(grid, float)
    ny, nx = grid.shape
    if ny < 2 or nx < 2:
        return ua, ui, grid
    a_dense = np.linspace(ua[0], ua[-1], (nx - 1) * fa + 1)
    i_dense = np.linspace(ui[0], ui[-1], (ny - 1) * fi + 1)
    dense = np.full((len(i_dense), len(a_dense)), np.nan)
    for row in range(ny - 1):
        for col in range(nx - 1):
            corners = grid[row:row + 2, col:col + 2]
            if not np.all(np.isfinite(corners)):
                continue
            nx_sample = fa + 1 if col == nx - 2 else fa
            ny_sample = fi + 1 if row == ny - 2 else fi
            u = np.linspace(0.0, 1.0, nx_sample)
            v = np.linspace(0.0, 1.0, ny_sample)
            uu, vv = np.meshgrid(u, v)
            z00, z01 = corners[0, 0], corners[0, 1]
            z10, z11 = corners[1, 0], corners[1, 1]
            dense[row * fi:row * fi + ny_sample,
                  col * fa:col * fa + nx_sample] = (
                (1 - uu) * (1 - vv) * z00 + uu * (1 - vv) * z01
                + (1 - uu) * vv * z10 + uu * vv * z11)
    return a_dense, i_dense, dense


def draw_map(ax, a, inc, values, title=None, norm=None, cmap=None):
    a = np.asarray(a, float)
    inc = np.asarray(inc, float)
    values = np.asarray(values, float)
    keep = (inc >= INC_LO) & (inc <= INC_HI)
    a, inc, values = a[keep], inc[keep], values[keep]
    finite = np.isfinite(values)
    cmap = paper_cmap() if cmap is None else cmap.copy()
    cmap.set_bad("black")
    if norm is None:
        vmax = float(np.nanmax(values[finite])) if finite.any() else 1.0
        vmin = float(np.nanmin(values[finite])) if finite.any() else 0.0
        if not np.isfinite(vmin) or vmax <= vmin:
            vmin, vmax = 0.0, max(vmax, 1e-3)
        norm = Normalize(vmin=vmin, vmax=vmax)
    ax.set_facecolor("black")
    ua, ui, grid = _regular_grid(a, inc, values)
    if len(ua) >= 2 and len(ui) >= 2:
        a_dense, i_dense, dense = _bilinear_upsample(ua, ui, grid)
        artist = ax.pcolormesh(a_dense, i_dense, np.ma.masked_invalid(dense),
                               cmap=cmap, norm=norm, shading="auto", zorder=2)
    else:
        artist = ax.scatter(a[finite], inc[finite], c=values[finite], cmap=cmap,
                            norm=norm, s=70, zorder=3, edgecolors="0.9",
                            linewidths=0.4)
    ax.set_xlabel("Semi-major axis a [km]", fontsize=FS)
    ax.set_ylabel("Inclination i [deg]", fontsize=FS)
    ax.set_ylim(INC_LO, INC_HI)
    ax.tick_params(labelsize=FS)
    if title:
        ax.set_title(title, fontsize=FS)
    return artist, norm
