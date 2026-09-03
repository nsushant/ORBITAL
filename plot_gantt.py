"""
plot_gantt.py — Gantt chart comparing MDLS and NSGA-III schedules.

Each row = one servicer vehicle.
Bars: transit legs (grey), service visits (coloured by satellite),
      intermediate depot visits (orange).

Run (knee):     python3 plot_gantt.py
Run (max-cov):  python3 plot_gantt.py --mdls outputs/maxcov_schedule_MDLS.json
                                     --nsga outputs/maxcov_schedule_NSGA-III.json
                                     --out  outputs/gantt_maxcov.pdf
"""

import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap

parser = argparse.ArgumentParser()
parser.add_argument("--mdls", default="outputs/knee_schedule_MDLS.json")
parser.add_argument("--nsga", default="outputs/knee_schedule_NSGA-III.json")
parser.add_argument("--out",  default="outputs/gantt.pdf")
args = parser.parse_args()

FILES = {"MDLS": args.mdls, "NSGA-III": args.nsga}

DEPOT_COLOR   = "#ff7f0e"   # orange for intermediate depot visits
TRANSIT_COLOR = "#d0d0d0"

def load_schedule(path):
    with open(path) as f:
        return json.load(f)["schedule"]

def plot_gantt(ax, schedule, title):
    all_sats = sorted({s for v in schedule for s in v["visitedSAT"]
                       if not s.startswith("depot")})
    cmap = matplotlib.colormaps.get_cmap("tab20")
    sat_color = {s: cmap(i % 20) for i, s in enumerate(all_sats)}

    n            = len(schedule)
    bar_h        = 0.6
    DEPOT_MIN_W  = 2.0   # minimum bar width so zero-dwell start depot is visible
    has_depot_visit = False

    for v_idx, vehicle in enumerate(schedule):
        sats       = vehicle["visitedSAT"]
        arrivals   = vehicle["arrivals"]
        departures = vehicle["departures"]
        y = n - 1 - v_idx

        for i in range(len(sats)):
            arr = arrivals[i]
            dep = departures[i]
            sat = sats[i]
            is_intermediate_depot = sat.startswith("depot") and any(
                not sats[j].startswith("depot") for j in range(i + 1, len(sats))
            )

            if sat.startswith("depot"):
                if is_intermediate_depot:
                    has_depot_visit = True
                # skip duplicate trailing depots (no satellite after them)
                is_trailing_duplicate = (
                    sat.startswith("depot") and not is_intermediate_depot
                    and i > 0 and sats[i - 1].startswith("depot")
                )
                if not is_trailing_duplicate:
                    dur = max(dep - arr, DEPOT_MIN_W)
                    ax.barh(y, dur, left=arr, height=bar_h,
                            color=DEPOT_COLOR, edgecolor="white", linewidth=0.4)
                    # transit to next stop (offset by expanded bar width)
                    if i + 1 < len(sats) and not sats[i + 1].startswith("depot"):
                        t_start = arr + dur
                        t_dur   = arrivals[i + 1] - t_start
                        if t_dur > 0:
                            ax.barh(y, t_dur, left=t_start, height=bar_h,
                                    color=TRANSIT_COLOR, edgecolor="white", linewidth=0.4)
            else:
                # service visit
                dur = dep - arr
                if dur > 0:
                    ax.barh(y, dur, left=arr, height=bar_h,
                            color=sat_color[sat], edgecolor="white", linewidth=0.4)
                    ax.text(arr + dur / 2, y, sat.replace("sat_", ""),
                            ha="center", va="center", fontsize=5.5,
                            color="white", fontweight="bold")
                # transit to next stop
                if i + 1 < len(sats):
                    t_dur = arrivals[i + 1] - dep
                    if t_dur > 0:
                        ax.barh(y, t_dur, left=dep, height=bar_h,
                                color=TRANSIT_COLOR, edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"V{n - i}" for i in range(n)], fontsize=9)
    ax.set_xlabel("Time [days]", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    handles = [
        mpatches.Patch(color=TRANSIT_COLOR, label="Transit"),
        mpatches.Patch(color=cmap(0),       label="Service visit"),
        mpatches.Patch(color=DEPOT_COLOR,   label="Depot" if not has_depot_visit else "Depot / Refuel"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

for ax, (label, path) in zip(axes, FILES.items()):
    plot_gantt(ax, load_schedule(path), label)

fig.tight_layout(h_pad=3)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved: {args.out}")
