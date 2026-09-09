"""Run the depot-location MILP using aggregate or operator-specific loss caps.

    python run_depot_fronts_milp.py S1_repair \
        --budgets "2e8,4e8,6e8,8e8,1e9" --loss-values "0.8,0.5,0.2,0.05"

Use ``--starlink-loss-values`` and ``--planet-loss-values`` together to impose
separate loss caps for the two client groups. Each is a fraction of that
group's total value. Without them, ``--loss-values`` retains the earlier
aggregate-cap mode.

Reads : outputs/depot_sweep_{scenario}.h5  (extended schema)
Writes: outputs/depot_milp_surface_{scenario}.csv and .h5
"""

import argparse
import csv
import os

import numpy as np

from oos.depot_fronts_milp import (
    Costs, best_recoverable, load_fronts, min_loss, prune_candidates, solve,
)


def _csv_floats(text):
    return [float(v) for v in text.split(",")]


def _client_groups(sweep, population):
    """Return one normalized operator label for each H5 client."""
    import h5py

    with h5py.File(sweep, "r") as f:
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f["client_names"][:]]
    with open(population, newline="") as fh:
        by_name = {r["name"]: r["group"].strip().lower()
                   for r in csv.DictReader(fh)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"population CSV has no group for {missing[:5]}")
    groups = np.asarray([by_name[name] for name in names])
    unexpected = sorted(set(groups) - {"starlink", "planet"})
    if unexpected:
        raise ValueError(f"unexpected client groups: {unexpected}")
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("key", help="scenario, e.g. S1_repair")
    p.add_argument("--sweep", default=None,
                   help="sweep H5 (default outputs/depot_sweep_{key}.h5)")
    p.add_argument("--out", default=None,
                   help="output prefix (default outputs/depot_milp_surface_{key})")
    p.add_argument("--budgets", default="2e8,4e8,6e8,8e8,1e9",
                   help="contract spend caps B [US$], comma separated")
    p.add_argument("--loss-values", default="0.8,0.5,0.2,0.05",
                   help="lost-value caps, as fractions of total value "
                        "Gamma (an asset lost when ANY of its requests is "
                        "unserved); theta = 1 - loss_frac is kept for "
                        "back-compat with the recovered-value floor")
    p.add_argument("--starlink-loss-values", default=None,
                   help="Starlink lost-value caps as fractions of Starlink "
                        "value; use with --planet-loss-values")
    p.add_argument("--planet-loss-values", default=None,
                   help="Planet Labs lost-value caps as fractions of Planet "
                        "Labs value; use with --starlink-loss-values")
    p.add_argument("--population", default="outputs/instance_population.csv",
                   help="CSV mapping sweep client names to operator groups")
    p.add_argument("--value-basis", default="request", choices=("request", "asset"))
    p.add_argument("--m-dep", type=float, default=300.0)
    p.add_argument("--c-srv", type=float, default=15e6)
    p.add_argument("--c-dep", type=float, default=60e6)
    p.add_argument("--launch-per-kg", type=float, default=3700.0)
    p.add_argument("--xe", type=float, default=340.0)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--mip-gap", type=float, default=0.001,
                   help="relative MIP optimality tolerance (default 0.001)")
    p.add_argument("--dv-cap", type=float, default=12000.0,
                   help="per-depot mission Delta-V cap [m/s], standing in for "
                        "depot propellant storage; 0 disables")
    p.add_argument("--solver", default="auto",
                   choices=("auto", "gurobi", "scipy"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    sweep = args.sweep or f"outputs/depot_sweep_{args.key}.h5"
    prefix = args.out or f"outputs/depot_milp_surface_{args.key}"
    costs = Costs(c_srv=args.c_srv, c_dep=args.c_dep, launch_kg=args.launch_per_kg,
                  m_dry_dep=args.m_dep, c_xe=args.xe)

    front = load_fronts(sweep, value_basis=args.value_basis, costs=costs)
    loc_id, cost = front["loc_id"], front["cost"]
    dv = np.asarray(front["dv"])
    covers = front["covers"]
    coversets = [frozenset(np.flatnonzero(covers[i])) for i in range(covers.shape[0])]
    kept, kept_locs = prune_candidates(loc_id, cost, coversets)
    print(f"{front['n_pts']} front points -> {len(kept)} after pruning "
          f"({len(kept_locs)} live locations), "
          f"total value {front['total_value']:,.0f} $")
    if args.dv_cap and args.dv_cap > 0:
        ok = dv <= args.dv_cap
        n_pts0, n_loc0 = len(kept), len(kept_locs)
        kept = kept[ok[kept]]
        kept_locs = np.unique(loc_id[kept])
        print(f"dv cap {args.dv_cap:.0f} m/s: dropped {n_pts0 - len(kept)}/"
              f"{n_pts0} front points, {n_loc0 - len(kept_locs)}/{n_loc0} "
              f"locations; {len(kept_locs)} live locations remain")
    theta_max = best_recoverable(covers[kept], front["client_value"],
                                 front["total_value"])
    loss_min = min_loss(covers[kept], front["client_value"],
                        front["total_value"])
    print(f"largest recoverable value fraction: {100 * theta_max:.1f} % "
          f"(min lost value {loss_min / 1e6:,.1f} M USD)")

    budgets = _csv_floats(args.budgets)
    group_mode = (args.starlink_loss_values is not None or
                  args.planet_loss_values is not None)
    if group_mode and (args.starlink_loss_values is None or
                       args.planet_loss_values is None):
        p.error("--starlink-loss-values and --planet-loss-values must be used together")
    if group_mode:
        groups = _client_groups(sweep, args.population)
        sl_mask, pl_mask = groups == "starlink", groups == "planet"
        values = np.asarray(front["client_value"])
        sl_total, pl_total = values[sl_mask].sum(), values[pl_mask].sum()
        cases = [(sl, pl) for sl in _csv_floats(args.starlink_loss_values)
                 for pl in _csv_floats(args.planet_loss_values)]
        print(f"group totals: Starlink ${sl_total/1e6:,.1f} M; "
              f"Planet Labs ${pl_total/1e6:,.1f} M")
    else:
        cases = [(loss, None) for loss in _csv_floats(args.loss_values)]
    rows = []
    h5_rows = []
    for B in budgets:
        for loss_frac, planet_loss_frac in cases:
            if group_mode:
                sl_loss_frac = loss_frac
                loss_usd = np.nan
                group_caps = [
                    ("starlink", sl_mask, sl_loss_frac * sl_total),
                    ("planet", pl_mask, planet_loss_frac * pl_total),
                ]
            else:
                sl_loss_frac = planet_loss_frac = np.nan
                loss_usd = loss_frac * front["total_value"]
                group_caps = None
            diagnostics = {}
            sol = solve(loc_id[kept], cost[kept], front["recovered"][kept],
                        covers[kept], front["client_value"],
                        front["total_value"], loss_usd, B,
                        group_loss_caps=group_caps,
                        time_limit=args.time_limit, mip_gap=args.mip_gap,
                        verbose=args.verbose, solver=args.solver,
                        diagnostics=diagnostics)
            base = dict(B_usd=B,
                        starlink_loss_frac=sl_loss_frac,
                        planet_loss_frac=planet_loss_frac,
                        starlink_loss_usd=(sl_loss_frac * sl_total
                                           if group_mode else np.nan),
                        planet_loss_usd=(planet_loss_frac * pl_total
                                         if group_mode else np.nan),
                        loss_frac=(np.nan if group_mode else loss_frac),
                        loss_usd=loss_usd,
                        theta=(np.nan if group_mode else 1.0 - loss_frac),
                        requested_mip_gap=args.mip_gap,
                        **diagnostics)
            if sol is None:
                rows.append(dict(**base, feasible=0,
                                 n_depots="", spend_usd="",
                                 recovered_usd="", recovered_frac="",
                                 recovered_starlink_usd="",
                                 recovered_planet_usd="",
                                 total_dv_m_s="", total_fleet="", locations="",
                                 dv_cap_m_s=args.dv_cap))
                continue
            chosen, ch_loc, ch_cost, ch_rec, rec, n_dep, spend = sol
            chosen_keep = kept[chosen]
            covered = np.any(covers[chosen_keep], axis=0)
            rec_sl = float(values[covered & sl_mask].sum()) if group_mode else np.nan
            rec_pl = float(values[covered & pl_mask].sum()) if group_mode else np.nan
            tot_dv = float(front["dv"][chosen_keep].sum())
            tot_fleet = int(front["fleet"][chosen_keep].sum())
            locs = ";".join(f"{front['a_km'][int(l)]:.0f}/"
                            f"{front['incl_deg'][int(l)]:.1f}".replace(
                                ".0/", "/").replace(".1", ".1")
                            for l in ch_loc)
            rows.append(dict(**base, feasible=1,
                             n_depots=n_dep, spend_usd=round(spend, 2),
                             recovered_usd=round(rec, 2),
                             recovered_frac=rec / front["total_value"],
                             recovered_starlink_usd=rec_sl,
                             recovered_planet_usd=rec_pl,
                             total_dv_m_s=round(tot_dv, 1),
                             total_fleet=tot_fleet, locations=locs,
                             dv_cap_m_s=args.dv_cap))
            for p in chosen_keep:
                h5_rows.append(dict(**base, point=int(p),
                                    loc=int(loc_id[p]),
                                    a_km=front["a_km"][int(loc_id[p])],
                                    incl_deg=front["incl_deg"][int(loc_id[p])],
                                    dv_m_s=front["dv"][p],
                                    fleet=int(front["fleet"][p]),
                                    recovered_usd=front["recovered"][p],
                                    cost_usd=cost[p]))
            cap_label = (f"SL loss {sl_loss_frac:.2f}, PL loss "
                         f"{planet_loss_frac:.2f}" if group_mode else
                         f"loss {loss_frac:.2f} (theta {1-loss_frac:.2f})")
            print(f"  B {B/1e6:8.1f} M  {cap_label} : "
                  f"{n_dep} depot(s), spend ${spend/1e6:9.2f} M, "
                  f"recovered {100 * rec / front['total_value']:5.1f} %, "
                  f"dv {tot_dv:,.0f} m/s, fleet {tot_fleet}  [{locs}]")

    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    with open(f"{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if h5_rows:
        import h5py
        with h5py.File(f"{prefix}.h5", "w") as f:
            for col in h5_rows[0]:
                values = [r[col] for r in h5_rows]
                if isinstance(values[0], str):
                    arr = np.asarray(values, dtype=h5py.string_dtype("utf-8"))
                else:
                    arr = np.asarray(values, dtype=float)
                f.create_dataset(col, data=arr)
            f.attrs["scenario"] = args.key
            f.attrs["value_basis"] = args.value_basis
            f.attrs["total_value_usd"] = front["total_value"]
            f.attrs["c_srv"] = args.c_srv
            f.attrs["c_dep"] = args.c_dep
            f.attrs["launch_per_kg"] = args.launch_per_kg
            f.attrs["m_dry_dep"] = args.m_dep
            f.attrs["c_xe"] = args.xe
            f.attrs["dv_cap_m_s"] = args.dv_cap
            f.attrs["loss_mode"] = "by_operator" if group_mode else "aggregate"
            if group_mode:
                f.attrs["starlink_total_value_usd"] = sl_total
                f.attrs["planet_total_value_usd"] = pl_total
    print(f"wrote {prefix}.csv" + (" and .h5" if h5_rows else ""))


if __name__ == "__main__":
    main()
