"""Generate the benchmark demand sets for S1/S2/S3, and report their coverage.

Replaces generate_experiment_demands.jl. Writes one HDF5 file per
(scenario, trial) into outputs/exp_demands/, readable by
oos.demands.load_demands into an oos.schedule.Demands.

Run:
    python generate_demands.py                      # defaults below
    python generate_demands.py --arrival-horizon 1826 --n-demands 400
    python generate_demands.py --dry-run            # report coverage, write nothing

The coverage table it prints is the point of the --dry-run mode. A demand
whose client cannot be reached from the depot within its own window before
its deadline is unservable no matter which algorithm runs, and a scenario
made mostly of those measures nothing. That check is what was missing when
the 90-180 day S1 window reached the paper: nothing in this instance can be
reached before day 105.

The default arrival horizon is deliberately tied to the cost table rather
than to the calendar. The table's departure grid stops at day 390, so a
request released after that has no departure epoch available to it at all,
whatever its deadline. --arrival-horizon accepts anything, and the coverage
report will show what a longer one actually costs.
"""

import argparse
import os
import sys

import numpy as np

from oos.schedule import load_cost_table
from oos import demands as dm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cost-table", default="outputs/cost_table.h5")
    p.add_argument("--population", default="outputs/instance_population.csv")
    p.add_argument("--out-dir", default="outputs/exp_demands")
    p.add_argument("--n-demands", type=int, default=400,
                   help="requests per (scenario, trial); repeats across clients allowed")
    p.add_argument("--arrival-horizon", type=float, default=None,
                   help="days over which requests arrive; defaults to the cost "
                        "table's last departure epoch")
    p.add_argument("--trial-offset", type=int, default=0,
                   help="added to every trial number, so a tuning set drawn "
                        "with --trial-offset 100 shares no instance with the "
                        "test set (TODO 2 requires them disjoint)")
    p.add_argument("--trials", type=int, default=5,
                   help="trials per scenario (Sec. 4 currently reports 5)")
    p.add_argument("--dry-run", action="store_true",
                   help="report coverage without writing any files")
    args = p.parse_args()

    ct = load_cost_table(args.cost_table)
    try:
        depot = ct.names.index("depot_1")
    except ValueError:
        sys.exit(f"'depot_1' not in {args.cost_table}")

    horizon = args.arrival_horizon
    if horizon is None:
        horizon = float(ct.dep_days[-1])

    pop = dm.load_population(args.population, ct)

    print(f"cost table : {ct.n_nodes} nodes, budget {ct.dv_budget:.0f} m/s, "
          f"departures {ct.dep_days[0]:.0f}-{ct.dep_days[-1]:.0f} d, "
          f"tof {ct.tof_days[0]:.0f}-{ct.tof_days[-1]:.0f} d")
    print(f"population : {len(pop)} clients, "
          f"${pop.value.sum()/1e6:.1f} M total recovery potential")
    print(f"generating : {args.n_demands} demands/trial, {args.trials} trials, "
          f"arrivals over {horizon:.0f} d ({horizon/365.25:.2f} yr)")
    if horizon > ct.dep_days[-1]:
        print(f"  ! arrivals run past the last departure epoch "
              f"({ct.dep_days[-1]:.0f} d): requests released after it have no "
              f"departure available and cannot be served by any algorithm.")
    print()

    hdr = f"{'scenario':12s} {'window (d)':>12s} {'trial':>5s} {'demands':>8s} " \
          f"{'servable':>9s} {'%':>6s} {'value at risk':>14s}"
    print(hdr)
    print("-" * len(hdr))

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for scen, cfg in dm.SCENARIOS.items():
        lo, hi = cfg["window"]
        fracs = []
        for t_i in range(1, args.trials + 1):
            trial = t_i + args.trial_offset
            d = dm.generate(scen, trial, pop, args.n_demands, horizon)
            ok = dm.direct_servable(d, ct, depot)
            fracs.append(ok.mean())
            print(f"{scen:12s} {f'{lo:.0f}-{hi:.0f}':>12s} {trial:5d} "
                  f"{len(d):8d} {int(ok.sum()):9d} {100*ok.mean():5.1f}% "
                  f"{'$' + format(d.value.sum()/1e6, '.1f') + ' M':>14s}")
            if not args.dry_run:
                path = os.path.join(args.out_dir, f"{scen}_{trial:02d}.h5")
                dm.save_demands(path, d, attrs=dict(
                    scenario=scen, trial=trial, window_lo=lo, window_hi=hi,
                    arrival_horizon_days=horizon, n_demands=args.n_demands,
                    seed=dm.seed_for(scen, trial),
                    cost_table=os.path.basename(args.cost_table),
                    population=os.path.basename(args.population)))
        summary[scen] = float(np.mean(fracs))
        print()

    print("mean directly servable fraction:")
    for scen, frac in summary.items():
        print(f"  {scen:12s} {100*frac:5.1f}%")
    print("\n('directly servable' = some affordable depot->client leg fits inside\n"
          " the request's own release/deadline window. Sufficient, not necessary:\n"
          " a client may still be reachable mid-tour from another client.)")

    if args.dry_run:
        print("\n--dry-run: no files written.")
    else:
        print(f"\nwrote {len(dm.SCENARIOS) * args.trials} files to {args.out_dir}/")


if __name__ == "__main__":
    main()
