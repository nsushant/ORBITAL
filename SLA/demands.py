"""Demand generation and loading for the three benchmark scenarios.

Replaces the Julia `demands/generate_demands.jl` + `generate_experiment_demands.jl`
pair. What changed, and why:

  * Output is an `oos.schedule.Demands`: node indices into the cost table, plus
    release, deadline, service and value. The Julia files keyed demands by
    satellite *name* against the pre-rebuild instance, carried no release time,
    and re-randomised asset values per trial (a V1/V2 coin flip). Values now
    come from `outputs/instance_population.csv`, which prices every client from
    its real hardware class and launch date (Sec. 3.5), so f3 means the same
    thing in every trial. D18: a value is required, there is no fallback.

  * The spatial filter is gone. The Julia pruned candidates by "min delta-V
    from depot <= deltaV_dist", with cutoffs of 5000-12000 m/s calibrated
    against the *old* Julia Edelbaum costs -- meaningless against a 500 m/s
    budget and the corrected model (F16/D17). Demands are drawn across the
    whole client population; whether one is reachable is the depot study's
    subject, not something the generator should quietly decide (user's call).

  * Demands arrive over time. Each request has a release epoch drawn over the
    arrival horizon and a deadline of `release + window`, where the window is
    what distinguishes the three scenarios. That makes the scenario axis "how
    much notice you get" rather than "how far into a fixed horizon the
    deadline falls", which is what the repair / refuel / deorbit framing
    actually describes.

Scenario windows (Sec. 4, Table tab:scenarios). S1 was specified as 90-180 d
and moved to 120-240 d: the earliest arrival any client can be reached by
within a 500 m/s budget is day 105, so a 90-day deadline was unservable by
construction and the tight end of that scenario tested nothing. See the
Round 10 entry in PAPER_COMPLETION_PLAN.md.

A note on multiplicity. A client may draw several requests, and each carries
that client's full recovery potential, so an unserved client with three
requests contributes its value three times to f3. That follows Eq.
(objective) read literally -- f3 = sum of p_q u_q over requests, not over
satellites -- but it does mean f3 is a value-weighted request count rather
than a portfolio value once repeats are allowed.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import hashlib

import numpy as np

from .schedule import Demands

# Windows in seconds from request release to deadline.
SCENARIOS = {
    "S1_repair":  dict(window=(10368000.0, 20736000.0), label="Reactive repair"),
    "S2_refuel":  dict(window=(15552000.0, 31536000.0), label="Planned refuelling"),
    "S3_deorbit": dict(window=(31536000.0, 63072000.0), label="End-of-life deorbit"),
}

SERVICE_S = (86400.0, 432000.0)  # Uniform[86400, 432000] seconds, stored in seconds


@dataclass
class Population:
    """Client nodes and their recovery potential, from the instance CSV."""

    node: np.ndarray      # index into the cost table's node list
    name: list
    value: np.ndarray     # $ per client

    def __len__(self):
        return len(self.node)


def load_population(csv_path, cost_table):
    """Match instance_population.csv rows to cost-table node indices by name.

    The depot carries no recovery potential and is not a client, so it is
    dropped here rather than filtered at every call site.
    """
    index = {n: i for i, n in enumerate(cost_table.names)}
    node, name, value = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["group"] == "depot":
                continue
            if row["name"] not in index:
                raise SystemExit(
                    f"{csv_path} names a client ({row['name']}) that is not in "
                    f"the cost table. The two were built from different "
                    f"instances -- rebuild one of them.")
            node.append(index[row["name"]])
            name.append(row["name"])
            value.append(float(row["value_usd"]))
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or (value < 0).any():
        raise SystemExit(f"{csv_path} has missing or negative value_usd entries; "
                         "f3 is unrecovered value in dollars (D18).")
    return Population(np.asarray(node, dtype=np.int64), name, value)


def seed_for(scenario, trial):
    """Reproducible per (scenario, trial), and independent across scenarios.

    Built on blake2b rather than `hash`. Python randomises string hashing per
    process unless PYTHONHASHSEED is set, so the previous version returned a
    different seed on every run: three consecutive processes gave 1407235557,
    1634963936 and 1075576696 for the same arguments. The demand sets on disk
    were therefore not regenerable from the code that claims to produce them,
    and nothing said so -- the files exist, so nobody looks (F42).
    """
    h = hashlib.blake2b(f"{scenario}|{trial}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 31)


def generate(scenario, trial, population, n_demands, arrival_horizon_s):
    """Draw one scenario's demand set.

    Requests arrive uniformly over [0, arrival_horizon_s] seconds; each gets a
    window drawn uniformly from the scenario's range, a service time from
    Uniform[86400, 432000] seconds, and a client drawn uniformly from the whole
    population, with repeats allowed.
    """
    if scenario not in SCENARIOS:
        raise SystemExit(f"unknown scenario {scenario!r}; "
                         f"expected one of {sorted(SCENARIOS)}")
    lo, hi = SCENARIOS[scenario]["window"]
    rng = np.random.default_rng(seed_for(scenario, trial))

    pick = rng.integers(0, len(population), n_demands)
    release = rng.uniform(0.0, arrival_horizon_s, n_demands)
    deadline = release + rng.uniform(lo, hi, n_demands)
    service = rng.uniform(*SERVICE_S, n_demands)

    return Demands(
        node=population.node[pick].copy(),
        release=release,
        deadline=deadline,
        service=service,
        value=population.value[pick].copy(),
    )


# ---------------------------------------------------------------------------
# Coverage: what fraction of a demand set is servable at all
# ---------------------------------------------------------------------------


def direct_servable(demands, cost_table, depot_node):
    """Mask of demands reachable by a single depot -> client leg in window.

    This is a *sufficient* condition for servability, not a necessary one: a
    client with no affordable direct leg may still be reachable partway
    through a tour from another client. It is reported because it is cheap,
    exact as far as it goes, and bounds from below how much of a scenario is
    solvable at all -- the check whose absence let a 90-day S1 deadline into
    the paper's scenario table when nothing can be reached before day 105.
    """
    dep = cost_table.dep_s
    tof = cost_table.tof_s
    arrive = dep[:, None] + tof[None, :] + cost_table.phasing[depot_node]  # (n,ndep,ntof)
    dv = cost_table.dv[depot_node]
    affordable = (~np.isnan(dv)) & (dv <= cost_table.dv_budget)

    ok = np.zeros(len(demands), dtype=bool)
    for q in range(len(demands)):
        j = demands.node[q]
        # depart no earlier than the request exists, finish before its deadline
        in_window = (dep[:, None] >= demands.release[q]) & \
                    (arrive[j] <= demands.deadline[q] - demands.service[q])
        ok[q] = bool((affordable[j] & in_window).any())
    return ok


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def save_demands(path, demands, attrs=None):
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("node", data=demands.node)
        f.create_dataset("release", data=demands.release)
        f.create_dataset("deadline", data=demands.deadline)
        f.create_dataset("service", data=demands.service)
        f.create_dataset("value", data=demands.value)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v


def load_demands(path):
    """Read a demand file into `oos.schedule.Demands`.

    Demands validates on construction, so a file missing values or carrying a
    negative one raises here rather than at evaluation time.
    """
    import h5py

    if not os.path.exists(path):
        raise SystemExit(
            f"Demand file not found: {path}\n"
            f"  Generate the scenario demand sets first:\n"
            f"    python generate_demands.py")
    with h5py.File(path, "r") as f:
        return Demands(
            node=f["node"][:], release=f["release"][:], deadline=f["deadline"][:],
            service=f["service"][:], value=f["value"][:])
