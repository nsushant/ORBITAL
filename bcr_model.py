"""OOS implied-contract BCR.

Each client pays a lump-sum mission-epoch contract F. F is not assumed
(no 10% invoice): it is the unique fee that equalizes client and operator
BCR when that client is the sole payer of the fleet cost C.

    F* = (-U + sqrt(U^2 + 4 R C)) / 2
    BCR* = F*/C = R / (F* + U)

R, U are recovered / unserved *for that client*. C is launch + xenon +
manufacturing of the whole fleet plus a one-time depot activation cost.
A later defence instance is one client in the same formulas
(replace outputs/sat_clients.json).
"""

import math
import numpy as np

ISP_XE       = 2500.0   # s — Edelbaum BCR Hall thruster
THRUST_N     = 0.01     # 10 mN
DV_BUDGET    = 7000.0   # m/s tank
M_DRY        = 300.0    # kg
G0           = 9.80665
VE_XE        = ISP_XE * G0

XE_COST      = 340.0
F9_COST      = 74_000_000.0
F9_CAPACITY  = 22_000.0
MARKET_RATE  = F9_COST / F9_CAPACITY

# Manufacturing: $3M bus (current $), first-unit $5M → α = 2/3.
BUS_COST     = 3.0e6
UNIT_COST_N1 = 5.0e6
ALPHA        = UNIT_COST_N1 / BUS_COST - 1.0   # 2/3
_LEARN_EXP   = -0.578                          # log2(0.67)

# One depot, independent of fleet size.
DEPOT_ACTIVATION = 20.0e6

CLIENT_LABELS = {
    "starlink": "Starlink",
    "planet":   "Planet Labs",
    "sda":      "SDA",
    "tracking": "Tracking",
    "transport": "Transport",
}

_CLIENT_PREF = ["starlink", "planet"]


def client_label(cid):
    return CLIENT_LABELS.get(cid, cid.replace("_", " ").title())


def sscm_unit_cost(N, m_dry=None, alpha=None):
    """Average manufacturing cost per servicer [$]. Bus and α are fixed; m_dry is ignored."""
    n = max(1, int(round(float(N))))
    a = ALPHA if alpha is None else float(alpha)
    c_total = BUS_COST * (1.0 + a)
    return 0.6 * c_total / n + 0.4 * c_total * (n ** _LEARN_EXP)


def operator_cost(f1, f3, dv, m_dry=None, depot_activation=None):
    """Operator cost C [$]: servicer launch + xenon + manufacturing + depot.

    Depot activation is a one-time cost (default $20 M), not per vehicle.
    """
    m = M_DRY if m_dry is None else float(m_dry)
    n = max(1, int(round(float(f3))))
    dv = float(dv)
    if dv <= 0:
        return float("nan")
    m_prop = m * (math.exp(dv / VE_XE) - 1.0)
    m_wet  = m + m_prop
    depot = DEPOT_ACTIVATION if depot_activation is None else float(depot_activation)
    return (n * m_wet * MARKET_RATE
            + (float(f1) / dv) * m_prop * XE_COST
            + n * sscm_unit_cost(n)
            + depot)


def dv_cost(f1, dv=None, m_dry=None):
    """Xenon / ΔV cost [$] for total ΔV f1 at tank dv. Not architecture."""
    m = M_DRY if m_dry is None else float(m_dry)
    dv = DV_BUDGET if dv is None else float(dv)
    if dv <= 0 or not np.isfinite(f1):
        return float("nan")
    m_prop = m * (math.exp(dv / VE_XE) - 1.0)
    return (float(f1) / dv) * m_prop * XE_COST


def implied_contract(recovered, unserved, cost):
    """Mission-epoch contract F* and BCR* that equalize client and operator.

    Returns (F_star, bcr_star). If recovered is 0, F*=0 and BCR*=0.
    BCR* >= 1 iff recovered >= cost + unserved: then F* is a fee at which
    both BCRs are at least 1. If BCR* < 1, no such fee exists.
    """
    rec = max(0.0, float(recovered))
    uns = max(0.0, float(unserved))
    C   = float(cost)
    if not np.isfinite(C) or C <= 0:
        return 0.0, float("nan")
    if rec <= 0:
        return 0.0, 0.0
    disc = uns * uns + 4.0 * rec * C
    F = 0.5 * (-uns + math.sqrt(max(0.0, disc)))
    return F, F / C


def as_front(arr):
    """(n, k) array; transpose Julia HDF5 (k, n) layout if needed."""
    data = np.asarray(arr, dtype=float)
    if data.ndim == 2 and data.shape[0] < data.shape[1] and data.shape[0] <= 12:
        data = data.T
    return data


def objectives3(arr):
    """(n, 3) f1/f2/f3, ignoring extra per-client columns if present."""
    data = as_front(arr)
    return data[:, :3] if data.ndim == 2 and data.shape[1] > 3 else data


def _attr_str(ds, key, default=""):
    v = ds.attrs.get(key, default)
    if v is None:
        return default
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("S", "O"):
            return ",".join(
                x.decode() if isinstance(x, bytes) else str(x) for x in v.tolist())
        return str(v)
    return str(v)


def unpack_front(ds):
    """Return dict with data, tdv, clients, tdv_k.

    data columns: f1, f2_total, f3, f2_<client>… when clients are present.
    Older 3-column files have clients=[].
    """
    data = as_front(ds[:])
    tdv = float(ds.attrs.get("total_demand_value", 0.0) or 0.0)
    raw = _attr_str(ds, "clients", "")
    clients = [c.strip() for c in raw.split(",") if c.strip()]
    if not clients:
        # Infer from f2_<name> column attr or extra columns named in "columns"
        cols = _attr_str(ds, "columns", "")
        parts = [p.strip() for p in cols.split(",") if p.strip()]
        for p in parts[3:]:
            if p.startswith("f2_"):
                clients.append(p[3:])
    tdv_k = {}
    for i, c in enumerate(clients):
        key = f"total_demand_value_{c}"
        tdv_k[c] = float(ds.attrs.get(key, 0.0) or 0.0)
    return {"data": data, "tdv": tdv, "clients": clients, "tdv_k": tdv_k}


def fee_bases(f2, tdv):
    """(recovered, unserved) from a scalar unserved value and catalog total."""
    uns = float(f2)
    rec = max(0.0, float(tdv) - uns)
    return rec, uns


def client_slice(front, row, client):
    """(recovered, unserved, tdv) for one client on one front row."""
    data = front["data"]
    clients = front["clients"]
    tdv_k = front["tdv_k"]
    if client not in clients:
        raise KeyError(client)
    uns = float(row[3 + clients.index(client)])
    tdv = float(tdv_k.get(client, 0.0))
    rec = max(0.0, tdv - uns)
    return rec, uns, tdv


def knee_row(data, f1_max=None, f2_max=None):
    """Index-minimizing knee of a front in normalized (f1, mix-f2)."""
    data = np.asarray(data, dtype=float)
    data = data[data[:, 0] < 1e6]
    if len(data) == 0:
        return None
    f1 = data[:, 0]
    f2 = data[:, 1]
    s1 = float(f1_max) if f1_max is not None else max(float(np.max(f1)), 1e-12)
    s2 = float(f2_max) if f2_max is not None else max(float(np.max(f2)), 1e-12)
    idx = int(np.argmin(np.sqrt((f1 / s1) ** 2 + (f2 / s2) ** 2)))
    return data[idx]


def _empty_contract():
    return {
        "F": float("nan"), "bcr": float("nan"),
        "f1": float("nan"), "f3": float("nan"), "C": float("nan"),
        "recovered": float("nan"), "unserved": float("nan"),
        "viable": False,
    }


def _front_rows(front, data=None):
    rows = front["data"] if data is None else data
    rows = np.asarray(rows, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if len(rows):
        rows = rows[rows[:, 0] < 1e6]
    return rows


def _min_F_pick(front, rows, cost_fn):
    """argmin F* among BCR* >= 1. cost_fn(row) -> C [$]."""
    clients = list(front["clients"])
    if len(rows) == 0 or not clients:
        return {c: _empty_contract() for c in clients}
    best = {c: None for c in clients}
    for row in rows:
        C = cost_fn(row)
        if not np.isfinite(C) or C <= 0:
            continue
        f1, f3 = float(row[0]), float(row[2])
        for c in clients:
            rec, uns, _ = client_slice(front, row, c)
            F, bcr = implied_contract(rec, uns, C)
            if not (np.isfinite(bcr) and np.isfinite(F) and bcr >= 1.0):
                continue
            prev = best[c]
            if prev is None or F < prev[0]:
                best[c] = (F, {
                    "F": F, "bcr": bcr, "f1": f1, "f3": f3, "C": float(C),
                    "recovered": rec, "unserved": uns, "viable": True,
                })
    return {c: (best[c][1] if best[c] is not None else _empty_contract())
            for c in clients}


def min_viable_contract_at_C(front, C, data=None):
    """Same argmin-F* / BCR*>=1 pick as min_viable_contract, with C given.

    C is a scalar operator budget (depot + servicers + ops), not from
    operator_cost. Use this when manufacturing / depot $ are left free.
    """
    rows = _front_rows(front, data)
    C = float(C)
    return _min_F_pick(front, rows, lambda _row: C)


def min_crossover_deal(front, data=None, dv=None, m_dry=None):
    """Min F* at the BCR=1 crossover, and the architecture C that implies.

    Client BCR  = R / (U + F)     recovered / (replacement of unserved + fee)
    Operator BCR = F / (C + C_dv)  fee / (architecture + xenon)
    Both = 1 at F* = R - U and C = F* - C_dv.
    Among Pareto points with C > 0, pick argmin F*.
    """
    rows = _front_rows(front, data)
    clients = list(front["clients"])
    if len(rows) == 0 or not clients:
        return {c: _empty_contract() for c in clients}
    best = {c: None for c in clients}
    for row in rows:
        f1, f3 = float(row[0]), float(row[2])
        c_dv = dv_cost(f1, dv, m_dry)
        if not np.isfinite(c_dv):
            continue
        for c in clients:
            rec, uns, _ = client_slice(front, row, c)
            F = rec - uns
            C_arch = F - c_dv
            if F <= 0 or C_arch <= 0:
                continue
            prev = best[c]
            if prev is None or F < prev[0]:
                best[c] = (F, {
                    "F": F, "bcr": 1.0, "f1": f1, "f3": f3, "C": C_arch,
                    "dv_cost": c_dv,
                    "recovered": rec, "unserved": uns, "viable": True,
                })
    return {c: (best[c][1] if best[c] is not None else _empty_contract())
            for c in clients}


def min_viable_contract(front, budget=None, m_dry=None, data=None):
    """Per client: cheapest equal-BCR fee with BCR* >= 1.

    Among rows (the full front, or `data` if given — e.g. one f3 slice),
    pick argmin F* subject to BCR* >= 1. That F* is the unique fee that
    equalizes client and operator BCR; BCR* >= 1 means both are at least 1.

    Returns {client: {F, bcr, f1, f3, C, recovered, unserved, viable}}.
    If no viable row exists, viable is False and F/bcr are nan.
    """
    dv = DV_BUDGET if budget is None else float(budget)
    rows = _front_rows(front, data)
    return _min_F_pick(
        front, rows,
        lambda row: operator_cost(float(row[0]), float(row[2]), dv, m_dry))


def knee_contract(front, budget=None, m_dry=None):
    """Implied contract at the (ΔV, unserved) knee, per client.

    Scheduler compromise, not the equal-deal pick. Prefer
    `min_viable_contract` for client–operator agreement.
    """
    dv = DV_BUDGET if budget is None else float(budget)
    data = front["data"]
    row = knee_row(data)
    if row is None or not front["clients"]:
        return {}
    f1, f3 = float(row[0]), float(row[2])
    C = operator_cost(f1, f3, dv, m_dry)
    out = {}
    for c in front["clients"]:
        rec, uns, _ = client_slice(front, row, c)
        F, bcr = implied_contract(rec, uns, C)
        out[c] = {
            "F": F, "bcr": bcr, "f1": f1, "f3": f3, "C": C,
            "recovered": rec, "unserved": uns,
            "viable": bool(np.isfinite(bcr) and bcr >= 1.0),
        }
    return out
