"""
loaders.py — data loading utilities for the OOS pymoo problem.

Functions
---------
load_sim_name_map(sim_path)   -> dict[str, int]
load_demands(demand_path)     -> dict
load_cost_table(table_path)   -> dict[tuple, float]
snap_cost(cost_table, ...)    -> float
"""

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Simulation name map
# ---------------------------------------------------------------------------

def load_sim_name_map(sim_path="outputs/simulation.h5", depot_idx=None):
    """
    Read /metadata/names from simulation.h5 and return a name → index dict
    using Julia-compatible 1-based indices that match the cost and min-TOF tables.

    Satellites named sat_N map to integer N.
    The depot index must be supplied via depot_idx (obtained from load_cost_table
    meta['depot_idx']) so the correct table-specific index is used.

    Returns
    -------
    dict[str, int]
        e.g. {"sat_1": 1, ..., "depot_1": 201}
    """
    if depot_idx is None:
        raise ValueError("depot_idx must be supplied from load_cost_table meta['depot_idx']")

    with h5py.File(sim_path, "r") as f:
        raw = f["metadata/names"][()]
    names = [n.decode() if isinstance(n, bytes) else n for n in raw]

    result = {}
    for name in names:
        parts = name.split("_")
        if parts[0] == "sat":
            result[name] = int(parts[1])
        elif parts[0] == "depot":
            result[name] = depot_idx
    return result


# ---------------------------------------------------------------------------
# Demand file loader
# ---------------------------------------------------------------------------

def load_demands(demand_path):
    """
    Read a JLD2 demand file and return its four arrays.

    Returns
    -------
    dict with keys:
        "UIDs"             : ndarray int64,   shape (Ndems,)
        "sat_identifiers"  : list[str],        length Ndems
        "service_times"    : ndarray float64,  shape (Ndems,)
        "demand_deadlines" : ndarray float64,  shape (Ndems,)
    """
    result = {}
    with h5py.File(demand_path, "r") as f:
        top_ref = f["demands"][()][0]
        kvvec = f[top_ref][()]           # array of refs, each → (key_bytes, val_ref)
        for ref in kvvec:
            pair = f[ref][()]            # structured: (key_bytes, val_ref)
            key = pair[0].decode()
            val = f[pair[1]][()]
            result[key] = val

    # decode byte strings to plain str
    if result["sat_identifiers"].dtype == object:
        result["sat_identifiers"] = [
            s.decode() if isinstance(s, bytes) else s
            for s in result["sat_identifiers"]
        ]
    else:
        result["sat_identifiers"] = list(result["sat_identifiers"])

    return result


# ---------------------------------------------------------------------------
# Cost table loader
# ---------------------------------------------------------------------------

def load_cost_table(table_path="outputs/cost_table.h5"):
    """
    Load the flat-array cost table exported by export_cost_table.jl.

    Returns
    -------
    cost_dict : dict[(int, int, float, float), float]
    meta      : dict with:
                  'depot_idx' — highest node index (the depot)
                  'dep_grid'  — sorted unique departure times
                  'arr_grid'  — sorted unique arrival times
    """
    with h5py.File(table_path, "r") as f:
        from_v = f["from_idx"][()]
        to_v   = f["to_idx"][()]
        dep_v  = f["dep"][()]
        arr_v  = f["arr"][()]
        cost_v = f["cost"][()]

    cost_dict = {
        (int(fr), int(to), float(dep), float(arr)): float(c)
        for fr, to, dep, arr, c in zip(from_v, to_v, dep_v, arr_v, cost_v)
    }
    meta = {
        "depot_idx": int(max(from_v.max(), to_v.max())),
        "dep_grid":  np.array(sorted(set(dep_v.tolist())), dtype=float),
        "arr_grid":  np.array(sorted(set(arr_v.tolist())), dtype=float),
    }
    return cost_dict, meta


# ---------------------------------------------------------------------------
# Cost lookup with grid snapping
# ---------------------------------------------------------------------------

def load_min_tof_table(path="outputs/min_tof_table.jld2"):
    """
    Load the min-TOF table from its JLD2 file.

    The file stores a structured array with dtype:
      [('first', [('1', i8), ('2', i8)]), ('second', [('1', f8), ('2', f8)])]
    where 'first' = (from_idx, to_idx) and 'second' = (min_tof_days, dv_m_s).

    Returns
    -------
    dict[(int, int), (float, float)]
        {(from_idx, to_idx): (min_tof, dv)}
    """
    with h5py.File(path, "r") as f:
        top_ref = f["MinTOFTable"][()][0]
        arr = f[top_ref][()]

    result = {}
    for row in arr:
        from_idx = int(row["first"]["1"])
        to_idx   = int(row["first"]["2"])
        min_tof  = float(row["second"]["1"])
        dv       = float(row["second"]["2"])
        result[(from_idx, to_idx)] = (min_tof, dv)
    return result


def snap_cost(cost_table, dep_grid, arr_grid, from_idx, to_idx, dep, arr):
    """
    Look up ΔV for a single leg, snapping dep/arr to the nearest grid point
    derived from the loaded cost table.

    Returns np.inf if the key is not in the table.
    """
    dep_s = float(dep_grid[np.searchsorted(dep_grid, dep).clip(0, len(dep_grid) - 1)])
    arr_s = float(arr_grid[np.searchsorted(arr_grid, arr).clip(0, len(arr_grid) - 1)])
    return cost_table.get((int(from_idx), int(to_idx), dep_s, arr_s), np.inf)
