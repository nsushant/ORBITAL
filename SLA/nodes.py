"""Reduce the propagated ephemeris to the secular quantities the cost table needs.

The Edelbaum model of Section 3.2 is a mean-element model: it works in the
velocity plane on (a, i) and closes the node with the secular J2 rate. Feeding
it osculating elements read off an hourly snapshot imports the J2 short-period
signature, which on this population is about +/-20 km in the semi-major axis and
+/-0.02 deg in inclination — roughly 11 m/s of jitter on each velocity-plane
endpoint, for a wobble the model does not represent. So each node is reduced to

    a, i        time means over the horizon
    raan0, rate secular node, fitted by least squares to the unwrapped RAAN
    u0, urate   the same for the argument of latitude, which is the phase angle
                the phasing manoeuvre needs (see F11)

The node can be unwrapped directly — it moves about 4.5 deg/day, far below the
hourly Nyquist limit. The argument of latitude cannot: it advances about
226 deg per hour, so an hourly record is *below* Nyquist for it and a naive
unwrap aliases the rate to the wrong sign. It is therefore fitted as a small
correction on top of the analytic draconitic rate (the J2 secular
argument-of-perigee rate plus the mean-anomaly rate), which is accurate enough
that the per-record residual stays well inside half a revolution.

Fitting the rate rather than evaluating the closed form on the initial
osculating semi-major axis matters: the two differ by about 0.45 % here, which
over a 400-day horizon is several degrees of node.

Two things follow. The required RAAN change between any two epochs becomes
exact and analytic, so the cost table needs no modular wrapping of departure
epochs onto the propagated grid — the approximation the Julia made to reuse a
400-day file for 5-year missions disappears. And every entry for an object pair
shares one set of ring bundles exactly rather than approximately, which is what
makes the batched `transfer_cost` call valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from .constants import J2, MU, R_E
from .guards import A_FLOOR

A, INCL, RAAN, NU, ARGLAT = 0, 1, 2, 3, 4


@dataclass
class Nodes:
    """Secular description of every node, in km, rad and seconds."""

    names: list
    a: np.ndarray          # (N,) mean semi-major axis
    incl: np.ndarray       # (N,) mean inclination
    raan0: np.ndarray      # (N,) node at t = 0
    raan_rate: np.ndarray  # (N,) rad/s
    u0: np.ndarray         # (N,) argument of latitude at t = 0
    u_rate: np.ndarray     # (N,) rad/s
    plane: np.ndarray      # (N,) plane index; -1 if the ephemeris predates it
    horizon: float         # s

    def __len__(self):
        return len(self.names)

    def raan_at(self, idx, t):
        return self.raan0[idx] + self.raan_rate[idx] * t

    def arglat_at(self, idx, t):
        return self.u0[idx] + self.u_rate[idx] * t

    @property
    def is_depot(self):
        return np.array([n.startswith("depot") for n in self.names])


def _fit_line(t, y):
    """Least-squares (intercept, slope) of a slowly varying angle against time."""
    yy = np.unwrap(y)
    slope, intercept = np.polyfit(t, yy, 1)
    return intercept, slope


def draconitic_rate(a, incl, ecc=0.0):
    """Secular rate of the argument of latitude, argp_dot + M_dot [rad/s]."""
    a = max(float(a), A_FLOOR)
    n = np.sqrt(MU / a**3)
    p = max(a * (1.0 - ecc * ecc), A_FLOOR)
    k = 0.75 * J2 * (R_E / p) ** 2
    c2 = np.cos(incl) ** 2
    argp_dot = k * n * (5.0 * c2 - 1.0)
    m_dot = n * (1.0 + k * np.sqrt(1.0 - ecc * ecc) * (3.0 * c2 - 1.0))
    return argp_dot + m_dot


def _fit_fast_angle(t, y, rate_guess):
    """Fit a fast angle by unwrapping only its residual about a known rate.

    The record grid is below Nyquist for the angle itself, so `np.unwrap` on it
    aliases. The residual about `rate_guess` moves slowly, and unwrapping that
    is safe as long as it stays inside +/- pi between records, which is checked.
    """
    resid = (y - rate_guess * t + np.pi) % (2.0 * np.pi) - np.pi
    step = np.abs(np.diff(resid))
    step = np.minimum(step, 2.0 * np.pi - step)
    if step.max() > 0.5 * np.pi:
        raise SystemExit(
            "the analytic rate is too far off to de-alias the argument of "
            f"latitude: worst step {np.degrees(step.max()):.1f} deg between records")
    resid = np.unwrap(resid)
    slope, intercept = np.polyfit(t, resid, 1)
    return intercept, rate_guess + slope


def load_nodes(path="outputs/simulation.h5"):
    """Read the secular description the propagator accumulated.

    These are no longer fitted here. They used to be, from the hourly records,
    and that was wrong: the orbit is 1.59 h, so an hourly grid is below Nyquist
    for the J2 short-period terms and a time-average over it is biased by each
    satellite's phase. Two satellites in the same plane came out with mean
    semi-major axes 19 km apart and nodal rates differing by enough to drift 13
    degrees over the horizon, which cannot be right -- they are the same orbit.
    `oos.propagate` now accumulates the averages and the least-squares rates over
    every integration step, where nothing is aliased, and stores them alongside
    the ephemeris.
    """
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n
                 for n in f["metadata/names"][:]]
        t = f["metadata/times"][:]
        n = len(names)
        g = f["metadata"]
        missing = [k for k in ("mean_a", "mean_incl", "raan0", "raan_rate",
                               "u0", "u_rate") if k not in g]
        if missing:
            raise SystemExit(
                f"{path} predates the secular-element fix and lacks {missing}. "
                "Rebuild it with oos.propagate -- fitting these from the hourly "
                "records aliases them. See F17 in the plan.")
        planes = (g["plane_of_node"][:] if "plane_of_node" in g
                  else np.full(n, -1, dtype=np.int64))
        return Nodes(names, g["mean_a"][:], g["mean_incl"][:], g["raan0"][:],
                     g["raan_rate"][:], g["u0"][:], g["u_rate"][:], planes,
                     float(t[-1]))
