"""Denominator guards, so that no arithmetic in this package can raise.

Why this exists. The ring search reached drift orbits of 24 m, where the rocket
equation underflowed the servicer's mass to exactly zero and dividing thrust by
it raised ZeroDivisionError. Numba's default error model is Python's, so that
propagated as an exception in a serial call — and, worse, as a silently all-NaN
table inside a `parallel=True` build. A table of NaNs reads downstream as "no
transfer exists" rather than "never computed", which is how a numerical failure
becomes a result.

The policy, in two parts, because a guard alone would trade a loud failure for a
quiet wrong number:

  1. Reject first. Wherever a degenerate value means the physics has left its
     domain — a drift orbit inside the Earth or on an escape trajectory —
     the caller returns NaN before any division happens. NaN is the honest
     answer: the entry is infeasible.

  2. Guard second. `den` clamps a denominator away from zero, preserving sign,
     as a backstop for any path the first rule misses. The floors are set many
     orders of magnitude below any physically meaningful value, so no valid
     result changes; they exist only so that a missed case yields a finite
     absurd number instead of a crash, and the validity checks then catch it.

The floors are deliberately not "1e-23 for everything". A guard is only
harmless if it sits far below the smallest legitimate value of *that* quantity,
and those differ by many orders of magnitude between, say, a squared speed in
km^2/s^2 and a mass in kg.
"""

from __future__ import annotations

try:
    from numba import njit
except ImportError:  # dependency-free MVP fallback
    from .optimization.compat import njit

# Floors, each far below the smallest legitimate value of its quantity.
V2_FLOOR = 1e-30       # km^2/s^2, squared speed in the velocity plane
A_FLOOR = 1e-12        # km, semi-major axis
MASS_FLOOR = 1e-300    # kg, effective mass inside the rocket equation
R_FLOOR = 1e-12        # km, position magnitude
GENERIC_FLOOR = 1e-300


@njit(cache=True, inline="always")
def den(x, floor=GENERIC_FLOOR):
    """Clamp a denominator away from zero, preserving its sign.

    `floor` must be chosen per quantity: it is only safe if it lies far below
    anything the quantity can legitimately take.
    """
    if x >= 0.0:
        return x if x > floor else floor
    return x if x < -floor else -floor
