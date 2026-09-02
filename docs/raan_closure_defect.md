# The drift-orbit search misses most feasible transfers

## What the model does

A transfer is thrust arc, ballistic J2 coast, thrust arc. The drift orbit
(a_d, I_d) is free, and is chosen so that the RAAN accumulated over arc, coast
and arc closes the gap to the target. Both implementations search for it the
same way: sample points in the velocity plane — along the chord joining the two
endpoints, then on ellipses with the endpoints as foci — and keep a point if the
RAAN it delivers matches the required gap.

## The defect

Delivered RAAN is a continuous function of position along a ring, but the ring
is sampled at 30 points and a sample is accepted only if it lands within a
tolerance (0.01 rad) of exact closure. Whether any of those 30 samples happens
to land inside the tolerance is essentially luck. Where the reachable set is
wide, the delivered RAAN sweeps through several radians between adjacent
samples, so it steps straight over the solution.

Measured on 200 same-shell geometries (a in 500-600 km, dI <= 2 deg, time of
flight 120-365 days, 10 mN on 335 kg):

| | count | share |
|---|---|---|
| a closing drift orbit provably exists (residual changes sign along a ring) | 128 / 200 | 64.0 % |
| found by the sampler | 10 / 128 | 7.8 % |
| **missed although a solution exists** | **118 / 128** | **92.2 %** |
| reported feasible with no root present | 0 | — |

The sampler is strictly conservative — it never invents a transfer — but it
finds fewer than one in ten of the transfers that exist. Every miss is recorded
as an infeasible leg, which is how a cost table ends up mostly sentinel values
and how a client population ends up looking unreachable.

Worked example, a0 = 6928 km, I0 = 53.0 deg, af = 6938 km, If = 53.2 deg,
RAAN gap 0.5 rad:

| time of flight | reachable delivered RAAN [rad] | best sampled residual |
|---|---|---|
| 30 d | -2.344 .. -2.334 | 2.834 rad — genuinely infeasible |
| 60 d | -4.694 .. -4.661 | 1.089 rad — genuinely infeasible |
| 180 d | -15.639 .. -12.609 | 0.543 rad — **range spans 3.0 rad, so a root exists and was stepped over** |
| 365 d | -35.480 .. -22.950 | 0.005 rad — found, by luck |

At 30 and 60 days the whole reachable set delivers the wrong RAAN and the
transfer really is infeasible. At 180 days the reachable set sweeps 3.0 rad, so
the residual passes through zero, and the sampler still misses it.

## Fix

Root-find instead of sample. Along each ring the residual
`wrap(delivered(t) - required)` is continuous in the ring parameter t except
where it steps by 2*pi. Scan coarsely for sign changes that are not 2*pi jumps,
then bisect each bracket to convergence. That returns the exact closing drift
orbit when one exists, is independent of sampling resolution, and removes the
arbitrary tolerance entirely — closure becomes exact rather than "within
0.57 degrees". Cost is one bisection (about 40 evaluations) per bracket, against
30 evaluations per ring now, so the search stays comparably cheap while
answering the question it was supposed to answer.

The time-of-flight feasibility test (the two arcs must fit inside the horizon)
is unaffected and stays as it is.

## Fix as implemented

Two changes, in `oos/edelbaum.py`:

**1. Closure is solved for, not tested.** `ring_closes` scans the ring
parameter, brackets every sign change of the residual that is not a 2*pi wrap,
and bisects it to convergence. `RAAN_CLOSURE_TOL` is gone — there is no
tolerance left to set, because the drift orbit is found rather than stumbled on.

**2. The search minimises over rings, not points.** A ring is an ellipse with
the two velocity-plane endpoints as foci, and the two arcs are the distances
from the drift orbit to each focus. Their sum is therefore exactly
`2 * (c + growth)`, independent of where on the ring the drift orbit sits —
verified to 2.8e-14 km/s in `test_ring_delta_v_identity`. So the transfer cost
depends only on which ring closes, and the search reduces to finding the
smallest ring that carries a closing drift orbit: scan the growth ladder outward
for the first ring that closes, then bisect against the last that did not. This
is what Section 3.2.4 of the manuscript describes, and it is now what the code
does. Feasibility is deliberately not assumed monotone in growth — too small a
ring cannot buy enough nodal drift, too large a one spends so long thrusting
that the arcs no longer fit the horizon — so the ladder is scanned rather than
bisected from the outset.

## Result

On the population of the table above, the search now finds **34 of 34** of the
transfers that provably exist (100 %), against 7.8 % before.
`test_search_finds_the_transfers_that_exist` guards this and fails below 95 %.

Cost: 19 ms per orbit pair over 7 times of flight, so a 225-node table over a
7-point grid is about 0.3 core-hours.
