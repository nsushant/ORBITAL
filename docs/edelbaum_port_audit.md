# Is oos/edelbaum.py a faithful port of cost/edelbaum_transfer.jl?

Line-by-line audit. Three categories: identical, deliberately changed (with the
reason and the measured effect), and not carried over.

## Identical

| piece | Julia | Python |
|---|---|---|
| velocity-plane map | `_eb_to_vspace` / `_eb_from_vspace` | `to_velocity_plane` / `from_velocity_plane` |
| arc delta-V | chord length in the velocity plane | same |
| burn duration, final mass | `dt = ve*(m/T)*(1-exp(-dv/ve))`, `m*exp(-dv/ve)` | same |
| RAAN accrued while thrusting | 16-point Gauss-Legendre over s in [0, dV] of `raan_rate/f(s)` | same, verified against a 200k-point trapezoid to 1e-12 |
| Gauss-Legendre nodes/weights | hardcoded from `numpy.polynomial.legendre.leggauss(16)` | `leggauss(16)` directly |
| ellipse geometry | centre, unit vector, normal, `b = sqrt(a_e^2 - c^2)` | same |
| chord parameterisation | uniform between the endpoints | same |
| J2 nodal rate form | `-1.5*J2*(Re/p)^2*n*cos(i)`, `p = a` for circular | same |
| growth cap | 4096 | 4096 |

## Deliberately changed

**1. Constants.** The Julia file uses `RE_ED = 6371.0`, Earth's *mean* radius.
J2 is defined with the *equatorial* radius, 6378.137 km, so the Julia nodal
regression rate is low by a factor `(6378.137/6371)^2 = 1.00224`, i.e. 0.22 %.
The other three differ in the last digits: `J2 1.0826e-3` against
`1.0825267e-3`, `mu 398600.44` against `398600.4418`, `g0 0.00981` against
`0.00980665`. `oos/constants.py` uses the standard values throughout, shared
with `oos/lu.py` so that the two models are compared on the same physics.

Measured effect, 99 instance geometries at 100 mN where both constant sets give
a transfer: **mean 0.79 %, max 7.80 %, signed mean -0.40 %** in delta-V, and no
change in which transfers are feasible. Not negligible - the same order as the
model-versus-model differences being measured - so the constant set has to be
stated in the paper rather than left implicit.

**2. RAAN closure is solved, not sampled.** Julia accepts a sampled drift orbit
when `|wrap(delivered - required)| <= 1e-2`. Python brackets sign changes of the
residual along the ring and bisects. Reason and measurements in
`docs/raan_closure_defect.md`: the sampled test finds 7.8 % of the transfers
that provably exist, root-finding finds 100 %. `RAAN_CLOSURE_TOL` no longer
exists.

**3. The search minimises over rings, not points.** Both explore the same
candidate set - the chord and ellipses with the velocity-plane endpoints as
foci. Julia evaluates sampled points and takes the cheapest, with a separate
bisection on ellipse size when the chord fails
(`_eb_find_by_bisection`). Python uses the identity that every point on a ring
costs exactly `2*(c + growth)` (verified to 2.8e-14 km/s), so cost depends only
on which ring closes, and searches for the smallest closing ring directly. Same
objective, and the Python bisection converges to `growth_rtol = 1e-3` rather
than the Julia's absolute `1e-2` on the ellipse semi-major axis.

**4. Growth ladder.** Julia: chord, then `10^linspace(-1, log10(8), 20)`, then
doubling the cap and adding `n_rings/2` rings each round to 4096. Python: chord,
then a single ladder `10^linspace(-2, log10(4096), 48)`, scanned outward, then
bisected. The Python ladder starts finer (1e-2 against 1e-1) and the bisection
makes the final resolution independent of ladder spacing.

## Not carried over

**`abs(s2.i - s1.i) > 2.0` returns NaN.** A guard in
`edelbaum_ellipse_search` and `_eb_search_tofs` rejecting inclination changes
above 2 radians (114.6 deg). It never binds for this client population, whose
largest separation is about 55 deg, and it declares very large plane changes
*infeasible* rather than merely expensive, which is not what the model means.
Left out deliberately; add it back if a population ever spans more than that.

**Candidate reuse across times of flight.** `_eb_search_tofs` computes each
drift orbit's arc quantities once and reuses them for every time of flight,
since only the coast term depends on the horizon. The Python search recomputes
per time of flight. This is a performance difference only - 19 ms per orbit pair
over 7 times of flight is fast enough for the table - and is worth restoring if
table generation becomes the bottleneck.

## Numerical gate

The audit above is by inspection. `reference_edelbaum.jl` dumps the Julia
model's delta-V on a fixed set of geometries and `compare_edelbaum_port.py`
diffs the Python against it, separating the constant-set effect from everything
else. Julia is not installed in the environment this session runs in, so that
gate has to be run on the machine that has it:

    julia --project=. reference_edelbaum.jl          # writes outputs/edelbaum_reference_julia.csv
    python3 compare_edelbaum_port.py                 # diffs against oos/edelbaum.py

Until it has been run, the port is audited but not numerically gated.
