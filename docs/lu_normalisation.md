# Which semi-major axis normalises Eq. 44 of Lu et al.?

Section 4.2 of Lu et al. (2026) linearises the J2 nodal drift rate about a
reference orbit, giving the RAAN-closure constraint

    -3.5 * x1 - tan(I0) * x2 = R,      x1 = da / a_ref,   x2 = dI

The paper does not state whether `a_ref` is the initial semi-major axis `a0` or
the mean `(a0 + af) / 2`. The choice shifts every general (Case 2) transfer, so
it shifts every cross-shell entry of the cost table, by of order 0.3 %.

## What the published test cases say

Reproducing the paper's Table 2 under both choices, at the optimum of the
one-dimensional problem:

| case | normalisation | coasting a_c [km] | error vs Lu | Delta V [m/s] | error vs Lu |
|---|---|---|---|---|---|
| 2 | a0    | 6871.323 | -2.388 km | 549.430 | -0.27 % |
| 2 | a_bar | 6871.591 | -2.120 km | 550.925 | +0.00 % |
| 2 | *Lu*  | 6873.711 |     -     | 550.915 |    -    |
| 3 | a0    | 7147.646 | **+0.020 km** | 252.044 | +0.06 % |
| 3 | a_bar | 7150.080 | +2.454 km | 252.044 | +0.06 % |
| 3 | *Lu*  | 7147.626 |     -     | 251.900 |    -    |

Case 2 is ambiguous on its own, and in a specific way: **the paper's own Case-2
row is internally inconsistent.** Its reported change in semi-major axis,
da = -304.426 km, is the value the `a0` form produces (we get -304.475 km at the
paper's reported dI, a 0.05 km difference). Its reported dI = 2.010 deg and
Delta V = 550.915 m/s are the values the `a_bar` form produces (we get 2.0102 deg
and 550.925 m/s). No single choice reproduces both halves of that row.

Case 3 breaks the tie. There `a0` reproduces the published coasting orbit to
0.02 km while `a_bar` is 2.45 km out, and the two give an identical Delta V, so
nothing is traded away by choosing `a0`.

## Decision

`EQ44_NORMALISATION = "a0"`, on three grounds:

1. Eq. 44 is a first-order Taylor expansion of the nodal rate about the
   *initial* orbit, so the perturbation it multiplies is naturally normalised by
   `a0`. The mean semi-major axis has no role in that expansion. (`a_bar` and
   `V_bar` do appear in the Delta V expressions, Eqs. 52-53, which span both
   orbits — a different quantity, left alone.)
2. It reproduces the coasting orbit, which is what Section 4.2 actually
   computes, in both general test cases. The Delta V is a derived quantity.
3. It agrees with the paper's own reported `da` in Case 2.

The cost is that the reported Delta V agreement in Case 2 loosens from 0.00 % to
0.27 %. That remains well inside the 0.4 % already accepted for the Section 4.3
NLP, and it is the honest number.

Both forms remain selectable —
`coasting_orbit_estimate(..., eq44_normalisation="a_bar")` — so the choice can
be re-audited rather than taken on trust.

## Consequence for the paper

Appendix A (Tables A.1-A.3, "This Work" columns) was generated with the `a_bar`
form and is stale under this decision. It must be regenerated, and the
Section 3.2.6 sentence "agreeing to within 0.06 %" becomes "within 0.27 %".
