# Does our Lu et al. implementation agree with the Lu et al. paper?

Checked against the source: S. Lu, L. Wang, Z. Hao, X. Li, Y. Wang, X. Lin, Y. Qi,
"Low-thrust transfer solution between circular orbits based on yaw switch
steering and analytical propagation", Acta Astronautica 245 (2026) 924-936.

Short answer: yes. Every equation we implement matches the printed form, our
coasting orbits reproduce the paper's exactly, and the one residual difference
is an inconsistency inside the paper rather than in our code.

## Eq. 44 settles the normalisation

An earlier version of this note treated the reference semi-major axis in Eq. 44
as ambiguous and inferred `a0` from Case 3. The paper states it outright:

    Delta Omega_dot_c = Omega_dot_0 * ( -3.5 * Delta a / a_0 - tan(I_0) * Delta I )    (44)

and Eq. 51 defines `x1 = Delta a / a_0`. The normalisation is **a0**, so
`EQ44_NORMALISATION = "a0"` is correct by the source, not by inference. The mean
semi-major axis `a_bar = (a0 + af)/2` and mean velocity `V_bar = (V0 + Vf)/2`
belong to the velocity-increment expressions, Eqs. 52-53, which we also follow.

The `a_bar` variant appeared to reproduce Case 2's published Delta V better.
It does not do so because it is right; it happens to compensate for an
arithmetic slip in the paper, shown below. That variant is kept selectable for
audit only.

## What reproduces exactly

| quantity | paper | ours | difference |
|---|---|---|---|
| Case 1 coasting orbit a_c | 6853.309 km | 6853.310 | +0.001 km |
| Case 2 coasting orbit a_c | 6873.711 km | 6873.711 | 0.000 km |
| Case 3 coasting orbit a_c | 7147.626 km | 7147.626 | 0.000 km |
| Case 1 Delta V (Eq. 49) | 484.664 m/s | 484.665 | +0.001 m/s |

The Section 4.2 coasting orbit - which is what Section 4.2 actually computes -
is reproduced to the last printed digit in all three cases.

**J2 nodal drift**, checked against Table 1's RAAN at t = 0 and t = 100 days:

| case | orbit | paper RAAN at 100 d | ours | difference |
|---|---|---|---|---|
| 1 | initial | 91.7 deg | 91.69 | -0.01 |
| 2 | target | 128.2 deg | 128.19 | -0.01 |
| 3 | initial | -445.7 deg | -445.73 | -0.03 |
| 3 | target | -430.3 deg | -430.29 | +0.01 |

This also pins the Earth radius. The agreement holds with the **equatorial**
radius 6378.137 km; with the mean radius 6371.0 km the drift would be low by
0.22 %, about 0.2 deg over 100 days, which this table would show. See
`docs/edelbaum_port_audit.md`, where the same constant was corrected in the
Edelbaum model.

**NLP reference values and seeds**, from Tables 3-8, all transcribed correctly:
xi_1 = [1,0,1,0] is the minimising integer choice in all three cases
(492.193 / 561.251 / 252.284 m/s), and the guidance parameters we seed from
match Tables 3, 5 and 7 exactly.

## The one difference, and it is the paper's

Substituting the paper's **own** Table 2 values for `Delta a` and `Delta I` into
the paper's **own** Eqs. 52-53:

| case | recomputed J | paper J | difference |
|---|---|---|---|
| 1 (Eq. 49) | 484.655 | 484.664 | -0.009 m/s |
| 2 (Eqs. 52-53) | **549.406** | **550.915** | **-1.509 m/s (-0.274 %)** |
| 3 (Eqs. 52-53) | 252.044 | 251.900 | +0.144 m/s (+0.057 %) |

Case 2 does not close. The inclination change implied by the paper's own
reported `J_t = 304.428 m/s` and `Delta a = -304.426 km` is **2.0152 deg**,
against the **2.010 deg** printed beside them - outside the rounding of the
printed value. Solving for the `(V_bar, a_bar)` pair that would reproduce both
`J_t` and `J_a` gives 7.43201 km/s and 7197.986 km, neither of which is any
natural combination of the case's orbits. Case 3's total agrees to 0.057 %, with
its `J_t`/`J_a` split off by about 1.2 m/s each way, which the coarse rounding of
`Delta I` to 0.014 deg accounts for.

So our Case 2 Delta V of 549.430 m/s is what Lu's equations give for Lu's
problem. The 550.915 m/s printed in the paper is not reproducible from the
inputs printed beside it.

## Consequence for the manuscript

Appendix A was generated with the `a_bar` variant and is stale. Regenerate it
from `validate_lu.py`, and state the Section 3.2.6 agreement as it is: the
coasting orbit reproduces to the last printed digit, Case 1 Delta V to
0.001 m/s, Cases 2 and 3 to 0.27 % and 0.06 %, with the Case 2 residual traced
to an internal inconsistency in the reference rather than to our implementation.
That is a stronger claim than the current "within 0.06 %", because it is
verifiable from the paper.
