# cost/lu_transfer.jl
# Low-thrust transfer cost using Lu's analytical model.
# Copied from Final_OOs/src/fuel_cost_calculations/Lu.jl

using NLsolve

# ── Constants ──────────────────────────────────────────────────────────────────
const J2_LT      = 1.0825267e-3
const R_E_LT     = 6378.137
const MU_LT      = 398600.4418
const SEC_PER_DAY_LT = 86400.0

# Minimum admissible coasting-orbit semi-major axis (Eq. 55, constraint g2).
# Below ~200 km altitude a multi-day J2 drift is meaningless — drag dominates —
# and the unconstrained estimator will happily return sub-surface coasting orbits
# with a plausible-looking ΔV. Override per call with the `a_min` keyword.
const A_MIN_LT = R_E_LT + 200.0

# ── RAAN utilities ─────────────────────────────────────────────────────────────

# RAAN closure is only defined mod 2π. Orbital elements come back from atan2 in
# (-π, π], so an unwrapped difference can demand a 340° plane change where 20°
# would do. Every required RAAN correction is wrapped onto the short arc.
@inline wrap_pi(x) = rem(x, 2π, RoundNearest)

function raan_drift_rate(a, i)
    return -(3.0/2.0) * J2_LT * R_E_LT^2 * sqrt(MU_LT / a^7) * cos(i)
end

function propagate_raan_lt(RAAN0, a, i, dt_sec)
    return RAAN0 + raan_drift_rate(a, i) * dt_sec
end

# ── Circular orbit velocity ────────────────────────────────────────────────────
function circular_velocity(a)
    return sqrt(MU_LT / a)
end

# ── Phasing ΔV + duration ──────────────────────────────────────────────────────
# Returns (dv_km_s, phasing_duration_s).
# Finds minimum k such that |ap - a0| ≤ Δa_max (km); no arbitrary orbit cap.
# The caller adds phasing_duration_s to the arrival epoch so the cost table
# reflects the true arrival time and the scheduler can trade off ΔV vs time.
function phasing(a0, p1, p2; Δa_max=50.0)
    n0 = sqrt(MU_LT / a0^3)
    T0 = 2π / n0
    δθ = mod(p2 - p1, 2π)
    δθ < 1e-10 && return 0.0, 0.0   # already co-located
    k = 1
    while true
        Tp_orb = T0 + δθ / (n0 * k)          # period per revolution (k revs in phasing orbit)
        ap = (MU_LT * (Tp_orb / (2π))^2)^(1/3)
        abs(ap - a0) ≤ Δa_max && break
        k += 1
        k > 10000 && break                    # safety cap
    end
    Tp_orb   = T0 + δθ / (n0 * k)
    Tp_total = k * Tp_orb                     # total phasing duration (k revolutions)
    ap = (MU_LT * (Tp_orb / (2π))^2)^(1/3)
    dv = 2 * abs(sqrt(MU_LT * (2/a0 - 1/ap)) - sqrt(MU_LT / a0))
    return dv, Tp_total   # km/s, seconds
end

# ── Internal solvers ───────────────────────────────────────────────────────────
function process_raan_convention_A(RAAN0, RAANf, a0, I0, af, If, Tf_sec)
    Omega_dot_0 = raan_drift_rate(a0, I0)
    Omega_dot_f = raan_drift_rate(af, If)
    RAAN0_Tf    = propagate_raan_lt(RAAN0, a0, I0, Tf_sec)
    RAANf_Tf    = propagate_raan_lt(RAANf, af, If, Tf_sec)
    return Dict(
        "RAAN0_t0"             => RAAN0,
        "RAANf_t0"             => RAANf,
        "RAAN0_Tf"             => RAAN0_Tf,
        "RAANf_Tf"             => RAANf_Tf,
        "Delta_RAAN_t0"        => wrap_pi(RAANf - RAAN0),
        "Delta_RAAN_Tf"        => wrap_pi(RAANf_Tf - RAAN0_Tf),
        "Omega_dot_0"          => Omega_dot_0,
        "Omega_dot_f"          => Omega_dot_f,
        "Delta_RAAN_objective" => wrap_pi(RAANf - RAAN0)
    )
end

function transfer_and_adjustment_deltaV(a0, I0, af, If, Delta_a_transfer, Delta_I_transfer)
    a_bar  = (a0 + af) / 2.0
    V_bar  = (circular_velocity(a0) + circular_velocity(af)) / 2.0
    Delta_a_total = af - a0
    Delta_I_total = If - I0
    Delta_a_adjust = Delta_a_total - Delta_a_transfer
    Delta_I_adjust = Delta_I_total - Delta_I_transfer
    Jt = V_bar * sqrt((Delta_a_transfer / (2 * a_bar))^2 + Delta_I_transfer^2)
    Ja = V_bar * sqrt((Delta_a_adjust   / (2 * a_bar))^2 + Delta_I_adjust^2)
    return Dict("Jt" => Jt, "Ja" => Ja, "J_total" => Jt + Ja)
end

function solve_coasting_orbit_case1(a0, I0, Delta_RAAN_objective, Tf_sec)
    V0          = circular_velocity(a0)
    Omega_dot_0 = raan_drift_rate(a0, I0)
    term1 = 49.0/2.0
    term2 = (tan(I0)^2) / 2.0
    term3 = 2.0 / (sin(I0)^2 * (Tf_sec * Omega_dot_0)^2)
    denom = term1 + term2 + term3
    lambda = -Delta_RAAN_objective / (Tf_sec * Omega_dot_0 * denom)
    x1 = 7.0 * lambda
    x2 = (tan(I0) / 2.0) * lambda
    x3 = (-2.0 / (sin(I0)^2 * Tf_sec * Omega_dot_0)) * lambda
    Delta_a = x1 * a0
    Delta_I = x2
    Jt = V0 * sqrt((Delta_a / (2 * a0))^2 + Delta_I^2 + (x3 * sin(I0) / 2)^2)
    return Dict("ac" => a0 + Delta_a, "Ic" => I0 + Delta_I,
                "Delta_a" => Delta_a, "Delta_I" => Delta_I,
                "Jt" => Jt, "Ja" => Jt, "J_total" => 2Jt)
end

# Golden-section search: minimise a unimodal function f on the bracket [a, b].
# Derivative-free and seed-free (needs only the bracket); for a convex function it
# converges to the unique global minimum.
function golden_section_min(f, a, b; tol=1e-10, maxit=500)
    invphi  = (sqrt(5.0) - 1.0) / 2.0     # 1/φ  ≈ 0.618
    invphi2 = (3.0 - sqrt(5.0)) / 2.0     # 1/φ² ≈ 0.382
    a, b = min(a, b), max(a, b)
    h = b - a
    h <= tol && return (a + b) / 2.0
    c = a + invphi2 * h
    d = a + invphi  * h
    fc = f(c);  fd = f(d)
    for _ in 1:maxit
        if fc < fd
            b, d, fd = d, c, fc
            h *= invphi
            c = a + invphi2 * h;  fc = f(c)
        else
            a, c, fc = c, d, fd
            h *= invphi
            d = a + invphi * h;   fd = f(d)
        end
        h <= tol && break
    end
    return fc < fd ? (a + d) / 2.0 : (c + b) / 2.0
end

# Case 2 (general non-coplanar transfer). Minimises the impulsive velocity increment
# J = Jt + Ja (Eqs. 52–53) subject to the linear RAAN-closure constraint (Eq. 59).
# The constraint slaves x1 = Δa/ā to x2 = ΔI, leaving a single decision variable x2,
# found by a seed-free bracketed golden-section minimisation (J is convex in x2, so the
# bracketed minimum is the unique global optimum). The coasting-orbit semi-major-axis
# floor g2 (Eq. 55) — x1 ≥ (a_min − a0)/ā — is folded in as a one-sided bound on the
# bracket: an interior optimum is the μ = 0 solution, a boundary optimum is the
# active-floor (μ > 0) solution, so Eqs. 57–60 are honoured in a single solve.
# `a_min` is the minimum allowed coasting semi-major axis [km].
function solve_coasting_orbit_case2(a0, I0, af, If, Delta_RAAN_objective, Tf_sec; a_min=A_MIN_LT)
    a_bar         = (a0 + af) / 2.0
    Delta_a_total = af - a0
    Delta_I_total = If - I0
    Omega_dot_0   = raan_drift_rate(a0, I0)
    Omega_dot_f   = raan_drift_rate(af, If)

    infeasible() = Dict("ac" => a0, "Ic" => I0,
        "Delta_a_transfer" => 0.0, "Delta_I_transfer" => 0.0,
        "Delta_a_adjust"   => Delta_a_total, "Delta_I_adjust" => Delta_I_total,
        "Jt" => 1e4, "Ja" => 1e4, "J_total" => 2e4)

    # Near-polar orbits have ~zero J2 RAAN drift; the coasting-orbit method degenerates.
    abs(Omega_dot_0) < 1e-14 && return infeasible()

    # RAAN-closure constant R (Eq. 59): -3.5*x1 - tan(I0)*x2 = R,  x1 = Δa/a0, x2 = ΔI.
    # Δa is normalised by a0, not ā: Eq. 44 is a Taylor expansion of Ω̇ about the
    # *initial* orbit. (The ΔV expressions, Eqs. 52–53, do use ā — that is a
    # different quantity and is left alone.) The total correction the maneuver must
    # supply is wrapped onto the short arc before dividing through.
    R = wrap_pi(Delta_RAAN_objective + (Omega_dot_f - Omega_dot_0) * Tf_sec) /
        (Omega_dot_0 * Tf_sec)
    Delta_a_of(x2) = -a0 * (R + tan(I0) * x2) / 3.5

    # Objective J(x2): Δa is slaved to x2 through the constraint (Eqs. 52–53).
    J_of(x2) = transfer_and_adjustment_deltaV(a0, I0, af, If, Delta_a_of(x2), x2)["J_total"]

    # Generous physical bracket for x2 = ΔI (total inclination change ± 90°).
    pad = deg2rad(90.0)
    lo  = min(0.0, Delta_I_total) - pad
    hi  = max(0.0, Delta_I_total) + pad

    # Altitude floor g2 (Eq. 55): Δa ≥ a_min − a0  →  one-sided bound on x2.
    tanI = tan(I0)
    if abs(tanI) > 1e-12
        x2_floor = (-3.5 * (a_min - a0) / a0 - R) / tanI
        if tanI > 0            # Δa decreases as x2 increases  ⇒  x2 ≤ x2_floor
            hi = min(hi, x2_floor)
        else                    # Δa increases as x2 increases  ⇒  x2 ≥ x2_floor
            lo = max(lo, x2_floor)
        end
    end
    lo >= hi && return infeasible()

    # Minimise; also test endpoints so an active-floor (boundary) optimum is exact.
    x2c = golden_section_min(J_of, lo, hi)
    x2  = reduce((p, q) -> J_of(p) <= J_of(q) ? p : q, (x2c, lo, hi))

    Delta_a_transfer = Delta_a_of(x2)
    Delta_I_transfer = x2
    dv = transfer_and_adjustment_deltaV(a0, I0, af, If, Delta_a_transfer, Delta_I_transfer)
    return Dict("ac" => a0 + Delta_a_transfer, "Ic" => I0 + Delta_I_transfer,
                "Delta_a_transfer" => Delta_a_transfer, "Delta_I_transfer" => Delta_I_transfer,
                "Delta_a_adjust"   => Delta_a_total - Delta_a_transfer,
                "Delta_I_adjust"   => Delta_I_total - Delta_I_transfer,
                "Jt" => dv["Jt"], "Ja" => dv["Ja"], "J_total" => dv["J_total"])
end

# ── Public entry point ─────────────────────────────────────────────────────────
"""
    calculate_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days) → Dict

Low-thrust transfer cost between circular orbits (Lu analytical model).
Returns Dict with `"deltaV_total"` [m/s].

The RAAN gap is taken on the short arc (mod 2π), and the coasting orbit is held
at or above `a_min` (Eq. 55); transfers that cannot close RAAN within that floor
come back at the 2e7 m/s infeasible sentinel rather than as a cheap sub-surface
solution.
"""
function calculate_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days;
                                 a_min=A_MIN_LT, continuous=false, fmax=3.5e-6, dv_cap=10000.0)
    Tf_sec    = Tf_days * SEC_PER_DAY_LT
    raan_info = process_raan_convention_A(RAAN0, RAANf, a0, I0, af, If, Tf_sec)
    tol       = 1e-6

    result = if abs(af - a0) < tol && abs(If - I0) < tol
        solve_coasting_orbit_case1(a0, I0, raan_info["Delta_RAAN_objective"], Tf_sec)
    else
        solve_coasting_orbit_case2(a0, I0, af, If, raan_info["Delta_RAAN_objective"], Tf_sec; a_min=a_min)
    end

    # Validity guard. Case 2 clamps to the floor via Eq. 55, but Eq. 51 (Case 1) is a
    # closed form with no such constraint and will return a sub-surface coasting orbit
    # at short TOF — and Case 1 is the common path here, since a Walker shell differs
    # only in RAAN. Either way, an estimate that relies on drifting below `a_min` is
    # outside the model's domain, so report it as unreachable, not as a cheap transfer.
    if result["ac"] < a_min
        return Dict(
            "deltaV_total"     => 2e7,
            "deltaV_impulsive" => 2e7,
            "deltaV_transfer"  => 1e7,
            "deltaV_adjust"    => 1e7,
            "coasting_orbit"   => Dict("a" => result["ac"], "I" => result["Ic"])
        )
    end

    dv_impulsive = result["J_total"] * 1000            # Section 4.2 estimate [m/s]
    dv_total     = dv_impulsive
    # Only run the (expensive) 4.3 NLP when the transfer is cheap enough to matter:
    # transfers above dv_cap are never selected, so the 4.2 estimate is kept as-is.
    if continuous && dv_impulsive <= dv_cap
        dv_c = continuous_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days,
                                        result["ac"], result["Ic"]; fmax=fmax)
        dv_total = (dv_c < 1e7) ? dv_c : dv_impulsive   # fall back to 4.2 if the NLP fails
    end

    return Dict(
        "deltaV_total"     => dv_total,
        "deltaV_impulsive" => dv_impulsive,
        "deltaV_transfer"  => result["Jt"] * 1000,
        "deltaV_adjust"    => result["Ja"] * 1000,
        "coasting_orbit"   => Dict("a" => result["ac"], "I" => result["Ic"])
    )
end
