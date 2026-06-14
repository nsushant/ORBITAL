# cost/lu_transfer.jl
# Low-thrust transfer cost using Lu's analytical model.
# Copied from Final_OOs/src/fuel_cost_calculations/Lu.jl

using NLsolve

# ── Constants ──────────────────────────────────────────────────────────────────
const J2_LT      = 1.0825267e-3
const R_E_LT     = 6378.137
const MU_LT      = 398600.4418
const SEC_PER_DAY_LT = 86400.0

# ── RAAN utilities ─────────────────────────────────────────────────────────────
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

# ── Phasing ΔV ─────────────────────────────────────────────────────────────────
function phasing(a0, p1, p2, k)
    n0 = sqrt(MU_LT / a0^3)
    T0 = 2π / n0
    δθ = mod(p2 - p1, 2π)
    Tp = k * T0 + (δθ / n0)
    ap = (MU_LT * (Tp / (2π))^2)^(1/3)
    arg = MU_LT * (2/a0 - 1/ap)
    arg < 0 && return Inf
    return 2 * abs(sqrt(arg) - sqrt(MU_LT / a0))
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
        "Delta_RAAN_t0"        => RAANf - RAAN0,
        "Delta_RAAN_Tf"        => RAANf_Tf - RAAN0_Tf,
        "Omega_dot_0"          => Omega_dot_0,
        "Omega_dot_f"          => Omega_dot_f,
        "Delta_RAAN_objective" => RAANf - RAAN0
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
    Jt = V0 * sqrt((Delta_a / (2 * a0))^2 + Delta_I^2 + (x3 / (2 * sin(I0)))^2)
    return Dict("ac" => a0 + Delta_a, "Ic" => I0 + Delta_I,
                "Delta_a" => Delta_a, "Delta_I" => Delta_I,
                "Jt" => Jt, "Ja" => Jt, "J_total" => 2Jt)
end

function solve_coasting_orbit_case2(a0, I0, af, If, Delta_RAAN_objective, Tf_sec)
    a_bar = (a0 + af) / 2.0
    V_bar = (circular_velocity(a0) + circular_velocity(af)) / 2.0
    Delta_a_total = af - a0
    Delta_I_total = If - I0
    Omega_dot_0   = raan_drift_rate(a0, I0)
    Omega_dot_f   = raan_drift_rate(af, If)

    function equations!(F, x)
        x1, x2, lam, mu_v = x
        Delta_a_xfer = x1 * a_bar;  Delta_I_xfer = x2
        Delta_a_adj  = Delta_a_total - Delta_a_xfer
        Delta_I_adj  = Delta_I_total - Delta_I_xfer
        eps = 1e-10
        Jt  = V_bar * sqrt((Delta_a_xfer/(2*a_bar))^2 + Delta_I_xfer^2 + eps)
        Ja  = V_bar * sqrt((Delta_a_adj /(2*a_bar))^2 + Delta_I_adj^2  + eps)
        Delta_Omega_dot_c = Omega_dot_0 * (-3.5 * x1 - tan(I0) * x2)
        F[1] = V_bar^2 / 4.0 * (x1/Jt + (x1 - Delta_a_total/a_bar)/Ja) -
               3.5 * lam * Omega_dot_0 * Tf_sec - mu_v
        F[2] = V_bar^2 * (x2/Jt + (x2 - Delta_I_total)/Ja) -
               lam * tan(I0) * Omega_dot_0 * Tf_sec
        F[3] = Delta_Omega_dot_c * Tf_sec - Delta_RAAN_objective -
               (Omega_dot_f - Omega_dot_0) * Tf_sec
        F[4] = mu_v
    end

    x0  = [Delta_a_total/(2*a_bar), Delta_I_total/2.0, 0.0, 0.0]
    sol = nlsolve(equations!, x0)
    x1, x2, _, _ = sol.zero
    Delta_a_transfer = x1 * a_bar
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
"""
function calculate_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days)
    Tf_sec    = Tf_days * SEC_PER_DAY_LT
    raan_info = process_raan_convention_A(RAAN0, RAANf, a0, I0, af, If, Tf_sec)
    tol       = 1e-6

    result = if abs(af - a0) < tol && abs(If - I0) < tol
        solve_coasting_orbit_case1(a0, I0, raan_info["Delta_RAAN_objective"], Tf_sec)
    else
        solve_coasting_orbit_case2(a0, I0, af, If, raan_info["Delta_RAAN_objective"], Tf_sec)
    end

    return Dict(
        "deltaV_total"    => result["J_total"] * 1000,
        "deltaV_transfer" => result["Jt"] * 1000,
        "deltaV_adjust"   => result["Ja"] * 1000,
        "coasting_orbit"  => Dict("a" => result["ac"], "I" => result["Ic"])
    )
end
