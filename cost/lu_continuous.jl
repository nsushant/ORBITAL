# cost/lu_continuous.jl
# Section 4.3 continuous-thrust refinement (Lu et al. 2026), case ξ1 = [1,0,1,0]
# (segment 1 only — the paper's optimal integer choice for all reported cases).
#
# Given the coasting orbit (aᶜ, Iᶜ) from Section 4.2, the trajectory
# transfer → J2 coast → adjustment is optimised with NLopt/SLSQP to minimise the
# continuous-thrust ΔV (Eq. 36) subject to the terminal constraint (Eq. 40) and
# the coasting-orbit constraint (Eq. 63). Requires NLopt.
#
# Validated (Python + Julia) to reproduce the paper's Tables 4/6/8 totals to <0.4%.
# NOTE: bounds are TOF-scaled and the RAAN constraint is wrapped mod 2π so this can
# run across a general cost table; edge cases fall back to the 4.2 estimate.
#
# Must be included AFTER lu_transfer.jl (uses J2_LT, R_E_LT, MU_LT, raan_drift_rate).

using NLopt

const K_J2_LT = 3 * J2_LT * R_E_LT^2 / (2 * MU_LT^3)   # s^6/km^7  (≈ 1.04305e-12)
const DAY_LT  = 86400.0
const FMAX_DEFAULT = 3.5e-6                            # km/s^2  (= 3.5e-3 m/s^2, paper value)

# ── Single-phase averaged-dynamics propagator (Section 3; validated) ──────────
function propagate_phase(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax)
    f1 = ε1*fmax; f2 = ε2*fmax
    fc1 = f1*cos(β1); fs1 = f1*sin(β1); fc2 = f2*cos(β2); fs2 = f2*sin(β2)
    V0 = sqrt(MU_LT/a0); κ = (4ϑ*fc1 + 2*(π-2ϑ)*fc2)/π; tiny = 1e-20
    a_var = abs(κ) > tiny; I_var = abs(fs1) > tiny
    a = a_var ? 4*MU_LT/(2V0-κ*T)^2 : a0
    if !I_var; I = I0
    elseif !a_var; ξ = (2fs1*sin(ϑ))/(π*V0); I = I0 + ξ*T
    else; z = log(2V0-κ*T); z0 = log(2V0); ξ̄ = -4fs1*sin(ϑ)/(π*κ); I = I0 + ξ̄*(z-z0); end
    if !a_var && !I_var
        Ω = Ω0 + ((2cos(ϑ)*fs2)/(π*V0*sin(I0)) - K_J2_LT*V0^7*cos(I0))*T
    elseif !a_var && I_var
        ξ = (2fs1*sin(ϑ))/(π*V0); IT = I0+ξ*T
        L = log(((1+cos(IT))*(1-cos(I0)))/((1-cos(IT))*(1+cos(I0))))
        Ω = Ω0 - K_J2_LT*(V0^7/ξ)*(sin(IT)-sin(I0)) - (cos(ϑ)*fs2/(π*ξ*V0))*L
    elseif a_var && !I_var
        z = log(2V0-κ*T); z0 = log(2V0)
        Ω = Ω0 - (4cos(ϑ)*fs2/(π*κ*sin(I0)))*(z-z0) + (K_J2_LT*cos(I0)/(1024κ))*(exp(8z)-exp(8z0))
    else
        z = log(2V0-κ*T); z0 = log(2V0); ξ̄ = -4fs1*sin(ϑ)/(π*κ); IT = I0+ξ̄*(z-z0)
        L = log(((1+cos(IT))*(1-cos(I0)))/((1-cos(IT))*(1+cos(I0))))
        g(zz,II) = exp(8zz)*(8cos(II)+ξ̄*sin(II))/(64+ξ̄^2)
        Ω = Ω0 + (2cos(ϑ)*fs2/(π*ξ̄*κ))*L + (K_J2_LT/(128κ))*(g(z,IT)-g(z0,I0))
    end
    return a, I, Ω
end

# Phase inversion (ξ1, segment 1) — produces a feasible seed at switch angle ϑ.
function _invert_phase(as, Is, at, It, ϑ, fmax)
    V0 = sqrt(MU_LT/as); Vt = sqrt(MU_LT/at); dI = It - Is
    β1 = atan(-dI*ϑ/(sin(ϑ)*log(Vt/V0)))
    (cos(β1)*(V0-Vt) < 0) && (β1 += π); β1 = mod(β1+π, 2π) - π
    T = π*(V0-Vt)/(2*ϑ*fmax*cos(β1)); return β1, T
end

function _fdgrad!(g, fn, x)
    f0 = fn(x)
    @inbounds for i in eachindex(x)
        h = 1e-7*max(1.0, abs(x[i])); xp = copy(x); xp[i] += h; g[i] = (fn(xp)-f0)/h
    end
    return f0
end

"""
    continuous_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days, ac, Ic; fmax=FMAX_DEFAULT) → ΔV [m/s]

Section 4.3 continuous-thrust ΔV (ξ1), given the coasting orbit (ac, Ic) from 4.2.
Angles in rad, a in km, `fmax` in km/s². Returns `1e8` on non-convergence (caller
should fall back to the 4.2 estimate).
"""
function continuous_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days, ac, Ic; fmax=FMAX_DEFAULT)
    Tf = Tf_days * DAY_LT
    Om0 = RAAN0
    OmT = RAANf + raan_drift_rate(af, If) * Tf         # target RAAN at Tf (consistent w/ 4.2)
    wrap(x) = rem(x, 2π, RoundNearest)                 # RAAN closure mod 2π (branch-robust)

    traj(x) = begin
        a1, I1, O1 = propagate_phase(a0, I0, Om0, x[1], 0.0, x[2], 1.0, 0.0, x[3]*DAY_LT, fmax)
        O2 = O1 + raan_drift_rate(a1, I1)*((x[6]-x[3])*DAY_LT)
        a3, I3, O3 = propagate_phase(a1, I1, O2, x[4], 0.0, x[5], 1.0, 0.0, Tf - x[6]*DAY_LT, fmax)
        (a1, I1, a3, I3, O3)
    end
    rawobj(x) = (2fmax/π)*(x[2]*x[3]*DAY_LT + x[5]*(Tf - x[6]*DAY_LT))*1000
    rawcon(x) = (t = traj(x); [(t[1]-ac)/100, t[2]-Ic, (t[3]-af)/100, t[4]-If, wrap(t[5]-OmT)])

    βt1, Tt  = _invert_phase(a0, I0, ac, Ic, deg2rad(10), fmax)
    βa1, Tad = _invert_phase(ac, Ic, af, If, deg2rad(10), fmax)
    x0 = [βt1, deg2rad(10), Tt/DAY_LT, βa1, deg2rad(10), (Tf - Tad)/DAY_LT]
    (any(!isfinite, x0) || Tt <= 0 || Tad <= 0 || (Tt+Tad) >= Tf) && return 1e8   # burns don't fit

    opt = Opt(:LD_SLSQP, 6)
    opt.lower_bounds = [deg2rad(-179), deg2rad(0.3), 0.02,          deg2rad(-179), deg2rad(0.3), 0.02]
    opt.upper_bounds = [deg2rad( 179), deg2rad(89),  Tf_days*0.95,  deg2rad( 179), deg2rad(89),  Tf_days*0.98]
    opt.xtol_rel = 1e-7; opt.maxeval = 200
    opt.min_objective = (x, g) -> (length(g)>0 ? _fdgrad!(g, rawobj, x) : rawobj(x))
    for k in 1:5
        let k = k
            ck = xx -> rawcon(xx)[k]
            equality_constraint!(opt, (x, g) -> (length(g)>0 ? _fdgrad!(g, ck, x) : ck(x)), 1e-6)
        end
    end
    # coast ≥ 0  (Ta ≥ Tt):  x[3] - x[6] ≤ 0
    inequality_constraint!(opt, (x, g) -> (length(g)>0 ? _fdgrad!(g, xx->xx[3]-xx[6], x) : x[3]-x[6]), 1e-6)

    local minf, minx, ret
    try
        (minf, minx, ret) = optimize(opt, x0)
    catch
        return 1e8
    end
    resid = maximum(abs.(rawcon(minx)))
    (resid < 1e-3 && isfinite(minf) && minf > 0) ? minf : 1e8
end
