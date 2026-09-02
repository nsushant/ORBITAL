# propagator_validation.jl
#
# Section 4.3 building block: the single-phase averaged-dynamics propagator
# (Lu et al. 2026, Section 3). Given a starting circular orbit and one burn
# phase's guidance (β1, β2, ϑ, ε1, ε2, duration T, f_max), it advances the mean
# elements (a, I, Ω) to the end of the phase using the closed-form solutions
# (Eqs. 18, 22–27, 28–35) with the four RAAN sub-cases.
#
# This script validates the analytical propagator against a numerical RK4
# integration of the averaged equations of motion (Eqs. 13–15) on the four
# reference configurations of Section 3.4, where the paper reports ~zero error.
#
# Run (Base Julia only, no packages):
#   julia propagator_validation.jl
#
# Units: km, s, rad; f_max in km/s^2 (paper value 3.5e-3 m/s^2 = 3.5e-6 km/s^2).

const J2_P  = 1.0825267e-3
const RE_P  = 6378.137
const MU_P  = 398600.4418
const K_J2  = 3 * J2_P * RE_P^2 / (2 * MU_P^3)   # s^6/km^7  (≈ 1.04305e-12)

# Segment thrust-acceleration components.
function _coeffs(β1, β2, ϑ, ε1, ε2, fmax)
    f1 = ε1 * fmax;  f2 = ε2 * fmax
    return f1*cos(β1), f1*sin(β1), f2*cos(β2), f2*sin(β2)   # fc1, fs1, fc2, fs2
end

# ── Analytical propagator (Section 3) ────────────────────────────────────────
"""
    propagate_phase(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax) → (a, I, Ω)

Advance mean elements over one piecewise-constant-yaw burn phase of duration T.
Angles in rad, a in km, fmax in km/s^2.
"""
function propagate_phase(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax)
    fc1, fs1, fc2, fs2 = _coeffs(β1, β2, ϑ, ε1, ε2, fmax)
    V0  = sqrt(MU_P / a0)
    κ   = (4ϑ*fc1 + 2*(π - 2ϑ)*fc2) / π          # Eq. 18 (V(t) = V0 - κt/2)
    tiny = 1e-20
    a_var = abs(κ)   > tiny
    I_var = abs(fs1) > tiny

    # a(T) — Eq. 18
    a = a_var ? 4*MU_P / (2V0 - κ*T)^2 : a0

    # I(T) — Eqs. 22–27
    if !I_var
        I = I0
    elseif !a_var
        ξ = (2fs1*sin(ϑ)) / (π*V0)               # Eq. 23
        I = I0 + ξ*T
    else
        z  = log(2V0 - κ*T);  z0 = log(2V0)
        ξ̄ = -4fs1*sin(ϑ) / (π*κ)                 # Eq. 26
        I  = I0 + ξ̄*(z - z0)                      # Eqs. 25 + 27 (sign-corrected: Ī0 = I0 − ξ̄·log(2V0))
    end

    # Ω(T) — Eqs. 28–35 (four sub-cases)
    if !a_var && !I_var                           # Eq. 28: a, I constant
        Ω = Ω0 + ((2cos(ϑ)*fs2)/(π*V0*sin(I0)) - K_J2*V0^7*cos(I0)) * T
    elseif !a_var && I_var                        # Eq. 31: a constant, I varies
        ξ  = (2fs1*sin(ϑ)) / (π*V0)
        IT = I0 + ξ*T
        L  = log(((1+cos(IT))*(1-cos(I0))) / ((1-cos(IT))*(1+cos(I0))))
        Ω  = Ω0 - K_J2*(V0^7/ξ)*(sin(IT) - sin(I0)) - (cos(ϑ)*fs2/(π*ξ*V0))*L
    elseif a_var && !I_var                        # Eq. 33: a varies, I constant
        z  = log(2V0 - κ*T);  z0 = log(2V0)
        Ω  = Ω0 - (4cos(ϑ)*fs2/(π*κ*sin(I0)))*(z - z0) +
                 (K_J2*cos(I0)/(1024κ))*(exp(8z) - exp(8z0))
    else                                          # Eq. 35: a and I both vary
        z  = log(2V0 - κ*T);  z0 = log(2V0)
        ξ̄ = -4fs1*sin(ϑ) / (π*κ)
        IT = I0 + ξ̄*(z - z0)
        L  = log(((1+cos(IT))*(1-cos(I0))) / ((1-cos(IT))*(1+cos(I0))))
        g(zz, II) = exp(8zz)*(8cos(II) + ξ̄*sin(II)) / (64 + ξ̄^2)
        Ω  = Ω0 + (2cos(ϑ)*fs2/(π*ξ̄*κ))*L + (K_J2/(128κ))*(g(z, IT) - g(z0, I0))
    end
    return a, I, Ω
end

# ── Numerical ground truth: RK4 on the averaged equations (Eqs. 13–15) ────────
function propagate_phase_numerical(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax; n=20000)
    fc1, fs1, fc2, fs2 = _coeffs(β1, β2, ϑ, ε1, ε2, fmax)
    f(y) = begin
        a, I, Ω = y
        V   = sqrt(MU_P / a)
        κ   = (4ϑ*fc1 + 2*(π - 2ϑ)*fc2) / π
        da  = sqrt(a^3/MU_P) * κ
        dI  = (2fs1*sin(ϑ)) / (π*V)
        dΩ  = (2fs2*cos(ϑ)) / (π*V*sin(I)) - K_J2*V^7*cos(I)
        (da, dI, dΩ)
    end
    y  = (a0, I0, Ω0);  dt = T/n
    add(u, v, s) = (u[1]+s*v[1], u[2]+s*v[2], u[3]+s*v[3])
    for _ in 1:n
        k1 = f(y)
        k2 = f(add(y, k1, dt/2))
        k3 = f(add(y, k2, dt/2))
        k4 = f(add(y, k3, dt))
        y  = (y[1] + dt/6*(k1[1]+2k2[1]+2k3[1]+k4[1]),
              y[2] + dt/6*(k1[2]+2k2[2]+2k3[2]+k4[2]),
              y[3] + dt/6*(k1[3]+2k2[3]+2k3[3]+k4[3]))
    end
    return y
end

# ── Validation against Section 3.4 ───────────────────────────────────────────
deg(x) = x*pi/180;  r2d(x) = x*180/pi
a0 = RE_P + 500.0;  I0 = deg(30.0);  Ω0 = deg(50.0);  fmax = 3.5e-6   # km/s^2
T  = 5*86400.0

cases = [
    ("Case 1  (a, I const)",        pi/2,  -pi/2, pi/4, 0.0, 1.0),
    ("Case 2  (a const, I varies)", pi/2,  -pi/2, pi/4, 1.0, 1.0),
    ("Case 3  (a varies, I const)", 0.0,    pi/4, pi/4, 1.0, 1.0),
    ("Case 4  (a, I both vary)",   -pi/4,   pi/4, pi/4, 1.0, 1.0),
]

using Printf
@printf("\n%-30s %12s %12s %12s\n", "case (analytic vs RK4)", "err a[km]", "err I[deg]", "err Ω[deg]")
println("-"^70)
for (name, β1, β2, ϑ, ε1, ε2) in cases
    an = propagate_phase(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax)
    nu = propagate_phase_numerical(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax)
    @printf("%-30s %12.2e %12.2e %12.2e\n", name,
            abs(an[1]-nu[1]), r2d(abs(an[2]-nu[2])), r2d(abs(an[3]-nu[3])))
end
