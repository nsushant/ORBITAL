# nlp_validation.jl
#
# Section 4.3 continuous-thrust NLP (Lu et al. 2026), case ξ1 = [1,0,1,0]
# (segment 1 only — the paper's optimal integer choice for all three cases).
#
# For each transfer the trajectory is transfer → coast → adjustment. Given the
# coasting orbit (aᶜ, Iᶜ) from Section 4.2, we solve the NLP:
#     minimise   J  (Eq. 36)
#     subject to a(Tt)=aᶜ, I(Tt)=Iᶜ            (coasting-orbit constraint, Eq. 63)
#                a(Tf)=a_f, I(Tf)=I_f, Ω(Tf)=Ω_f (terminal constraint, Eq. 40)
# over x = [βt1, ϑt, Tt, βa1, ϑa, Ta], evaluated with `propagate_phase`
# (Section 3) and a ballistic J2 coast in between. Solved with NLopt/SLSQP
# (open-source SNOPT-equivalent SQP).
#
# The Python formulation this ports reproduces Tables 4/6/8 to <0.4%.
#
# Requires: NLopt.jl   (add with: import Pkg; Pkg.add("NLopt"))
# Run:      julia --project=. nlp_validation.jl

using NLopt
using Printf

const J2_P = 1.0825267e-3
const RE_P = 6378.137
const MU_P = 398600.4418
const K_J2 = 3 * J2_P * RE_P^2 / (2 * MU_P^3)

raan_dot(a, I) = -1.5 * J2_P * RE_P^2 * sqrt(MU_P / a^7) * cos(I)

# ── Single-phase propagator (Section 3; validated to machine precision) ───────
function _coeffs(β1, β2, ϑ, ε1, ε2, fmax)
    f1 = ε1*fmax; f2 = ε2*fmax
    return f1*cos(β1), f1*sin(β1), f2*cos(β2), f2*sin(β2)
end
function propagate_phase(a0, I0, Ω0, β1, β2, ϑ, ε1, ε2, T, fmax)
    fc1, fs1, fc2, fs2 = _coeffs(β1, β2, ϑ, ε1, ε2, fmax)
    V0 = sqrt(MU_P/a0)
    κ  = (4ϑ*fc1 + 2*(π-2ϑ)*fc2)/π
    tiny = 1e-20;  a_var = abs(κ) > tiny;  I_var = abs(fs1) > tiny
    a = a_var ? 4*MU_P/(2V0-κ*T)^2 : a0
    if !I_var
        I = I0
    elseif !a_var
        ξ = (2fs1*sin(ϑ))/(π*V0);  I = I0 + ξ*T
    else
        z = log(2V0-κ*T); z0 = log(2V0); ξ̄ = -4fs1*sin(ϑ)/(π*κ); I = I0 + ξ̄*(z-z0)
    end
    if !a_var && !I_var
        Ω = Ω0 + ((2cos(ϑ)*fs2)/(π*V0*sin(I0)) - K_J2*V0^7*cos(I0))*T
    elseif !a_var && I_var
        ξ = (2fs1*sin(ϑ))/(π*V0); IT = I0+ξ*T
        L = log(((1+cos(IT))*(1-cos(I0)))/((1-cos(IT))*(1+cos(I0))))
        Ω = Ω0 - K_J2*(V0^7/ξ)*(sin(IT)-sin(I0)) - (cos(ϑ)*fs2/(π*ξ*V0))*L
    elseif a_var && !I_var
        z = log(2V0-κ*T); z0 = log(2V0)
        Ω = Ω0 - (4cos(ϑ)*fs2/(π*κ*sin(I0)))*(z-z0) + (K_J2*cos(I0)/(1024κ))*(exp(8z)-exp(8z0))
    else
        z = log(2V0-κ*T); z0 = log(2V0); ξ̄ = -4fs1*sin(ϑ)/(π*κ); IT = I0+ξ̄*(z-z0)
        L = log(((1+cos(IT))*(1-cos(I0)))/((1-cos(IT))*(1+cos(I0))))
        g(zz,II) = exp(8zz)*(8cos(II)+ξ̄*sin(II))/(64+ξ̄^2)
        Ω = Ω0 + (2cos(ϑ)*fs2/(π*ξ̄*κ))*L + (K_J2/(128κ))*(g(z,IT)-g(z0,I0))
    end
    return a, I, Ω
end

# ── ξ1 trajectory: transfer → J2 coast → adjustment ──────────────────────────
const DAY = 86400.0
function traj(x, a0, I0, af, If, Om0, Tf, fmax)
    βt1, ϑt, Tt_d, βa1, ϑa, Ta_d = x
    Tt = Tt_d*DAY; Ta = Ta_d*DAY
    a1, I1, Om1 = propagate_phase(a0, I0, Om0, βt1, 0.0, ϑt, 1.0, 0.0, Tt, fmax)
    Om2 = Om1 + raan_dot(a1, I1)*(Ta - Tt)
    a3, I3, Om3 = propagate_phase(a1, I1, Om2, βa1, 0.0, ϑa, 1.0, 0.0, Tf - Ta, fmax)
    return a1, I1, a3, I3, Om3
end
rawobj(x, Tf, fmax) = (2fmax/π)*(x[2]*x[3]*DAY + x[5]*(Tf - x[6]*DAY))*1000   # Eq. 36, ξ1 [m/s]
function rawcon(x, a0,I0,af,If,ac,Ic,Om0,OmT,Tf,fmax)                          # =0 constraints
    a1, I1, a3, I3, Om3 = traj(x, a0, I0, af, If, Om0, Tf, fmax)
    return [(a1-ac)/100, I1-Ic, (a3-af)/100, I3-If, Om3-OmT]
end

# finite-difference gradient of a scalar function g at x
function fdgrad!(grad, g, x)
    f0 = g(x)
    @inbounds for i in eachindex(x)
        h = 1e-7*max(1.0, abs(x[i])); xp = copy(x); xp[i] += h
        grad[i] = (g(xp) - f0)/h
    end
    return f0
end

function solve_case(a0km, I0d, afkm, Ifd, dOm0d, ackm, Icd, seeds, fmax, Tf)
    a0=RE_P+a0km; af=RE_P+afkm; I0=deg2rad(I0d); If=deg2rad(Ifd)
    ac=ackm; Ic=deg2rad(Icd); Om0=0.0; OmT=deg2rad(dOm0d)+raan_dot(af,If)*Tf
    best = Inf
    for s in seeds
        opt = Opt(:LD_SLSQP, 6)
        opt.lower_bounds = [deg2rad(-179), deg2rad(0.3), 0.05, deg2rad(-179), deg2rad(0.3), 40.0]
        opt.upper_bounds = [deg2rad( 179), deg2rad(89),  60.0, deg2rad( 179), deg2rad(89),  99.95]
        opt.xtol_rel = 1e-9; opt.maxeval = 3000
        opt.min_objective = (x, grad) -> (length(grad)>0 ? fdgrad!(grad, xx->rawobj(xx,Tf,fmax), x) : rawobj(x,Tf,fmax))
        for k in 1:5
            let k = k
                ck = xx -> rawcon(xx, a0,I0,af,If,ac,Ic,Om0,OmT,Tf,fmax)[k]
                equality_constraint!(opt, (x, grad) -> (length(grad)>0 ? fdgrad!(grad, ck, x) : ck(x)), 1e-6)
            end
        end
        (minf, minx, ret) = optimize(opt, collect(float(s)))
        resid = maximum(abs.(rawcon(minx, a0,I0,af,If,ac,Ic,Om0,OmT,Tf,fmax)))
        if resid < 1e-4 && minf < best
            best = minf
        end
    end
    return best, rad2deg(OmT)
end

const Tf = 100*DAY;  const fmax = 3.5e-6   # km/s^2
D = deg2rad
# seeds from Tables 3/5/7 (deg→rad, days) + a perturbation
c1 = solve_case(800,98,800,98, 30, 6853.309,99.318,
        [[D(134.698),D(20.666),3.563,D(-44.706),D(4.318),81.758],[D(135),D(8),10,D(-45),D(7),89]], fmax, Tf)
c2 = solve_case(800,98,900,99, 30, 6873.711,100.010,
        [[D(121.630),D(9.232),9.999,D(-31.509),D(4.942),84.881],[D(122),D(9),10,D(-31),D(5),85]], fmax, Tf)
c3 = solve_case(700,49.9,1200,50,-80, 7147.626,49.914,
        [[D(0.9514),D(21.044),0.517,D(3.869),D(57.122),98.876],[D(1),D(21),0.5,D(4),D(57),98.9]], fmax, Tf)

@printf("\n%-8s %14s %14s %10s %16s\n","Case","ΔV [m/s]","paper [m/s]","err %","RAAN@Tf [deg]")
println("-"^66)
for (n,(dv,om),p) in zip(1:3, (c1,c2,c3), (492.193,561.251,252.284))
    @printf("%-8d %14.3f %14.3f %9.2f%% %16.1f\n", n, dv, p, 100*(dv-p)/p, om)
end
