# cost/ht_transfer.jl
# Izzo Lambert solver (copied from Final_OOs/src/fuel_cost_calculations/lambertKep.jl)
# + HT_cost_calculation wrapper that reads (r,v) from the HDF5 trajectory.

using LinearAlgebra
using HDF5

const MU_HT_LOCAL    = 398600.4418   # km³/s²
const SEC_PER_DAY_HT_LOCAL = 86400.0

# ── Izzo Lambert solver ────────────────────────────────────────────────────────

function taking_derivatives(lambda::Float64, x::Float64, T::Float64)
    l2 = lambda^2;  l3 = l2 * lambda
    umx2 = 1.0 - x^2
    y = sqrt(1.0 - l2 * umx2);  y2 = y^2;  y3 = y2 * y
    DT   = 1.0/umx2 * (3T*x - 2 + 2l3*x/y)
    DDT  = 1.0/umx2 * (3T + 5x*DT + 2(1-l2)*l3/y3)
    DDDT = 1.0/umx2 * (7x*DDT + 8DT - 6(1-l2)*l2*l3*x/(y3*y2))
    return DT, DDT, DDDT
end

function tof_lagrange(x::Float64, N::Int, lambda::Float64)
    a = 1.0 / (1.0 - x^2)
    if a > 0
        alfa = 2acos(x)
        beta = 2asin(sqrt(lambda^2 / a))
        lambda < 0.0 && (beta = -beta)
        return (a*sqrt(a)*((alfa - sin(alfa)) - (beta - sin(beta)) + 2π*N)) / 2
    else
        alfa = 2acosh(x)
        beta = 2asinh(sqrt(-lambda^2 / a))
        lambda < 0.0 && (beta = -beta)
        return (-a*sqrt(-a)*((beta - sinh(beta)) - (alfa - sinh(alfa)))) / 2
    end
end

function hypergeometricF(z::Float64, tol::Float64)
    Sj = 1.0;  Cj = 1.0;  err = 1.0;  j = 0
    while err > tol
        Cj1 = Cj * (3.0+j)*(1.0+j)/(2.5+j) * z/(j+1)
        Sj += Cj1;  err = abs(Cj1);  Cj = Cj1;  j += 1
    end
    return Sj
end

function calc_tof(x::Float64, N::Int, lambda::Float64)
    battin = 0.01;  lagrange = 0.2;  dist = abs(x - 1.0)
    dist < lagrange && dist > battin && return tof_lagrange(x, N, lambda)
    K = lambda^2;  E = x^2 - 1.0;  rho = abs(E);  z = sqrt(1.0 + K*E)
    if dist < battin
        eta = z - lambda*x
        S1  = 0.5*(1.0 - lambda - x*eta)
        Q   = 4/3 * hypergeometricF(S1, 1e-11)
        return (eta^3*Q + 4lambda*eta)/2 + N*π/rho^1.5
    else
        y = sqrt(rho);  g = x*z - lambda*E
        d = E < 0 ? N*π + acos(g) : log(y*(z - lambda*x) + g)
        return (x - lambda*z - d/y) / E
    end
end

function iterate_householder(T, x0, N, eps, iter_max, lambda)
    x = x0;  err = 1.0;  it = 0
    while err > eps && it < iter_max
        tof = calc_tof(x, N, lambda)
        DT, DDT, DDDT = taking_derivatives(lambda, x, tof)
        delta = tof - T;  DT2 = DT^2
        xnew  = x - delta*(DT2 - delta*DDT/2) / (DT*(DT2 - delta*DDT) + DDDT*delta^2/6)
        err = abs(x - xnew);  x = xnew;  it += 1
    end
    return x, it
end

"""
    lambert_solver(r1, r2, tof, mu, retrograde, max_revolutions)

Solve Lambert's problem (Izzo's algorithm). Returns a Vector of [v1; v2] solutions.
"""
function lambert_solver(r1::Vector{Float64}, r2::Vector{Float64},
                        tof::Float64, mu::Float64,
                        retrograde::Int, max_revolutions::Int)
    tof <= 0 && return [fill(1e8, 6)]
    vec_c = r2 - r1;  c = norm(vec_c)
    r1n = norm(r1);  r2n = norm(r2)
    s   = (r1n + r2n + c) / 2.0
    d1  = r1/r1n;  d2 = r2/r2n
    h   = cross(d1, d2);  hn = norm(h)
    hn == 0.0 && return [fill(1e8, 6)]
    hd = h / hn
    lambda = sqrt(1.0 - c/s)
    if hd[3] < 0.0
        lambda = -lambda
        t1 = normalize(cross(d1, hd));  t2 = normalize(cross(d2, hd))
    else
        t1 = normalize(cross(hd, d1));  t2 = normalize(cross(hd, d2))
    end
    retrograde == 1 && (lambda = -lambda; t1 = -t1; t2 = -t2)
    l2 = lambda^2;  l3 = l2*lambda
    T  = sqrt(2mu / s^3) * tof
    M_max = floor(Int, T/π)
    T_00 = acos(lambda) + lambda*sqrt(1-l2)
    T_0  = T_00 + M_max*π
    T_1  = 2/3*(1 - l3)
    if M_max > 0 && T < T_0
        T_min = T_0;  x_now = 0.0
        for _ in 0:12
            DT, DDT, DDDT = taking_derivatives(lambda, x_now, T_min)
            x_next = DT != 0.0 ? x_now - DT*DDT/(DDT^2 - DT*DDDT/2) : x_now
            abs(x_now - x_next) < 1e-13 && break
            T_min = calc_tof(x_next, M_max, lambda);  x_now = x_next
        end
        T_min > T && (M_max -= 1)
    end
    M_max = min(max_revolutions, M_max)
    n_sol = 2*M_max + 1
    mx = zeros(n_sol)
    mx[1] = T >= T_00 ? -(T-T_00)/(T-T_00+4) :
            T <= T_1  ? T_1*(T_1-T)/(2/5*(1-l2*l3)*T)+1 :
                        (T/T_00)^(log(2)/log(T_1/T_00)) - 1
    mx[1], _ = iterate_householder(T, mx[1], 0, 1e-5, 15, lambda)
    for i in 1:M_max
        tmp = ((i*π+π)/(8T))^(2/3)
        mx[2i]  = (tmp-1)/(tmp+1);  mx[2i],  _ = iterate_householder(T, mx[2i],   i, 1e-8, 15, lambda)
        tmp = (8T/(i*π))^(2/3)
        mx[2i+1]= (tmp-1)/(tmp+1);  mx[2i+1],_ = iterate_householder(T, mx[2i+1], i, 1e-8, 15, lambda)
    end
    gamma = sqrt(mu*s/2);  rho = (r1n-r2n)/c;  sigma = sqrt(1-rho^2)
    sols  = Vector{Vector{Float64}}()
    for x in mx
        y_arg = 1 - l2 + l2*x^2
        y_arg < 0 && (push!(sols, fill(NaN,6)); continue)
        y   = sqrt(y_arg)
        vr1 = gamma*((lambda*y-x) - rho*(lambda*y+x))/r1n
        vr2 = -gamma*((lambda*y-x) + rho*(lambda*y+x))/r2n
        vt1 = gamma*sigma*(y+lambda*x)/r1n
        vt2 = gamma*sigma*(y+lambda*x)/r2n
        push!(sols, vcat(vr1.*d1 .+ vt1.*t1, vr2.*d2 .+ vt2.*t2))
    end
    return sols
end

get_v1(sols, k) = sols[k][1:3]
get_v2(sols, k) = sols[k][4:6]

# ── HT ΔV from two (r,v) states ───────────────────────────────────────────────
"""
    ht_dv(r0, v0_orb, rf, vf_orb, ΔT_s) → Float64 [m/s]

Minimum impulsive ΔV for Lambert transfer r0→rf in time ΔT_s [seconds].
Tries short-way and long-way, up to k_max revolutions.
Returns 9999.0 if no valid solution found.
"""
function ht_dv(r0::Vector{Float64}, v0_orb::Vector{Float64},
               rf::Vector{Float64}, vf_orb::Vector{Float64},
               ΔT_s::Float64; mu::Float64 = MU_HT_LOCAL) :: Float64
    ΔT_s <= 0.0 && return 1e8
    T_orbit = 2π * sqrt(norm(r0)^3 / mu)
    k_max   = min(5, floor(Int, ΔT_s / T_orbit))
    best    = 1e8
    for retro in (0, 1)
        try
            sols = lambert_solver(r0, rf, ΔT_s, mu, retro, k_max)
            for s in eachindex(sols)
                v0l = get_v1(sols, s);  vfl = get_v2(sols, s)
                (any(isnan, v0l) || any(isnan, vfl)) && continue
                dv = (norm(v0l .- v0_orb) + norm(vfl .- vf_orb)) * 1000.0  # km/s → m/s
                dv < best && (best = dv)
            end
        catch; end
    end
    return best
end
