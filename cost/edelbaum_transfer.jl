# cost/edelbaum_transfer.jl
# Port of cost/cost_calculation.py (Edelbaum velocity-plane + J2 RAAN ellipse search).
#
# Units match the Python oracle: km, s, kg. Thrust T is kg·km/s² (1 N = 1e-3).
# Returned ΔV is km/s except `edelbaum_transfer_cost`, which reports m/s like Lu.
#
# TOF for ellipse_search / edelbaum_build_cost_table is seconds.
# `edelbaum_transfer_cost` takes TOF in days (same as calculate_transfer_cost).

# ── Constants (Python values, not Lu's) ───────────────────────────────────────
const MU_ED = 398600.44
const G0_ED = 0.00981          # km/s²
const J2_ED = 1.0826e-3
const RE_ED = 6371.0

# numpy.polynomial.legendre.leggauss(16)
const GL16_NODES = (
    -9.8940093499164994e-01, -9.4457502307323260e-01, -8.6563120238783164e-01,
    -7.5540440835500311e-01, -6.1787624440264377e-01, -4.5801677765722737e-01,
    -2.8160355077925892e-01, -9.5012509837637441e-02,  9.5012509837637441e-02,
     2.8160355077925892e-01,  4.5801677765722737e-01,  6.1787624440264377e-01,
     7.5540440835500311e-01,  8.6563120238783164e-01,  9.4457502307323260e-01,
     9.8940093499164994e-01,
)
const GL16_WEIGHTS = (
     2.7152459411754055e-02,  6.2253523938647609e-02,  9.5158511682492994e-02,
     1.2462897125553395e-01,  1.4959598881657665e-01,  1.6915651939500262e-01,
     1.8260341504492361e-01,  1.8945061045506859e-01,  1.8945061045506859e-01,
     1.8260341504492361e-01,  1.6915651939500262e-01,  1.4959598881657665e-01,
     1.2462897125553395e-01,  9.5158511682492994e-02,  6.2253523938647609e-02,
     2.7152459411754055e-02,
)

struct EBState
    a::Float64
    i::Float64
    raan::Float64
    m::Float64
    Isp::Float64
    T::Float64
end

# ── Velocity plane ────────────────────────────────────────────────────────────

@inline function _eb_to_vspace(a, i, mu=MU_ED)
    v = sqrt(mu / a)
    return (v * cos(π/2 * i), v * sin(π/2 * i))
end

@inline function _eb_from_vspace(x, y, mu=MU_ED)
    a = mu / (x*x + y*y)
    i = 2 / π * atan(y, x)
    return a, i
end

@inline _eb_wrap_2pi(x) = rem(x, 2π, RoundNearest)

# Discrete (a, i) samples will not hit the RAAN identity exactly.
# ~0.57° is small vs a nodal gap and large enough for a 30-point ellipse.
const RAAN_CLOSE_TOL = 1e-2

@inline function _eb_j2_raan_rate(a, i; e=0.0)
    n = sqrt(MU_ED / a^3)
    p = a * (1 - e*e)
    return -1.5 * J2_ED * (RE_ED / p)^2 * n * cos(i)
end

function _eb_dv_lt(s1::EBState, s2::EBState)
    v1x, v1y = _eb_to_vspace(s1.a, s1.i)
    v2x, v2y = _eb_to_vspace(s2.a, s2.i)
    dv = hypot(v2x - v1x, v2y - v1y)
    ve = s1.Isp * G0_ED
    dt = ve * (s1.m / s1.T) * (1 - exp(-dv / ve))
    mnew = s1.m * exp(-dv / ve)
    return dv, v1x, v1y, v2x, v2y, mnew, dt
end

function _eb_trace(s, dv, s1::EBState, v1x, v1y, v2x, v2y)
    dv <= 0 && return (NaN, NaN, NaN)
    xs = v1x + (s / dv) * (v2x - v1x)
    ys = v1y + (s / dv) * (v2y - v1y)
    a_s = MU_ED / (xs*xs + ys*ys)
    i_s = 2 / π * atan(ys, xs)
    ve  = s1.Isp * G0_ED
    m_s = s1.m * exp(-s / ve)
    f_s = s1.T / m_s
    return a_s, i_s, f_s
end

function _eb_raan_acc(s1::EBState, v1x, v1y, v2x, v2y, dv)
    dv <= 1e-15 && return 0.0
    acc = 0.0
    @inbounds for k in 1:16
        s = dv * 0.5 * (GL16_NODES[k] + 1.0)
        w = dv * 0.5 * GL16_WEIGHTS[k]
        a_s, i_s, f_s = _eb_trace(s, dv, s1, v1x, v1y, v2x, v2y)
        acc += (_eb_j2_raan_rate(a_s, i_s) / f_s) * w
    end
    return acc
end

# ── Ellipse candidates ────────────────────────────────────────────────────────

function _eb_grow_ellipse!(pts::AbstractMatrix{Float64}, growth, v1x, v1y, v2x, v2y)
    n = size(pts, 1)
    if growth == 0.0
        @inbounds for k in 1:n
            t = n == 1 ? 0.0 : (k - 1) / (n - 1)
            pts[k, 1] = (1 - t) * v1x + t * v2x
            pts[k, 2] = (1 - t) * v1y + t * v2y
        end
        return pts
    end
    cx = 0.5 * (v1x + v2x)
    cy = 0.5 * (v1y + v2y)
    dx = v2x - v1x
    dy = v2y - v1y
    d  = hypot(dx, dy)
    # RAAN-only pairs share a velocity-plane point; Python then divides by zero.
    # Grow a circle about that point instead of a degenerate ellipse.
    if d < 1e-12
        nx, ny = 1.0, 0.0
        c = 0.0
    else
        nx = dx / d
        ny = dy / d
        c  = d / 2
    end
    vx, vy = -ny, nx
    a_e = c + growth
    b   = sqrt(max(a_e*a_e - c*c, 0.0))
    @inbounds for k in 1:n
        θ = n == 1 ? 0.0 : 2π * (k - 1) / (n - 1)
        pts[k, 1] = cx + a_e * cos(θ) * nx + b * sin(θ) * vx
        pts[k, 2] = cy + a_e * cos(θ) * ny + b * sin(θ) * vy
    end
    return pts
end

function _eb_candidates(v1x, v1y, v2x, v2y, num_points, growths)
    n_g   = length(growths)
    cands = Matrix{Float64}(undef, n_g * num_points, 2)
    ring  = Matrix{Float64}(undef, num_points, 2)
    for g in 1:n_g
        _eb_grow_ellipse!(ring, growths[g], v1x, v1y, v2x, v2y)
        off = (g - 1) * num_points
        cands[off+1:off+num_points, :] .= ring
    end
    return cands
end

# Feasible drift: leftover coast is exactly t_drift = TOF − t₁ − t₂, and
#   ΔΩ₁ + Ω̇(a_d, i_d) t_drift + ΔΩ₂  ≡  ΔΩ   (mod 2π).
# (a_d, i_d) already sets the rotation sense; no extra sign filter.
function _eb_raan_closed(acc1, acc2, r, t_drift, draan; tol::Float64=RAAN_CLOSE_TOL)
    t_drift < 0 && return false
    !isfinite(r) && return false
    abs(_eb_wrap_2pi(acc1 + r * t_drift + acc2 - draan)) <= tol
end

# ── Single-point evaluation (chord / bisection path) ──────────────────────────

function _eb_eval_point(px, py, draan, s1::EBState, s2::EBState, tof)
    a_d, i_d = _eb_from_vspace(px, py)

    drift = EBState(a_d, i_d, s1.raan, s1.m, s1.Isp, s1.T)
    dv1, v1x, v1y, vdx, vdy, m_drift, dt1 = _eb_dv_lt(s1, drift)
    acc1 = _eb_raan_acc(s1, v1x, v1y, vdx, vdy, dv1)
    (tof - dt1) <= 0 && return NaN

    drift = EBState(a_d, i_d, s1.raan + acc1, m_drift, s1.Isp, s1.T)
    dv2, _, _, v2tx, v2ty, _, dt2 = _eb_dv_lt(drift, s2)
    acc2 = _eb_raan_acc(drift, vdx, vdy, v2tx, v2ty, dv2)
    t_drift = tof - dt1 - dt2
    r = _eb_j2_raan_rate(drift.a, drift.i)
    _eb_raan_closed(acc1, acc2, r, t_drift, draan) || return NaN
    return dv1 + dv2
end

function _eb_find_by_bisection(draan, s1::EBState, s2::EBState, v1x, v1y, v2x, v2y, tof, num_points)
    pts = Matrix{Float64}(undef, num_points, 2)
    a_init = 1.0
    a_infeasible = 0.0
    found = false
    while !found
        a_init *= 2.0
        a_init > 4096.0 && return NaN
        _eb_grow_ellipse!(pts, a_init, v1x, v1y, v2x, v2y)
        @inbounds for k in 1:num_points
            dv = _eb_eval_point(pts[k, 1], pts[k, 2], draan, s1, s2, tof)
            if !isnan(dv)
                found = true
                break
            end
        end
        found || (a_infeasible = a_init)
    end

    best_dv = NaN
    a_feasible = a_init
    while a_feasible - a_infeasible > 1e-2
        a_mid = 0.5 * (a_infeasible + a_feasible)
        _eb_grow_ellipse!(pts, a_mid, v1x, v1y, v2x, v2y)
        best_inner = Inf
        any_ok = false
        @inbounds for k in 1:num_points
            dv = _eb_eval_point(pts[k, 1], pts[k, 2], draan, s1, s2, tof)
            if !isnan(dv)
                any_ok = true
                dv < best_inner && (best_inner = dv)
            end
        end
        if any_ok
            a_feasible = a_mid
            best_dv = best_inner
        else
            a_infeasible = a_mid
        end
    end
    return best_dv
end

"""
    edelbaum_ellipse_search(state1, state2, tof_s; num_points=30) → ΔV [km/s]

Single-TOF search: sample drift orbits on the velocity-plane chord, then grow
an ellipse if needed. A point is feasible only if the leftover coast
`t_drift = TOF − t₁ − t₂` lands on the target node (mod 2π). Returns the
minimum ΔV among feasible points, or `NaN` if none.
"""
function edelbaum_ellipse_search(s1::EBState, s2::EBState, tof; num_points::Int=30)
    abs(s2.i - s1.i) > 2.0 && return NaN
    draan = s2.raan - s1.raan
    v1x, v1y = _eb_to_vspace(s1.a, s1.i)
    v2x, v2y = _eb_to_vspace(s2.a, s2.i)

    pts = Matrix{Float64}(undef, num_points, 2)
    _eb_grow_ellipse!(pts, 0.0, v1x, v1y, v2x, v2y)

    best = Inf
    @inbounds for k in 1:num_points
        dv = _eb_eval_point(pts[k, 1], pts[k, 2], draan, s1, s2, tof)
        !isnan(dv) && dv < best && (best = dv)
    end
    best < Inf && return best
    return _eb_find_by_bisection(draan, s1, s2, v1x, v1y, v2x, v2y, tof, num_points)
end

function edelbaum_ellipse_search(state1::AbstractVector, state2::AbstractVector, tof; num_points::Int=30)
    s1 = EBState(state1[1], state1[2], state1[3], state1[4], state1[5], state1[6])
    s2 = EBState(state2[1], state2[2], state2[3], state2[4], state2[5], state2[6])
    edelbaum_ellipse_search(s1, s2, tof; num_points=num_points)
end

# ── Batched TOF search (candidate reuse) ──────────────────────────────────────

function _eb_compute_bundle!(bundles, c, s1::EBState, s2::EBState, px, py)
    a_d, i_d = _eb_from_vspace(px, py)
    drift = EBState(a_d, i_d, s1.raan, s1.m, s1.Isp, s1.T)
    dv1, v1x, v1y, vdx, vdy, m_drift, dt1 = _eb_dv_lt(s1, drift)
    drift = EBState(a_d, i_d, s1.raan, m_drift, s1.Isp, s1.T)
    acc1  = _eb_raan_acc(s1, v1x, v1y, vdx, vdy, dv1)
    drift = EBState(a_d, i_d, s1.raan + acc1, m_drift, s1.Isp, s1.T)
    dv2, _, _, v2tx, v2ty, _, dt2 = _eb_dv_lt(drift, s2)
    acc2 = _eb_raan_acc(drift, vdx, vdy, v2tx, v2ty, dv2)
    @inbounds begin
        bundles[c, 1] = dv1 + dv2
        bundles[c, 2] = dt1
        bundles[c, 3] = dt2
        bundles[c, 4] = acc1
        bundles[c, 5] = acc2
        bundles[c, 6] = _eb_j2_raan_rate(a_d, i_d)
    end
    return nothing
end

function _eb_best_for_tof(bundles, n_cand, tof, draan)
    best = Inf
    @inbounds for c in 1:n_cand
        r      = bundles[c, 6]
        dv_tot = bundles[c, 1]
        dt1    = bundles[c, 2]
        dt2    = bundles[c, 3]
        t_drift = tof - dt1 - dt2
        _eb_raan_closed(bundles[c, 4], bundles[c, 5], r, t_drift, draan) || continue
        dv_tot < best && (best = dv_tot)
    end
    return best
end

function _eb_search_tofs(s1::EBState, s2::EBState, tofs::Vector{Float64};
                         draans::Union{Nothing,Vector{Float64}}=nothing,
                         num_points::Int=30, n_rings::Int=20)
    n_tofs  = length(tofs)
    dv_vals = fill(NaN, n_tofs)
    abs(s2.i - s1.i) > 2.0 && return dv_vals
    draans !== nothing && length(draans) != n_tofs &&
        throw(ArgumentError("draans ($(length(draans))) must match tofs ($n_tofs)"))

    draan0  = s2.raan - s1.raan
    v1x, v1y = _eb_to_vspace(s1.a, s1.i)
    v2x, v2y = _eb_to_vspace(s2.a, s2.i)

    growth_max = 8.0
    MAX_GROWTH = 4096.0

    chord = Matrix{Float64}(undef, num_points, 2)
    _eb_grow_ellipse!(chord, 0.0, v1x, v1y, v2x, v2y)
    growths   = 10 .^ range(-1.0, log10(growth_max); length=n_rings)
    rings     = _eb_candidates(v1x, v1y, v2x, v2y, num_points, growths)
    all_cands = vcat(chord, rings)

    n_cand  = size(all_cands, 1)
    bundles = Matrix{Float64}(undef, n_cand, 6)
    @inbounds for c in 1:n_cand
        _eb_compute_bundle!(bundles, c, s1, s2, all_cands[c, 1], all_cands[c, 2])
    end

    missing = trues(n_tofs)
    while growth_max <= MAX_GROWTH
        still_missing = false
        for k in 1:n_tofs
            missing[k] || continue
            dΩ   = draans === nothing ? draan0 : draans[k]
            best = _eb_best_for_tof(bundles, n_cand, tofs[k], dΩ)
            if best < Inf
                dv_vals[k]  = best
                missing[k]  = false
            else
                still_missing = true
            end
        end
        still_missing || break

        growth_max *= 2.0
        n_new_g = max(n_rings ÷ 2, 1)
        new_growths = 10 .^ range(log10(growth_max / 2.0) + 0.01,
                                  log10(growth_max); length=n_new_g)
        new_rings = _eb_candidates(v1x, v1y, v2x, v2y, num_points, new_growths)
        n_new  = size(new_rings, 1)
        old_n  = n_cand
        temp_c = Matrix{Float64}(undef, old_n + n_new, 2)
        temp_b = Matrix{Float64}(undef, old_n + n_new, 6)
        temp_c[1:old_n, :] .= all_cands
        temp_c[old_n+1:end, :] .= new_rings
        temp_b[1:old_n, :] .= bundles
        @inbounds for c in 1:n_new
            _eb_compute_bundle!(temp_b, old_n + c, s1, s2, new_rings[c, 1], new_rings[c, 2])
        end
        all_cands = temp_c
        bundles   = temp_b
        n_cand    = old_n + n_new
    end
    return dv_vals
end

"""
    edelbaum_ellipse_search_tofs(state1, state2, tofs_s; num_points=30, n_rings=20)

One orbital pair, many TOFs [s]. Drift-orbit candidates (a, i) are evaluated
once and reused. A candidate is feasible for a TOF only if leftover coast
`TOF − t₁ − t₂` closes RAAN (mod 2π). Optional `draans` is per-TOF ΔΩ [rad];
default is Ω₂ − Ω₁. Returns the minimum ΔV [km/s] among feasible candidates,
`NaN` where none exist.
"""
function edelbaum_ellipse_search_tofs(s1::EBState, s2::EBState, tofs::AbstractVector;
                                      draans=nothing, num_points::Int=30, n_rings::Int=20)
    dΩ = draans === nothing ? nothing : collect(Float64, draans)
    _eb_search_tofs(s1, s2, collect(Float64, tofs); draans=dΩ,
                    num_points=num_points, n_rings=n_rings)
end

function edelbaum_ellipse_search_tofs(state1::AbstractVector, state2::AbstractVector,
                                      tofs::AbstractVector; draans=nothing,
                                      num_points::Int=30, n_rings::Int=20)
    s1 = EBState(state1[1], state1[2], state1[3], state1[4], state1[5], state1[6])
    s2 = EBState(state2[1], state2[2], state2[3], state2[4], state2[5], state2[6])
    edelbaum_ellipse_search_tofs(s1, s2, tofs; draans=draans,
                                 num_points=num_points, n_rings=n_rings)
end

"""
    edelbaum_build_cost_table(a_i_raan1, a_i_raan2, tofs_s, m, Isp, T; num_points=30, n_rings=20)

Batched oracle. `a_i_raan1/2` are n_pairs × 3 (a [km], i [rad], RAAN [rad]).
`tofs_s` in seconds. Returns n_pairs × n_tofs ΔV [km/s].
"""
function edelbaum_build_cost_table(a_i_raan1::AbstractMatrix, a_i_raan2::AbstractMatrix,
                                   tofs, m, Isp, T; num_points::Int=30, n_rings::Int=20)
    n_pairs = size(a_i_raan1, 1)
    tofs_v  = collect(Float64, tofs)
    n_tofs  = length(tofs_v)
    out     = Matrix{Float64}(undef, n_pairs, n_tofs)
    mT, IspT, TT = Float64(m), Float64(Isp), Float64(T)

    Threads.@threads for p in 1:n_pairs
        s1 = EBState(a_i_raan1[p, 1], a_i_raan1[p, 2], a_i_raan1[p, 3], mT, IspT, TT)
        s2 = EBState(a_i_raan2[p, 1], a_i_raan2[p, 2], a_i_raan2[p, 3], mT, IspT, TT)
        out[p, :] = _eb_search_tofs(s1, s2, tofs_v; num_points=num_points, n_rings=n_rings)
    end
    return out
end

# ── Lu-compatible wrapper ─────────────────────────────────────────────────────

"""
    edelbaum_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days; m, Isp, T) → Dict

Same argument order as `calculate_transfer_cost`. Angles in rad, a in km,
TOF in days. Returns `"deltaV_total"` in m/s. Infeasible → 2e7.

Default servicer: m=300 kg, Isp=2500 s, T=10 mN (1e-5 kg·km/s²).
If `T` is omitted, T = m × fmax with fmax=3.5e-6 km/s² (~1 N, Lu paper).
"""
function edelbaum_transfer_cost(a0, I0, RAAN0, af, If, RAANf, Tf_days;
                                m=300.0, Isp=2800.0, T=nothing, fmax=3.5e-6,
                                num_points::Int=30, n_rings::Int=20)
    T_force = T === nothing ? Float64(m) * Float64(fmax) : Float64(T)
    s1 = EBState(Float64(a0), Float64(I0), Float64(RAAN0), Float64(m), Float64(Isp), T_force)
    s2 = EBState(Float64(af), Float64(If), Float64(RAANf), Float64(m), Float64(Isp), T_force)
    dv_km_s = _eb_search_tofs(s1, s2, [Float64(Tf_days) * 86400.0];
                             num_points=num_points, n_rings=n_rings)[1]
    dv_m_s  = (isnan(dv_km_s) || isinf(dv_km_s)) ? 2e7 : dv_km_s * 1000.0
    return Dict{String,Any}(
        "deltaV_total"     => dv_m_s,
        "deltaV_impulsive" => dv_m_s,
        "deltaV_transfer"  => dv_m_s,
        "deltaV_adjust"    => 0.0,
        "coasting_orbit"   => Dict("a" => NaN, "I" => NaN),
    )
end
