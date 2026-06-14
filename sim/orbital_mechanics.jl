# sim/orbital_mechanics.jl
# Orbital mechanics primitives copied from Final_OOs/src/sim.jl
# Two-body + J2, RK4 integrator, Walker Delta constellation builder.

using LinearAlgebra

# ── Constants ──────────────────────────────────────────────────────────────────
const MU_SIM = 3.986004418e5   # km³/s²
const Re_SIM = 6371.0          # km
const J2_SIM = 1.08263e-3      # dimensionless

# ── Types ──────────────────────────────────────────────────────────────────────
struct OrbElem
    sma  :: Float64   # semi-major axis [km]
    ecc  :: Float64   # eccentricity
    inc  :: Float64   # inclination [rad]
    raan :: Float64   # RAAN [rad]
    aop  :: Float64   # argument of periapsis [rad]
    nu   :: Float64   # true anomaly [rad]
end

mutable struct Sat
    name :: String
    r    :: Vector{Float64}   # position [km]
    v    :: Vector{Float64}   # velocity [km/s]
end

# ── ECI from orbital elements ──────────────────────────────────────────────────
function eci_from_oelem(el::OrbElem)
    p    = el.sma * (1 - el.ecc^2)
    rmag = p / (1 + el.ecc * cos(el.nu))
    h    = sqrt(MU_SIM * p)

    r_p = [rmag * cos(el.nu), rmag * sin(el.nu), 0.0]
    v_p = [-MU_SIM/h * sin(el.nu), MU_SIM/h * (el.ecc + cos(el.nu)), 0.0]

    cΩ, sΩ = cos(el.raan), sin(el.raan)
    ci, si = cos(el.inc),  sin(el.inc)
    cω, sω = cos(el.aop),  sin(el.aop)

    R = [ cΩ*cω - sΩ*sω*ci    -cΩ*sω - sΩ*cω*ci    sΩ*si ;
          sΩ*cω + cΩ*sω*ci    -sΩ*sω + cΩ*cω*ci   -cΩ*si ;
          sω*si                 cω*si                ci    ]

    return R * r_p, R * v_p
end

# ── Orbital elements from r, v ─────────────────────────────────────────────────
function oelem_from_rv(r, v)
    h_vec = cross(r, v);  h = norm(h_vec)
    inc   = acos(clamp(h_vec[3] / h, -1, 1))

    n_vec = cross([0.0, 0.0, 1.0], h_vec);  n = norm(n_vec)
    raan  = n < 1e-10 ? 0.0 : atan(n_vec[2], n_vec[1])

    rn, vn = norm(r), norm(v)
    e_vec  = ((vn^2 - MU_SIM/rn) .* r .- dot(r, v) .* v) ./ MU_SIM
    ecc    = norm(e_vec)

    aop = if ecc < 1e-10 || n < 1e-10;  0.0
          elseif e_vec[3] >= 0;          acos(clamp(dot(n_vec, e_vec) / (n*ecc), -1, 1))
          else                      2π - acos(clamp(dot(n_vec, e_vec) / (n*ecc), -1, 1))
          end

    nu  = if ecc < 1e-10;  0.0
          else
              c = clamp(dot(e_vec, r) / (ecc * rn), -1, 1)
              dot(r, v) < 0 ? 2π - acos(c) : acos(c)
          end

    energy = 0.5 * vn^2 - MU_SIM / rn
    sma    = abs(energy) < 1e-10 ? Inf : -MU_SIM / (2energy)

    return OrbElem(sma, ecc, inc, raan, aop, nu)
end

# ── Acceleration: two-body + optional J2 ──────────────────────────────────────
function accel(r::Vector{Float64}, use_J2::Bool)
    rmag = norm(r)
    a    = (-MU_SIM / rmag^3) .* r

    if use_J2
        r2, z2 = rmag^2, r[3]^2
        c = -3J2_SIM * MU_SIM * Re_SIM^2 / (2 * rmag^5)
        a .+= c .* [(1 - 5z2/r2) * r[1],
                    (1 - 5z2/r2) * r[2],
                    (3 - 5z2/r2) * r[3]]
    end
    return a
end

# ── RK4 step (in-place) ────────────────────────────────────────────────────────
function rk4!(sat::Sat, dt::Float64, use_J2::Bool)
    k1r = sat.v;                   k1v = accel(sat.r, use_J2)
    k2r = sat.v .+ 0.5dt .* k1v;  k2v = accel(sat.r .+ 0.5dt .* k1r, use_J2)
    k3r = sat.v .+ 0.5dt .* k2v;  k3v = accel(sat.r .+ 0.5dt .* k2r, use_J2)
    k4r = sat.v .+    dt .* k3v;  k4v = accel(sat.r .+    dt .* k3r, use_J2)

    sat.r .+= (k1r .+ 2k2r .+ 2k3r .+ k4r) .* (dt / 6)
    sat.v .+= (k1v .+ 2k2v .+ 2k3v .+ k4v) .* (dt / 6)
end

# ── Walker Delta constellation ─────────────────────────────────────────────────
function walker_delta(num_planes::Int, total_sats::Int, phase::Int,
                      sma::Float64, inc::Float64;
                      name_offset::Int = 0)
    S    = total_sats ÷ num_planes
    sats = Sat[]
    for p in 0:num_planes-1
        raan = 2π * p / num_planes
        for s in 0:S-1
            nu      = 2π * (s + phase * p / num_planes) / S
            r, v    = eci_from_oelem(OrbElem(sma, 0.0, inc, raan, 0.0, nu))
            push!(sats, Sat("sat_$(name_offset + p*S + s)", r, v))
        end
    end
    return sats
end
