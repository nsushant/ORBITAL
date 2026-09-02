# cost/cost_functions.jl
# LT_cost_calculation and HT_cost_calculation.
# Both read trajectory data from sim.traj_file (HDF5) at the nearest hourly index.

include(joinpath(@__DIR__, "lu_transfer.jl"))
include(joinpath(@__DIR__, "lu_continuous.jl"))
include(joinpath(@__DIR__, "ht_transfer.jl"))
include(joinpath(@__DIR__, "edelbaum_transfer.jl"))

# ── Nearest index lookup ───────────────────────────────────────────────────────
@inline function nearest_idx(times::Vector{Float64}, t_sec::Float64) :: Int
    argmin(abs.(times .- t_sec))
end

# ── LT cost: Lu analytical model + phasing ΔV ─────────────────────────────────
"""
    LT_cost_calculation(sim, satinit, satarr, dep_days, arr_days) → (dv_m_s, Tp_days)

Low-thrust transfer cost from node `satinit` to node `satarr`.
Returns (ΔV in m/s, phasing duration in days). Phasing duration must be added
to the arrival epoch to get the true rendezvous time.
Returns (1e8, 0.0) on failure.
"""
function LT_cost_calculation(sim, satinit::Int, satarr::Int,
                              dep_days::Float64, arr_days::Float64;
                              continuous::Bool=false, fmax::Float64=3.5e-6,
                              oracle::Symbol=:lu,
                              m_servicer::Float64=300.0,
                              Isp::Float64=2800.0,
                              T::Union{Float64,Nothing}=nothing) :: Tuple{Float64,Float64}

    tof_days = arr_days - dep_days
    tof_days <= 0.0 && return 1e8, 0.0

    k_dep = nearest_idx(sim.times, dep_days * 86400.0)
    k_arr = nearest_idx(sim.times, arr_days * 86400.0)

    name_i = sim.names[satinit]
    name_j = sim.names[satarr]

    a_i = inc_i = raan_i = nu_i = 0.0
    a_j = inc_j = raan_j = nu_j = 0.0

    h5open(sim.traj_file, "r") do f
        oe_i = f["$name_i/orbital_elements"][:, k_dep]  # [a, i, raan, nu]
        oe_j = f["$name_j/orbital_elements"][:, k_arr]
        a_i, inc_i, raan_i, nu_i = oe_i
        a_j, inc_j, raan_j, nu_j = oe_j
    end

    n_i         = sqrt(MU_LT / a_i^3)
    nu_i_at_arr = mod(nu_i + n_i * tof_days * 86400.0, 2π)

    # ── Same-plane shortcut: skip full LT solve, only phasing ΔV needed ───────
    # wrap_pi: RAAN comes from atan2 in (-π, π], so a raw difference would miss the
    # shortcut for co-planar pairs straddling the ±180° seam.
    same_plane = abs(inc_i - inc_j)           < 0.01 &&   # ~0.57°
                 abs(wrap_pi(raan_i - raan_j)) < 0.01

    if same_plane
        dv_phase, Tp_s = phasing(a_j, nu_i_at_arr, nu_j)
        return dv_phase * 1000.0, Tp_s / 86400.0
    end

    # ── Full LT transfer ΔV ───────────────────────────────────────────────────
    result = try
        if oracle === :edelbaum
            T_force = T === nothing ? m_servicer * fmax : T
            edelbaum_transfer_cost(a_i, inc_i, raan_i, a_j, inc_j, raan_j, tof_days;
                                   m=m_servicer, Isp=Isp, T=T_force, fmax=fmax)
        else
            calculate_transfer_cost(a_i, inc_i, raan_i, a_j, inc_j, raan_j, tof_days;
                                    continuous=continuous, fmax=fmax)
        end
    catch
        return 1e8, 0.0
    end

    dv_transfer = result["deltaV_total"]
    dv_phase, Tp_s = phasing(a_j, nu_i_at_arr, nu_j)

    return dv_transfer + dv_phase * 1000.0, Tp_s / 86400.0
end

# ── HT cost: Izzo Lambert on RK4+J2 propagated positions ──────────────────────
"""
    HT_cost_calculation(sim, satinit, satarr, dep_days, arr_days) → Float64 [m/s]

High-thrust (impulsive Lambert) transfer cost from node `satinit` to node `satarr`.
Reads (r, v) from HDF5 at the nearest hourly snapshot — no analytical J2 approximation.
Returns 9999.0 on failure.
"""
function HT_cost_calculation(sim, satinit::Int, satarr::Int,
                              dep_days::Float64, arr_days::Float64) :: Float64

    tof_s = (arr_days - dep_days) * 86400.0
    tof_s <= 0.0 && return 1e8

    k_dep = nearest_idx(sim.times, dep_days * 86400.0)
    k_arr = nearest_idx(sim.times, arr_days * 86400.0)

    name_i = sim.names[satinit]
    name_j = sim.names[satarr]

    r0 = zeros(3);  v0 = zeros(3)
    rf = zeros(3);  vf = zeros(3)

    h5open(sim.traj_file, "r") do f
        r0 .= f["$name_i/positions"][:,  k_dep]
        v0 .= f["$name_i/velocities"][:, k_dep]
        rf .= f["$name_j/positions"][:,  k_arr]
        vf .= f["$name_j/velocities"][:, k_arr]
    end

    return ht_dv(r0, v0, rf, vf, tof_s)
end
