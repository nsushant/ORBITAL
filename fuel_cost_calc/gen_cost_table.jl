
using Base.Threads
using JLD2
using ProgressMeter

include(joinpath(@__DIR__, "..", "cost", "cost_functions.jl"))


struct table_pairs
    satinit::Int
    satarr::Int
    dep::Float64
    arr::Float64
end


function build_cost_table(sim, keys; type="LT", prog=nothing, continuous::Bool=false, fmax::Float64=3.5e-6, oracle::Symbol=:lu)

    vals    = Vector{Float64}(undef, length(keys))
    tp_vals = Vector{Float64}(undef, length(keys))

    @threads for i in eachindex(keys)
      if type=="LT"
        vals[i], tp_vals[i] = LT_cost_calculation(sim, keys[i]...; continuous=continuous, fmax=fmax, oracle=oracle)
      elseif type=="HT"
        vals[i]    = HT_cost_calculation(sim, keys[i]...)
        tp_vals[i] = 0.0
      else
        dv_lt, tp_lt = LT_cost_calculation(sim, keys[i]...; continuous=continuous, fmax=fmax, oracle=oracle)
        dv_ht        = HT_cost_calculation(sim, keys[i]...)
        if dv_ht < dv_lt
            vals[i] = dv_ht; tp_vals[i] = 0.0
        else
            vals[i] = dv_lt; tp_vals[i] = tp_lt
        end
      end
      prog !== nothing && next!(prog)
    end

    return Dict(keys .=> vals), Dict(keys .=> tp_vals)
end


# ── Plane grouping ────────────────────────────────────────────────────────────
function get_plane_groups(sim, sats; raan_bin_deg=5.0, inc_bin_deg=5.0)
    raan_bin = deg2rad(raan_bin_deg)
    inc_bin  = deg2rad(inc_bin_deg)
    groups = Dict{Int, Vector{Int}}()
    h5open(sim.traj_file, "r") do f
        for idx in sats
            name = sim.names[idx]
            oe   = f["$name/orbital_elements"][:, 1]
            inc  = oe[2];  raan = oe[3]
            raan_id = floor(Int, raan / raan_bin)
            inc_id  = floor(Int, inc  / inc_bin)
            bin_id  = inc_id * 100 + raan_id
            push!(get!(groups, bin_id, Int[]), idx)
        end
    end
    return groups
end


# ── Load all OE matrices into memory (serial, one h5open) ────────────────────
# Returns Dict: sat_idx → 4×N Float64 matrix (rows: a, inc, raan, nu)
function load_oe_data(sim, sat_indices)
    oe_data = Dict{Int, Matrix{Float64}}()
    h5open(sim.traj_file, "r") do f
        for idx in sat_indices
            oe_data[idx] = read(f["$(sim.names[idx])/orbital_elements"])
        end
    end
    return oe_data
end


# Edelbaum spirals: min ΔV vs unique TOF, then copy into the (dep, arr) grid
# MDLS expects. Candidate ellipses depend only on (a, i, Ω) of the two orbits;
# every calendar slot with the same TOF reuses that ΔV.
# Returns ΔV in m/s (NaN → 1e8).
function _edelbaum_batch_spirals(oe_data, node_pairs::Vector{Tuple{Int,Int}},
                                 dep_arr_pairs, dep_to_kidx, arr_to_kidx;
                                 fmax::Float64=3.5e-6,
                                 m::Float64=300.0, Isp::Float64=2800.0,
                                 T::Union{Float64,Nothing}=nothing,
                                 desc::String="  spirals:          ",
                                 verbose::Bool=true)
    n_pairs  = length(node_pairs)
    tof_days = sort(unique(round(arr - dep; digits=6) for (dep, arr) in dep_arr_pairs))
    n_tof    = length(tof_days)
    tofs_s   = tof_days .* 86400.0
    tof_idx  = Dict(tof_days[k] => k for k in 1:n_tof)
    Tforce   = T === nothing ? m * fmax : T
    kref     = dep_to_kidx[minimum(d for (d, _) in dep_arr_pairs)]
    verbose && @info "Edelbaum TOF grid → (dep, arr) copy" n_pairs n_tof n_calendar=length(dep_arr_pairs) Isp Tforce

    pair_dvs = Vector{Vector{Float64}}(undef, n_pairs)
    p = verbose ? Progress(n_pairs; desc=desc, barlen=40, showspeed=true) : nothing
    @threads for i in eachindex(node_pairs)
        from_idx, to_idx = node_pairs[i]
        oi = oe_data[from_idx];  oj = oe_data[to_idx]
        s1 = EBState(oi[1, kref], oi[2, kref], oi[3, kref], m, Isp, Tforce)
        s2 = EBState(oj[1, kref], oj[2, kref], oj[3, kref], m, Isp, Tforce)
        dvs = _eb_search_tofs(s1, s2, tofs_s)
        out = Vector{Float64}(undef, n_tof)
        @inbounds for k in 1:n_tof
            dv = dvs[k]
            out[k] = (isnan(dv) || isinf(dv)) ? 1e8 : dv * 1000.0
        end
        pair_dvs[i] = out
        p !== nothing && next!(p)
    end

    table = Dict{Tuple{Int,Int,Float64,Float64},Float64}()
    sizehint!(table, n_pairs * length(dep_arr_pairs))
    for i in eachindex(node_pairs)
        from_idx, to_idx = node_pairs[i]
        dvs = pair_dvs[i]
        for (dep, arr) in dep_arr_pairs
            k = get(tof_idx, round(arr - dep; digits=6)) do
                argmin(abs.(tof_days .- (arr - dep)))
            end
            table[(from_idx, to_idx, dep, arr)] = dvs[k]
        end
    end
    return table
end


# ── Fast sat-to-sat table: plane-level spiral + per-sat phasing ───────────────
# Pre-loads all OE matrices into memory so @threads never touches HDF5.
function build_cost_table_fast(sim, sat_to_sat_keys, plane_groups; continuous::Bool=false, fmax::Float64=3.5e-6, oracle::Symbol=:lu,
                                Isp::Float64=2800.0, T::Union{Float64,Nothing}=nothing)

    sat_to_plane = Dict{Int,Int}()
    rep_sat      = Dict{Int,Int}()
    for (pid, ss) in plane_groups
        rep_sat[pid] = ss[1]
        for s in ss; sat_to_plane[s] = pid; end
    end

    plane_ids     = sort(collect(keys(plane_groups)))
    dep_arr_pairs = Vector{Tuple{Float64,Float64}}(unique([(k[3], k[4]) for k in sat_to_sat_keys]))

    # Pre-compute dep/arr → HDF5 column index (36 unique values, no alloc in @threads)
    dep_to_kidx = Dict(dep => nearest_idx(sim.times, dep * 86400.0) for (dep, _) in dep_arr_pairs)
    arr_to_kidx = Dict(arr => nearest_idx(sim.times, arr * 86400.0) for (_, arr) in dep_arr_pairs)

    # Load ALL satellite OEs into memory once (no HDF5 inside @threads)
    all_sat_indices = collect(keys(sat_to_plane))
    @info "Loading OE data for $(length(all_sat_indices)) sats …"
    oe_data = load_oe_data(sim, all_sat_indices)

    # Phase 1: spiral ΔV per cross-plane pair
    plane_pairs = Tuple{Int,Int}[(pi, pj) for pi in plane_ids for pj in plane_ids if pi != pj]
    spiral_table = if oracle === :edelbaum
        node_pairs = Tuple{Int,Int}[(rep_sat[pi], rep_sat[pj]) for (pi, pj) in plane_pairs]
        @info "Edelbaum sat↔sat: TOF search then copy to (dep, arr)" n_plane_pairs=length(plane_pairs)
        node_table = _edelbaum_batch_spirals(oe_data, node_pairs, dep_arr_pairs, dep_to_kidx, arr_to_kidx;
                                             fmax=fmax, Isp=Isp, T=T, desc="  sat↔sat spirals:  ")
        st = Dict{Tuple{Int,Int,Float64,Float64},Float64}()
        sizehint!(st, length(plane_pairs) * length(dep_arr_pairs))
        for (pi, pj) in plane_pairs
            ri, rj = rep_sat[pi], rep_sat[pj]
            for (dep, arr) in dep_arr_pairs
                st[(pi, pj, dep, arr)] = node_table[(ri, rj, dep, arr)]
            end
        end
        st
    else
        cross_pairs = Tuple{Int,Int,Float64,Float64}[
            (pi, pj, dep, arr)
            for pi in plane_ids, pj in plane_ids, (dep, arr) in dep_arr_pairs
            if pi != pj
        ]
        spiral_vals = Vector{Float64}(undef, length(cross_pairs))
        p1 = Progress(length(cross_pairs); desc="  sat↔sat spirals:  ", barlen=40, showspeed=true)
        @threads for i in eachindex(cross_pairs)
            pi, pj, dep, arr = cross_pairs[i]
            ki = dep_to_kidx[dep];  kj = arr_to_kidx[arr]
            oi = oe_data[rep_sat[pi]];  oj = oe_data[rep_sat[pj]]
            spiral_vals[i] = LT_spiral_only_cached(oi[1,ki], oi[2,ki], oi[3,ki],
                                                    oj[1,kj], oj[2,kj], oj[3,kj],
                                                    arr - dep; continuous=continuous, fmax=fmax, oracle=oracle)
            next!(p1)
        end
        Dict(cross_pairs .=> spiral_vals)
    end

    # Phase 2: per-satellite phasing (pure arithmetic, no HDF5)
    vals    = Vector{Float64}(undef, length(sat_to_sat_keys))
    tp_vals = Vector{Float64}(undef, length(sat_to_sat_keys))
    @info "sat↔sat phasing" n=length(sat_to_sat_keys)
    @threads for i in eachindex(sat_to_sat_keys)
        sat_i, sat_j, dep, arr = sat_to_sat_keys[i]
        ki = dep_to_kidx[dep];  kj = arr_to_kidx[arr]
        oi = oe_data[sat_i];  oj = oe_data[sat_j]

        a_i  = oi[1, ki];  nu_i = oi[4, ki]
        a_j  = oj[1, kj];  nu_j = oj[4, kj]
        n_i  = sqrt(MU_LT / a_i^3)
        tof_s       = (arr - dep) * 86400.0
        nu_i_at_arr = mod(nu_i + n_i * tof_s, 2π)
        dv_km_s, Tp_s = phasing(a_j, nu_i_at_arr, nu_j)
        dv_phase = dv_km_s * 1000.0
        Tp_days  = Tp_s / 86400.0

        pi = sat_to_plane[sat_i]
        pj = sat_to_plane[sat_j]
        if pi == pj
            vals[i]    = dv_phase
            tp_vals[i] = Tp_days
        else
            dv_spiral = get(spiral_table, (pi, pj, dep, arr), 1e8)
            vals[i]   = (dv_spiral >= 1e8 || dv_phase >= 1e8) ? 1e8 : dv_spiral + dv_phase
            tp_vals[i] = Tp_days
        end
    end

    return Dict(sat_to_sat_keys .=> vals), Dict(sat_to_sat_keys .=> tp_vals)
end


# ── LT spiral only (cached OE variant) ───────────────────────────────────────
function LT_spiral_only_cached(a_i, inc_i, raan_i, a_j, inc_j, raan_j,
                                tof_days::Float64;
                                continuous::Bool=false, fmax::Float64=3.5e-6,
                                oracle::Symbol=:lu,
                                m_servicer::Float64=300.0,
                                Isp::Float64=2800.0,
                                T::Union{Float64,Nothing}=nothing) :: Float64
    tof_days <= 0.0 && return 1e8
    same_plane = abs(inc_i - inc_j) < 0.01 && abs(wrap_pi(raan_i - raan_j)) < 0.01
    same_plane && return 0.0
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
        return 1e8
    end
    return result["deltaV_total"]
end


# ── depot→sat: NLsolve once per plane rep, phasing per individual sat ─────────
function build_cost_table_depot_to_sat_fast(sim, leg_keys, plane_groups; continuous::Bool=false, fmax::Float64=3.5e-6, oracle::Symbol=:lu, oe_data=nothing, verbose::Bool=true,
                                             Isp::Float64=2800.0, T::Union{Float64,Nothing}=nothing)
    sat_to_plane = Dict{Int,Int}()
    rep_sat      = Dict{Int,Int}()
    for (pid, ss) in plane_groups
        rep_sat[pid] = ss[1]
        for s in ss; sat_to_plane[s] = pid; end
    end

    plane_ids     = sort(collect(keys(plane_groups)))
    from_idxs     = unique(Int[k[1] for k in leg_keys])
    dep_arr_pairs = Vector{Tuple{Float64,Float64}}(unique([(k[3], k[4]) for k in leg_keys]))
    dep_to_kidx   = Dict(dep => nearest_idx(sim.times, dep * 86400.0) for (dep, _) in dep_arr_pairs)
    arr_to_kidx   = Dict(arr => nearest_idx(sim.times, arr * 86400.0) for (_, arr) in dep_arr_pairs)

    all_sat_js    = unique([k[2] for k in leg_keys])
    rep_sats_list = unique(collect(values(rep_sat)))
    needed = unique(vcat(from_idxs, rep_sats_list, all_sat_js))
    if oe_data === nothing
        verbose && @info "Loading OE data for depot→sat …"
        oe_data = load_oe_data(sim, needed)
    end

    rep_table, rep_tp_table = if oracle === :edelbaum
        node_pairs = Tuple{Int,Int}[(d, rep_sat[pid]) for d in from_idxs for pid in plane_ids]
        verbose && @info "Edelbaum depot→sat: TOF search then copy to (dep, arr)" n_pairs=length(node_pairs)
        rt = _edelbaum_batch_spirals(oe_data, node_pairs, dep_arr_pairs, dep_to_kidx, arr_to_kidx;
                                     fmax=fmax, Isp=Isp, T=T, desc="  depot→sat spirals: ", verbose=verbose)
        rtp = Dict(k => 0.0 for k in keys(rt))
        rt, rtp
    else
        from_dep_arr = unique([(k[1], k[3], k[4]) for k in leg_keys])
        rep_keys = vec(Tuple{Int,Int,Float64,Float64}[
            (from_idx, rep_sat[pid], dep, arr)
            for (from_idx, dep, arr) in from_dep_arr, pid in plane_ids
        ])
        rep_vals = Vector{Float64}(undef, length(rep_keys))
        rep_tp   = Vector{Float64}(undef, length(rep_keys))
        p1 = Progress(length(rep_keys); desc="  depot→sat spirals: ", barlen=40, showspeed=true)
        @threads for i in eachindex(rep_keys)
            rep_vals[i], rep_tp[i] = LT_cost_calculation(sim, rep_keys[i]...; continuous=continuous, fmax=fmax, oracle=oracle)
            next!(p1)
        end
        Dict(rep_keys .=> rep_vals), Dict(rep_keys .=> rep_tp)
    end

    # Phase 2: phasing from rep→sat (pure arithmetic)
    vals    = Vector{Float64}(undef, length(leg_keys))
    tp_vals = Vector{Float64}(undef, length(leg_keys))
    @threads for i in eachindex(leg_keys)
        from_idx, sat_j, dep, arr = leg_keys[i]
        pid   = sat_to_plane[sat_j]
        rep_j = rep_sat[pid]
        ki    = dep_to_kidx[dep];  kj = arr_to_kidx[arr]

        dv_rep = get(rep_table,    (from_idx, rep_j, dep, arr), 1e8)
        tp_rep = get(rep_tp_table, (from_idx, rep_j, dep, arr), 0.0)

        or = oe_data[rep_j];  oj = oe_data[sat_j]
        a_rep = or[1,ki];  nu_rep = or[4,ki]
        a_j   = oj[1,kj];  nu_j   = oj[4,kj]
        n_rep       = sqrt(MU_LT / a_rep^3)
        tof_s       = (arr - dep) * 86400.0
        nu_rep_arr  = mod(nu_rep + n_rep * tof_s, 2π)
        dv_km_s, Tp_s = phasing(a_j, nu_rep_arr, nu_j)
        dv_pha  = dv_km_s * 1000.0
        Tp_days = Tp_s / 86400.0

        vals[i]    = (dv_rep >= 1e8 || dv_pha >= 1e8) ? 1e8 : dv_rep + dv_pha
        tp_vals[i] = tp_rep + Tp_days
    end
    return Dict(leg_keys .=> vals), Dict(leg_keys .=> tp_vals)
end


# ── sat→depot: copy rep cost to all sats in same plane ───────────────────────
function build_cost_table_sat_to_depot_fast(sim, leg_keys, plane_groups; continuous::Bool=false, fmax::Float64=3.5e-6, oracle::Symbol=:lu, oe_data=nothing, verbose::Bool=true,
                                             Isp::Float64=2800.0, T::Union{Float64,Nothing}=nothing)
    sat_to_plane = Dict{Int,Int}()
    rep_sat      = Dict{Int,Int}()
    for (pid, ss) in plane_groups
        rep_sat[pid] = ss[1]
        for s in ss; sat_to_plane[s] = pid; end
    end

    plane_ids     = sort(collect(keys(plane_groups)))
    to_idxs       = unique(Int[k[2] for k in leg_keys])
    dep_arr_pairs = Vector{Tuple{Float64,Float64}}(unique([(k[3], k[4]) for k in leg_keys]))
    dep_to_kidx   = Dict(dep => nearest_idx(sim.times, dep * 86400.0) for (dep, _) in dep_arr_pairs)
    arr_to_kidx   = Dict(arr => nearest_idx(sim.times, arr * 86400.0) for (_, arr) in dep_arr_pairs)

    rep_table, rep_tp_table = if oracle === :edelbaum
        needed = unique(vcat(to_idxs, collect(values(rep_sat))))
        if oe_data === nothing
            oe_data = load_oe_data(sim, needed)
        end
        node_pairs = Tuple{Int,Int}[(rep_sat[pid], d) for d in to_idxs for pid in plane_ids]
        verbose && @info "Edelbaum sat→depot: TOF search then copy to (dep, arr)" n_pairs=length(node_pairs)
        rt = _edelbaum_batch_spirals(oe_data, node_pairs, dep_arr_pairs, dep_to_kidx, arr_to_kidx;
                                     fmax=fmax, Isp=Isp, T=T, desc="  sat→depot spirals: ", verbose=verbose)
        rtp = Dict(k => 0.0 for k in keys(rt))
        rt, rtp
    else
        to_dep_arr = unique([(k[2], k[3], k[4]) for k in leg_keys])
        rep_keys = vec(Tuple{Int,Int,Float64,Float64}[
            (rep_sat[pid], to_idx, dep, arr)
            for (to_idx, dep, arr) in to_dep_arr, pid in plane_ids
        ])
        rep_vals = Vector{Float64}(undef, length(rep_keys))
        rep_tp   = Vector{Float64}(undef, length(rep_keys))
        p1 = Progress(length(rep_keys); desc="  sat→depot spirals: ", barlen=40, showspeed=true)
        @threads for i in eachindex(rep_keys)
            rep_vals[i], rep_tp[i] = LT_cost_calculation(sim, rep_keys[i]...; continuous=continuous, fmax=fmax, oracle=oracle)
            next!(p1)
        end
        Dict(rep_keys .=> rep_vals), Dict(rep_keys .=> rep_tp)
    end

    vals    = Vector{Float64}(undef, length(leg_keys))
    tp_vals = Vector{Float64}(undef, length(leg_keys))
    @threads for i in eachindex(leg_keys)
        sat_i, to_idx, dep, arr = leg_keys[i]
        pid        = sat_to_plane[sat_i]
        vals[i]    = get(rep_table,    (rep_sat[pid], to_idx, dep, arr), 1e8)
        tp_vals[i] = get(rep_tp_table, (rep_sat[pid], to_idx, dep, arr), 0.0)
    end
    return Dict(leg_keys .=> vals), Dict(leg_keys .=> tp_vals)
end


function load_cost_table()
    path = "outputs/cost_table.jld2"
    isfile(path) || error("No cost table found at $path — run gen_cost_table() first.")
    local CostTable
    @load path CostTable
    return CostTable
end

function gen_cost_table(sim; tof_step = 30.0, t_end = 400.0, continuous::Bool=false, fmax::Float64=3.5e-6,
                        oracle::Symbol=:lu,
                        ct_path::String="outputs/cost_table.jld2",
                        pt_path::String="outputs/phasing_time_table.jld2",
                        prompt::Bool=true,
                        Isp::Float64=2800.0,
                        T::Union{Float64,Nothing}=nothing)

        if prompt && isfile(ct_path)

        println("Cost table file found at $ct_path. Load existing (y) or rebuild (n)?")
        choice = strip(readline())

          if choice == "y"

            @load ct_path CostTable
              return CostTable

          end

        end

        arrivals   = collect(tof_step:tof_step:t_end)
        departures = collect(tof_step:tof_step:t_end)

        depots  = findall(startswith("depot"),  sim.names)
        sats    = findall(startswith("sat"),    sim.names)
        debris  = findall(startswith("debris"), sim.names)

        plane_groups = get_plane_groups(sim, sats)
        @info "Plane groups" n_planes=length(plane_groups)

        has_debris = !isempty(debris)

        sat_to_sat_keys = [(satinit, satarr, dep, dep + arr)
                            for arr in arrivals
                            for dep in departures
                            for satinit in sats
                            for satarr in sats
                            if satinit != satarr]

        sat_to_depot_keys = [(sat, depot, dep, dep + arr)
                            for arr in arrivals
                            for dep in departures
                            for sat in sats
                            for depot in depots]

        depot_to_sat_keys = [(depot, sat, dep, dep + arr)
                            for arr in arrivals
                            for dep in departures
                            for sat in sats
                            for depot in depots]

        if has_debris
            sat_to_debris_keys = [(sat, deb, dep, dep + arr)
                                    for arr in arrivals for dep in departures
                                    for sat in sats for deb in debris]
            debris_to_sat_keys = [(deb, sat, dep, dep + arr)
                                    for arr in arrivals for dep in departures
                                    for sat in sats for deb in debris]
            depot_to_debris_keys = [(depot, deb, dep, dep + arr)
                                    for arr in arrivals for dep in departures
                                    for depot in depots for deb in debris]
            debris_to_depot_keys = [(deb, depot, dep, dep + arr)
                                    for arr in arrivals for dep in departures
                                    for depot in depots for deb in debris]
        end

        @info "Building cost tables" n_sat_pairs=length(sat_to_sat_keys) oracle Isp T

        CT_sat_to_sat,   PT_sat_to_sat   = build_cost_table_fast(sim, sat_to_sat_keys, plane_groups; continuous=continuous, fmax=fmax, oracle=oracle, Isp=Isp, T=T)
        CT_depot_to_sat, PT_depot_to_sat = build_cost_table_depot_to_sat_fast(sim, depot_to_sat_keys, plane_groups; continuous=continuous, fmax=fmax, oracle=oracle, Isp=Isp, T=T)
        CT_sat_to_depot, PT_sat_to_depot = build_cost_table_sat_to_depot_fast(sim, sat_to_depot_keys, plane_groups; continuous=continuous, fmax=fmax, oracle=oracle, Isp=Isp, T=T)

        if has_debris
            CT_sat_to_debris,   PT_sat_to_debris   = build_cost_table(sim, sat_to_debris_keys; continuous=continuous, fmax=fmax, oracle=oracle)
            CT_debris_to_sat,   PT_debris_to_sat   = build_cost_table(sim, debris_to_sat_keys; continuous=continuous, fmax=fmax, oracle=oracle)
            CT_depot_to_debris, PT_depot_to_debris = build_cost_table(sim, depot_to_debris_keys; continuous=continuous, fmax=fmax, oracle=oracle)
            CT_debris_to_depot, PT_debris_to_depot = build_cost_table(sim, debris_to_depot_keys; continuous=continuous, fmax=fmax, oracle=oracle)
        end

        @info "Merging cost tables …"
        CostTable        = merge(CT_depot_to_sat, CT_sat_to_depot, CT_sat_to_sat)
        PhasingTimeTable = merge(PT_depot_to_sat, PT_sat_to_depot, PT_sat_to_sat)

        if has_debris
            CostTable        = merge(CostTable, CT_sat_to_debris, CT_debris_to_sat,
                                     CT_depot_to_debris, CT_debris_to_depot)
            PhasingTimeTable = merge(PhasingTimeTable, PT_sat_to_debris, PT_debris_to_sat,
                                     PT_depot_to_debris, PT_debris_to_depot)
        end

        @info "Saving cost table …"
        @save ct_path CostTable
        @save pt_path PhasingTimeTable
        @info "Cost tables complete" ct_path pt_path oracle

        return CostTable

end

function build_min_tof_table(;
        ct_path::String="outputs/cost_table.jld2",
        pt_path::String="outputs/phasing_time_table.jld2",
        out_path::String="outputs/min_tof_table.jld2",
        force::Bool=false)
    if !force && isfile(out_path)
        local MinTOFTable
        @load out_path MinTOFTable
        return MinTOFTable
    end

    isfile(ct_path) || error("No cost table found at $ct_path — run gen_cost_table() first.")
    local CostTable
    @load ct_path CostTable

    PhasingTimeTable = if isfile(pt_path)
        local PhasingTimeTable
        @load pt_path PhasingTimeTable
        PhasingTimeTable
    else
        Dict{Tuple{Int,Int,Float64,Float64}, Float64}()
    end

    MinTOFTable = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()

    for ((from, to, dep, arr), dv) in CostTable
        dv >= 1e7 && continue
        Tp       = get(PhasingTimeTable, (from, to, dep, arr), 0.0)
        true_tof = (arr - dep) + Tp
        key      = (from, to)
        if !haskey(MinTOFTable, key) || true_tof < MinTOFTable[key][1]
            MinTOFTable[key] = (true_tof, dv)
        end
    end

    @save out_path MinTOFTable
    return MinTOFTable
end
