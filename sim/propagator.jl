# sim/propagator.jl
# gen_simulation: builds a Simulation struct by propagating satellites with
# RK4+J2 and writing hourly snapshots to outputs/simulation.h5 (HDF5).

include(joinpath(@__DIR__, "orbital_mechanics.jl"))

using HDF5
using ProgressMeter

const TRAJ_FILE    = "outputs/simulation.h5"
const WRITE_EVERY  = 60       # steps between HDF5 writes (dt=60s → 1 hour per record)

# ── Simulation struct ──────────────────────────────────────────────────────────
struct Simulation
    positions        :: Matrix{Float64}   # 3×N at t=0 [km]
    orbital_elements :: Matrix{Float64}   # 4×N at t=0 (a, i, raan, nu)
    id_to_idx        :: Dict{String,Int}
    names            :: Vector{String}
    times            :: Vector{Float64}   # shared time grid [seconds]
    traj_file        :: String
end

# ── Helpers ────────────────────────────────────────────────────────────────────

function _existing_sat_indices(h5file::String) :: Vector{Int}
    isfile(h5file) || return Int[]
    indices = Int[]
    h5open(h5file, "r") do f
        for name in keys(f)
            m = match(r"^sat_(\d+)$", name)
            m !== nothing && push!(indices, parse(Int, m.captures[1]))
        end
    end
    return indices
end

"""
    load_sim() → Simulation

Load an existing simulation from `outputs/simulation.h5`.
"""
function load_sim() :: Simulation
    isfile(TRAJ_FILE) || error("No simulation file found at $TRAJ_FILE — run gen_simulation() first.")
    return _load_simulation()
end

function _load_simulation() :: Simulation
    h5open(TRAJ_FILE, "r") do f
        names     = read(f["metadata/names"])
        times     = read(f["metadata/times"])
        N         = length(names)
        positions = hcat([read(f["$(names[i])/positions"])[:,  1] for i in 1:N]...)
        oe_init   = hcat([read(f["$(names[i])/orbital_elements"])[:, 1] for i in 1:N]...)
        id_to_idx = Dict(names[i] => i for i in 1:N)
        return Simulation(positions, oe_init, id_to_idx, names, times, TRAJ_FILE)
    end
end

function _next_sat_offset(h5file::String) :: Int
    idxs = _existing_sat_indices(h5file)
    isempty(idxs) ? 0 : maximum(idxs) + 1
end

function _build_sats(params_list::Vector, sat_offset::Int) :: Tuple{Vector{Sat}, Int}
    sats       = Sat[]
    depot_count = 0
    current_sat_offset = sat_offset

    for params in params_list
        t = get(params, "type", "depot")

        if t == "delta_walker"
            num_planes = Int(params["num_planes"])
            num_sats   = Int(params["num_satellites"])
            phase      = Int(get(params, "phasing", 1))
            alt        = Float64(params["altitude"])
            inc        = deg2rad(Float64(params["inclination"]))
            sma        = Re_SIM + alt

            new_sats = walker_delta(num_planes, num_sats, phase, sma, inc;
                                    name_offset = current_sat_offset)
            append!(sats, new_sats)
            current_sat_offset += num_sats

        elseif t == "depot"
            depot_count += 1
            alt  = Float64(params["altitude"])
            inc  = deg2rad(Float64(params["inclination"]))
            raan = deg2rad(Float64(get(params, "RAAN", 0.0)))
            ecc  = Float64(get(params, "e", 0.0))
            sma  = Re_SIM + alt
            r, v = eci_from_oelem(OrbElem(sma, ecc, inc, raan, 0.0, 0.0))
            push!(sats, Sat("depot_$depot_count", r, v))

        elseif t == "debris"
            # reserved for future use — silently skip
            continue

        else
            @warn "Unknown params type \"$t\" — skipping"
        end
        # Vinit key is intentionally ignored for now
    end

    return sats, depot_count
end

# ── Main entry point ───────────────────────────────────────────────────────────
"""
    gen_simulation(params_list, sim_params) → Simulation

Propagate all satellites in `params_list` with RK4+J2 and write hourly snapshots
to `outputs/simulation.h5`.

`params_list` is a Vector of Dicts, each with a `"type"` key:
  - `"delta_walker"` — Walker Delta constellation
  - `"depot"`        — single depot satellite
  - `"debris"`       — reserved, skipped for now

`sim_params` keys: `"J2"` (Bool), `"dt"` (seconds), `"t_end"` (seconds)
"""
function gen_simulation(params_list::Vector, sim_params::Dict) :: Simulation

    use_J2 = Bool(sim_params["J2"])
    dt     = Float64(sim_params["dt"])
    t_end  = Float64(sim_params["t_end"])

    mkpath("outputs")

    # ── Prompt if simulation file already exists ───────────────────────────────
    if isfile(TRAJ_FILE)
        println("Simulation file found at $TRAJ_FILE. Re-run? (y/n): ")
        choice = strip(readline())
        if choice != "y" && choice != "Y"
            @info "Loading existing simulation from $TRAJ_FILE"
            return _load_simulation()
        end
    end

    # ── Check for existing sat files and warn ──────────────────────────────────
    sat_offset   = _next_sat_offset(TRAJ_FILE)
    existing_idx = _existing_sat_indices(TRAJ_FILE)

    # Build sat list to detect potential name collisions before propagating
    sats_preview, _ = _build_sats(params_list, sat_offset)
    new_names        = [s.name for s in sats_preview]
    if isfile(TRAJ_FILE) && !isempty(existing_idx)
        collisions = filter(n -> startswith(n, "sat_") &&
                            parse(Int, split(n, "_")[2]) ∈ existing_idx, new_names)
        if !isempty(collisions)
            collision_str = join(collisions, ", ")
            @warn "gen_simulation: overwriting existing trajectory data for: $collision_str"
        end
    end

    sats, _ = _build_sats(params_list, sat_offset)
    N       = length(sats)
    n_steps = round(Int, t_end / dt)
    n_rec   = n_steps ÷ WRITE_EVERY   # total hourly records per node

    t_end_years = t_end / (365.25 * 86400.0)
    names_preview = [s.name for s in sats]
    n_sats   = count(n -> startswith(n, "sat_"),   names_preview)
    n_depots = count(n -> startswith(n, "depot_"), names_preview)

    @info "Starting simulation" n_sats=n_sats n_depots=n_depots t_end_years=round(t_end_years, digits=2) dt_s=dt J2=use_J2

    # ── Allocate per-node trajectory buffers ───────────────────────────────────
    # [node_idx][field, record]
    pos_bufs = [zeros(3, n_rec) for _ in 1:N]
    vel_bufs = [zeros(3, n_rec) for _ in 1:N]
    oe_bufs  = [zeros(4, n_rec) for _ in 1:N]   # (a, i, raan, nu)
    times    = zeros(n_rec)

    # ── Propagate ─────────────────────────────────────────────────────────────
    prog = Progress(n_rec; desc="Propagating: ", barlen=40, showspeed=true)

    # Work on independent copies per satellite (thread-safe)
    sat_copies = [Sat(sats[i].name, copy(sats[i].r), copy(sats[i].v)) for i in 1:N]

    rec   = 1
    t     = 0.0
    for step in 1:n_steps
        if step % WRITE_EVERY == 0 && rec <= n_rec
            times[rec] = t
            for i in 1:N
                el = oelem_from_rv(sat_copies[i].r, sat_copies[i].v)
                pos_bufs[i][:, rec] .= sat_copies[i].r
                vel_bufs[i][:, rec] .= sat_copies[i].v
                oe_bufs[i][:, rec]  .= [el.sma, el.inc, el.raan, el.nu]
            end
            next!(prog)
            rec += 1
        end
        for i in 1:N
            rk4!(sat_copies[i], dt, use_J2)
        end
        t += dt
    end
    finish!(prog)

    # ── Write HDF5 ─────────────────────────────────────────────────────────────
    @info "Writing to $TRAJ_FILE …"
    h5open(TRAJ_FILE, isfile(TRAJ_FILE) ? "r+" : "w") do f
        # metadata
        if !haskey(f, "metadata")
            g = create_group(f, "metadata")
        else
            g = f["metadata"]
        end
        all_names = [s.name for s in sats]
        haskey(g, "names") && delete_object(g, "names")
        haskey(g, "times") && delete_object(g, "times")
        g["names"] = all_names
        g["times"] = times
        attrs(g)["J2"]    = use_J2
        attrs(g)["dt"]    = dt
        attrs(g)["t_end"] = t_end

        for i in 1:N
            name = sats[i].name
            haskey(f, name) && delete_object(f, name)
            sg = create_group(f, name)
            sg["positions"]        = pos_bufs[i]
            sg["velocities"]       = vel_bufs[i]
            sg["orbital_elements"] = oe_bufs[i]
        end
    end

    # ── Build Simulation struct (t=0 initial conditions) ──────────────────────
    names     = [s.name for s in sats]
    id_to_idx = Dict(names[i] => i for i in 1:N)

    positions = hcat([pos_bufs[i][:, 1] for i in 1:N]...)
    oe_init   = hcat([oe_bufs[i][:, 1]  for i in 1:N]...)

    file_mb = isfile(TRAJ_FILE) ? round(filesize(TRAJ_FILE) / 1e6, digits=1) : 0.0
    @info "Simulation complete" records_per_node=n_rec file_size_mb=file_mb

    return Simulation(positions, oe_init, id_to_idx, names, times, TRAJ_FILE)
end
