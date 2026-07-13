# algoLNSGA.jl — multi-objective adaptation of LNS-AGA
#
# Ported from: Han, P., Guo, Y., Li, C., Zhi, H., Lv, Y. (2022). "Multiple GEO
# satellites on-orbit repairing mission planning using large neighborhood
# search-adaptive genetic algorithm." Advances in Space Research 70, 286-302.
#
# The paper's LNS-AGA is a SINGLE-objective algorithm (scalar fitness = total
# ΔV + penalty terms) built around a closed-form two-body ΔV model (planar
# change + phasing maneuver, with the number of phasing revolutions n_ij
# solved by a cheap heuristic). This port keeps the paper's core algorithmic
# ideas but adapts them to this repo's 3-objective formulation (ΔV, unserved
# asset value, vehicles used) and reuses the repo's existing cost-table /
# min-TOF infrastructure instead of the paper's analytic ΔV formulas — the
# cost-table's (dep_grid × arr_grid) lookup already plays the same role as
# the paper's n_ij search (both answer "what does it cost to arrive at time
# t?"), just via precomputed real propagation instead of a two-body closed
# form.
#
# What's ported (adapted to multi-objective):
#   • Fixed-length permutation-with-splitters chromosome (paper Fig. 6),
#     generalized so up to `maxV` vehicle segments can be empty (unused).
#   • PMX crossover (paper Fig. 7).
#   • Swap mutation (paper Fig. 8).
#   • Fitness-adaptive crossover/mutation probabilities (paper Eq. 32-33),
#     using the paper's own tuned bounds (Pc ∈ [0.7,0.9], Pm ∈ [0.01,0.2]),
#     driven by a Pareto rank + crowding-distance fitness surrogate instead
#     of the paper's scalar fitness (there is no single scalar fitness in a
#     3-objective problem).
#   • Elitist selection + large-neighborhood-search hill-climbing on the top
#     ε fraction of the population each generation (paper Algorithm 2,
#     ε = 10%, the paper's own tuned value from Fig. 11), reusing this
#     repo's existing destroy/repair operators (Shaw removal, worst-cost
#     destroy-and-repair, consolidate, opt-times block shifts) from
#     algoMDLS.jl instead of the paper's Eq. 38-41 relatedness measure —
#     this repo's operators already use real ΔV as the relatedness signal,
#     which is strictly better information than the paper's dihedral-
#     angle/phase-difference proxy.
#
# What's NOT ported (see design discussion): the paper's closed-form
# planar-change/phasing ΔV model, its n_ij revolution-count heuristic
# (Algorithm 1), and its single-objective penalty-function fitness — none
# of these are needed once real cost tables are available, and using them
# would make LNS-GA incomparable to MDLS/NSGA-III/MOEA-D/PSO in this repo,
# which all consume the same cost tables.
#
# Requires (already defined by the time this file is included from
# GATests.jl, after the PSO section): SchedIndividual, sched_eval,
# nondominated_sort, _dominates, _extract_pareto, RunContext, vehicle,
# copy_schedule, snap_cost, _merge_unassigned, MAX_VEHICLES, REFUEL_TIME,
# POP_SIZE, BUDGET_EVALS, and the algoMDLS.jl operator functions
# (opt_times_combined, destroy_and_repair, shaw_removal_repair,
# consolidate_demands).

using Random

# ═════════════════════════════════════════════════════════════════════════════
#  CHROMOSOME  (permutation of 1:(N + maxV - 1))
# ═════════════════════════════════════════════════════════════════════════════
#
#   genes[k] ∈ 1:N            → demand POSITION (index into ctx.uids / ctx.sat_ids
#                                / ctx.svc_times / ctx.deadlines — matching the
#                                convention already used by decode_tour).
#   genes[k] ∈ N+1:N+maxV-1   → splitter symbol; the j-th splitter (value N+j)
#                                marks the boundary between vehicle-segment j
#                                and vehicle-segment j+1.
#
# A chromosome therefore always has exactly maxV segments (some possibly
# empty), regardless of how many vehicles actually end up used.

mutable struct LNSGAIndividual
    chrom :: Vector{Int}
    sched :: SchedIndividual
end

# ── Decode: chromosome → SchedIndividual ──────────────────────────────────────
# Greedy earliest-feasible-time walk per segment, identical in spirit to
# decode_tour / create_vehicle: demands that can't be feasibly reached (timing
# or missing min-TOF entry) are simply skipped and become unserved.
function decode_chromosome(chrom::Vector{Int}, ctx::RunContext, maxV::Int) :: SchedIndividual
    N        = length(ctx.uids)
    ct       = ctx.adaptive_cost_table !== nothing ? ctx.adaptive_cost_table : ctx.cost_table
    n_depots = length(ctx.depot_names)

    segs   = [Int[] for _ in 1:maxV]
    seg_i  = 1
    for g in chrom
        if g <= N
            push!(segs[seg_i], g)
        else
            seg_i = min(seg_i + 1, maxV)
        end
    end

    schedule     = vehicle[]
    assigned     = Set{Int}()
    next_dep_uid = -1

    for (vi, seg) in enumerate(segs)
        isempty(seg) && continue

        depot_name = n_depots > 0 ? ctx.depot_names[((vi - 1) % n_depots) + 1] : ctx.sim.names[1]
        depot_idx  = ctx.name_to_idx[depot_name]
        depot_dep  = 0.5

        v_uids  = Int[next_dep_uid];  next_dep_uid -= 1
        v_sats  = String[depot_name]
        v_arrs  = Float64[depot_dep]
        v_deps  = Float64[depot_dep]
        v_costs = Float64[0.0]

        prev_dep = depot_dep
        prev_idx = depot_idx
        cum_dv   = 0.0

        for pos in seg
            uid  = ctx.uids[pos]
            sat  = ctx.sat_ids[pos]
            sidx = ctx.name_to_idx[sat]

            tof_info = get(ctx.mintof_table, (prev_idx, sidx), nothing)
            tof_info === nothing && continue   # unreachable from current state — leave unserved

            leg_dv = snap_cost(ct, prev_idx, sidx, prev_dep, prev_dep + tof_info[1])

            # ΔV budget exceeded → return to depot to refuel first
            if cum_dv + leg_dv > ctx.dv_budget && prev_idx != depot_idx
                ret_info = get(ctx.mintof_table, (prev_idx, depot_idx), nothing)
                if ret_info !== nothing
                    arr_d = prev_dep + ret_info[1]
                    dep_d = arr_d + REFUEL_TIME
                    dv_d  = snap_cost(ct, prev_idx, depot_idx, prev_dep, arr_d)
                    push!(v_uids,  next_dep_uid); next_dep_uid -= 1
                    push!(v_sats,  depot_name)
                    push!(v_arrs,  arr_d)
                    push!(v_deps,  dep_d)
                    push!(v_costs, dv_d)
                    prev_dep = dep_d; prev_idx = depot_idx; cum_dv = 0.0
                    tof_info = get(ctx.mintof_table, (prev_idx, sidx), nothing)
                    tof_info === nothing && continue
                end
            end

            arr = prev_dep + tof_info[1]
            arr + ctx.svc_times[pos] > ctx.deadlines[pos] && continue   # infeasible — leave unserved
            dep = arr + ctx.svc_times[pos]
            dv  = snap_cost(ct, prev_idx, sidx, prev_dep, arr)

            push!(v_uids, uid);  push!(v_sats, sat)
            push!(v_arrs, arr);  push!(v_deps, dep);  push!(v_costs, dv)
            push!(assigned, uid)
            prev_dep = dep; prev_idx = sidx; cum_dv += dv
        end

        # Every vehicle returns to depot
        if prev_idx != depot_idx
            ret_info = get(ctx.mintof_table, (prev_idx, depot_idx), nothing)
            if ret_info !== nothing
                arr_ret = prev_dep + ret_info[1]
                dep_ret = arr_ret + REFUEL_TIME
                dv_ret  = snap_cost(ct, prev_idx, depot_idx, prev_dep, arr_ret)
                push!(v_uids,  next_dep_uid); next_dep_uid -= 1
                push!(v_sats,  depot_name)
                push!(v_arrs,  arr_ret)
                push!(v_deps,  dep_ret)
                push!(v_costs, dv_ret)
            end
        end

        length(v_uids) > 1 && push!(schedule, vehicle(v_uids, v_sats, v_arrs, v_deps, v_costs))
    end

    unassigned_uids = [ctx.uids[pos] for pos in 1:N if ctx.uids[pos] ∉ assigned]
    unas = _merge_unassigned(nothing, unassigned_uids, ctx.demands)
    return sched_eval(schedule, unas)
end

# ── Encode: SchedIndividual → chromosome ──────────────────────────────────────
# Inverse of decode_chromosome. Vehicle k's demand positions (in visiting
# order) fill segment k; unused segments (k > n_routes) are empty; any demand
# not currently placed in any vehicle route is appended after the last
# segment so it stays "in" the chromosome for future GA operators to reposition.
function encode_schedule(sched_ind::SchedIndividual, ctx::RunContext, maxV::Int) :: Vector{Int}
    N        = length(ctx.uids)
    n_routes = length(sched_ind.schedule)
    genes    = Int[]
    placed   = falses(N)

    for slot in 1:maxV
        if slot <= n_routes
            for uid in sched_ind.schedule[slot].visitedUID
                uid > 0 || continue
                haskey(ctx.uid_to_pos, uid) || continue
                pos = ctx.uid_to_pos[uid]
                push!(genes, pos)
                placed[pos] = true
            end
        end
        slot < maxV && push!(genes, N + slot)
    end

    for pos in 1:N
        placed[pos] || push!(genes, pos)
    end

    return genes
end

# ═════════════════════════════════════════════════════════════════════════════
#  GA OPERATORS — PMX crossover (Fig. 7) and swap mutation (Fig. 8)
# ═════════════════════════════════════════════════════════════════════════════

function _pmx_child(seg_donor::Vector{Int}, other_parent::Vector{Int}, i::Int, j::Int) :: Vector{Int}
    n = length(seg_donor)
    child = zeros(Int, n)
    child[i:j] = seg_donor[i:j]
    seg_set   = Set(seg_donor[i:j])
    value_map = Dict{Int,Int}()
    for k in i:j
        value_map[other_parent[k]] = seg_donor[k]
    end

    for k in 1:n
        (k >= i && k <= j) && continue
        v = other_parent[k]
        while v in seg_set
            v = value_map[v]
        end
        child[k] = v
    end
    return child
end

function pmx_crossover(p1::Vector{Int}, p2::Vector{Int}) :: Tuple{Vector{Int},Vector{Int}}
    n    = length(p1)
    i, j = extrema((rand(1:n), rand(1:n)))
    i == j && (j = min(n, j + 1))
    c1 = _pmx_child(p1, p2, i, j)
    c2 = _pmx_child(p2, p1, i, j)
    return c1, c2
end

function swap_mutation(chrom::Vector{Int}) :: Vector{Int}
    c = copy(chrom)
    n = length(c)
    n < 2 && return c
    i, j = rand(1:n), rand(1:n)
    c[i], c[j] = c[j], c[i]
    return c
end

# ═════════════════════════════════════════════════════════════════════════════
#  RANK + CROWDING  (multi-objective substitute for the paper's scalar fitness)
# ═════════════════════════════════════════════════════════════════════════════

function _crowding_distance(scheds::AbstractVector, fronts::Vector{Vector{Int}}) :: Vector{Float64}
    n     = length(scheds)
    crowd = zeros(Float64, n)
    for front in fronts
        m = length(front)
        m == 0 && continue
        if m <= 2
            for i in front; crowd[i] = Inf; end
            continue
        end
        for obj_k in 1:3
            vals = obj_k == 1 ? [scheds[i].f1 for i in front] :
                   obj_k == 2 ? [scheds[i].f2 for i in front] :
                                [scheds[i].f3 for i in front]
            order = sortperm(vals)
            crowd[front[order[1]]]   = Inf
            crowd[front[order[end]]] = Inf
            rng = vals[order[end]] - vals[order[1]]
            rng < 1e-10 && continue
            for j in 2:m-1
                if isfinite(crowd[front[order[j]]])
                    crowd[front[order[j]]] += (vals[order[j+1]] - vals[order[j-1]]) / rng
                end
            end
        end
    end
    return crowd
end

# Rank-and-crowding fitness surrogate (higher = better). Used ONLY to drive
# the paper's adaptive crossover/mutation-probability formulas (Eq. 32-33) —
# NOT for environmental/survivor selection, which uses full Pareto dominance
# via _nsga2_truncate below.
function _rank_crowding_fitness(scheds::AbstractVector, fronts::Vector{Vector{Int}}) :: Vector{Float64}
    n    = length(scheds)
    rank = zeros(Int, n)
    for (r, front) in enumerate(fronts), i in front
        rank[i] = r
    end
    max_rank = maximum(rank)
    crowd    = _crowding_distance(scheds, fronts)

    crowd_norm = similar(crowd)
    for front in fronts
        finite_vals = [crowd[i] for i in front if isfinite(crowd[i])]
        cmax = isempty(finite_vals) ? 1.0 : max(maximum(finite_vals), 1e-10)
        for i in front
            crowd_norm[i] = isfinite(crowd[i]) ? crowd[i] / cmax : 1.0
        end
    end
    return [Float64(max_rank - rank[i] + 1) + crowd_norm[i] for i in 1:n]
end

# Paper Eq. 32: adaptive crossover probability from the larger fitness of the
# two selected parents.
function _adaptive_pc(f_prime::Float64, fmax::Float64, favg::Float64, pc1::Float64, pc2::Float64) :: Float64
    f_prime >= favg ? pc1 - (pc1 - pc2) * (f_prime - favg) / max(fmax - favg, 1e-10) : pc1
end

# Paper Eq. 33: adaptive mutation probability from the individual's own fitness.
function _adaptive_pm(fi::Float64, fmax::Float64, favg::Float64, pm1::Float64, pm2::Float64) :: Float64
    fi >= favg ? pm1 - (pm1 - pm2) * (fmax - fi) / max(fmax - favg, 1e-10) : pm1
end

function _tournament(pop_size::Int, rank::Vector{Int}, crowd::Vector{Float64}) :: Int
    a, b = rand(1:pop_size), rand(1:pop_size)
    rank[a] != rank[b] && return rank[a] < rank[b] ? a : b
    return crowd[a] >= crowd[b] ? a : b
end

# NSGA-II-style elitist environmental selection: fill front-by-front, then
# truncate the last partial front by crowding distance.
function _nsga2_truncate(pop::Vector{LNSGAIndividual}, target_n::Int) :: Vector{LNSGAIndividual}
    scheds = [ind.sched for ind in pop]
    fronts = nondominated_sort(scheds)
    selected = Int[]
    for front in fronts
        if length(selected) + length(front) <= target_n
            append!(selected, front)
        else
            remaining = target_n - length(selected)
            remaining <= 0 && break
            crowd = _crowding_distance(scheds, [front])
            order = sortperm(crowd[front], rev = true)
            append!(selected, front[order[1:remaining]])
            break
        end
    end
    return pop[selected]
end

# ═════════════════════════════════════════════════════════════════════════════
#  ELITE LNS HILL-CLIMB  (paper Algorithm 2)
# ═════════════════════════════════════════════════════════════════════════════
# Repeatedly destroy-and-repair, accepting a move only if it Pareto-dominates
# the current solution (hill-climbing acceptance, matching the paper's
# accept(x',x) rule — but on the Pareto dominance relation instead of a
# scalar cost c(x)).
function _lns_hillclimb(ind::SchedIndividual, ctx::RunContext; iters::Int = 10) :: Union{SchedIndividual,Nothing}
    x = ind
    improved_any = false
    ops = [
        (s, u) -> (opt_times_combined(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.sim;
                                      ag = ctx.adaptive_cost_table), u),
        (s, u) -> destroy_and_repair(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab, ctx.sim, u),
        (s, u) -> shaw_removal_repair(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.min_dv_tab, ctx.sim, u),
        (s, u) -> consolidate_demands(s, ctx.demands, ctx.cost_table, ctx.mintof_table, ctx.sim, u;
                                      dv_budget = ctx.dv_budget),
    ]

    for _ in 1:iters
        op = rand(ops)
        new_s, new_u = try
            op(x.schedule, x.unassigned)
        catch
            continue
        end
        isempty(new_s) && continue
        cand = sched_eval(new_s, new_u)
        if _dominates(cand, x)
            x = cand
            improved_any = true
        end
    end
    return improved_any ? x : nothing
end

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

function run_lnsga(ctx::RunContext;
                   pop_size::Int              = POP_SIZE,
                   budget_evals::Int          = BUDGET_EVALS,
                   maxV::Int                  = MAX_VEHICLES,
                   elite_frac::Float64        = 0.10,   # paper's own tuned ε (Fig. 11)
                   lns_iters::Int             = 10,
                   pc_bounds::Tuple{Float64,Float64} = (0.9, 0.7),   # paper §4.1: 0.7–0.9
                   pm_bounds::Tuple{Float64,Float64} = (0.2, 0.01)) :: Tuple{Matrix{Float64}, Float64}  # paper §4.1: 0.01–0.2

    N = length(ctx.uids)
    @info "Running LNS-GA …" budget_evals=budget_evals pop_size=pop_size maxV=maxV elite_frac=elite_frac

    t0 = time()

    # ── Initial population: greedy seed + random permutations ──────────────
    seed_sched = sched_eval(copy_schedule(ctx.init_sched), ctx.init_unas)
    seed_chrom = encode_schedule(seed_sched, ctx, maxV)
    pop = Vector{LNSGAIndividual}(undef, pop_size)
    pop[1] = LNSGAIndividual(seed_chrom, seed_sched)
    Threads.@threads for i in 2:pop_size
        chrom  = randperm(N + maxV - 1)
        pop[i] = LNSGAIndividual(chrom, decode_chromosome(chrom, ctx, maxV))
    end

    n_evaluated = pop_size
    n_elite     = max(1, round(Int, elite_frac * pop_size))
    pc1, pc2    = pc_bounds
    pm1, pm2    = pm_bounds

    while n_evaluated < budget_evals
        scheds  = [ind.sched for ind in pop]
        fronts  = nondominated_sort(scheds)
        rank    = zeros(Int, pop_size)
        for (r, front) in enumerate(fronts), i in front
            rank[i] = r
        end
        crowd   = _crowding_distance(scheds, fronts)
        fitness = _rank_crowding_fitness(scheds, fronts)
        fmax, favg = maximum(fitness), sum(fitness) / length(fitness)

        # ── GA: tournament selection + adaptive PMX + adaptive swap mutation ──
        offspring = Vector{LNSGAIndividual}(undef, pop_size)
        n_pairs   = cld(pop_size, 2)
        Threads.@threads for k in 1:n_pairs
            p1 = _tournament(pop_size, rank, crowd)
            p2 = _tournament(pop_size, rank, crowd)

            f_prime = max(fitness[p1], fitness[p2])
            pc      = _adaptive_pc(f_prime, fmax, favg, pc1, pc2)

            c1c, c2c = rand() < pc ?
                pmx_crossover(pop[p1].chrom, pop[p2].chrom) :
                (copy(pop[p1].chrom), copy(pop[p2].chrom))

            pmv1 = _adaptive_pm(fitness[p1], fmax, favg, pm1, pm2)
            pmv2 = _adaptive_pm(fitness[p2], fmax, favg, pm1, pm2)
            rand() < pmv1 && (c1c = swap_mutation(c1c))
            rand() < pmv2 && (c2c = swap_mutation(c2c))

            idx1 = 2k - 1
            offspring[idx1] = LNSGAIndividual(c1c, decode_chromosome(c1c, ctx, maxV))
            if 2k <= pop_size
                offspring[2k] = LNSGAIndividual(c2c, decode_chromosome(c2c, ctx, maxV))
            end
        end
        n_evaluated += pop_size

        # ── Elitist environmental selection (NSGA-II-style rank+crowding) ─────
        pop = _nsga2_truncate(vcat(pop, offspring), pop_size)

        # ── LNS hill-climb on the current elite fraction ──────────────────────
        elite_fronts = nondominated_sort([ind.sched for ind in pop])
        elite_order  = vcat(elite_fronts...)[1:min(n_elite, pop_size)]
        for i in elite_order
            n_evaluated >= budget_evals && break
            improved = _lns_hillclimb(pop[i].sched, ctx; iters = lns_iters)
            n_evaluated += lns_iters
            if improved !== nothing
                pop[i] = LNSGAIndividual(encode_schedule(improved, ctx, maxV), improved)
            end
        end
    end

    elapsed = time() - t0
    front   = _extract_pareto(NTuple{3,Float64}[(ind.sched.f1, ind.sched.f2, ind.sched.f3) for ind in pop])
    @info "LNS-GA done" n_evaluated=n_evaluated elapsed_sec=round(elapsed; digits=1) front_size=size(front,1)
    return front, elapsed
end
