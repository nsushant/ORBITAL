# compare_lu_edelbaum.jl
# Evaluate Lu 4.2 and the Edelbaum oracle on the same plane-representative
# spirals used to build the cost table. Does not write cost_table.jld2.
#
# Run: julia --project=. -t auto compare_lu_edelbaum.jl
# Out:  outputs/lu_edelbaum_compare.csv
#       stdout summary

using Statistics, Random, DelimitedFiles, Printf

include("sim/propagator.jl")
include("fuel_cost_calc/gen_cost_table.jl")

const INFEAS     = 1e7          # both oracles use 2e7 / 1e8 sentinels
const DEP_DAYS   = [30.0, 180.0]
const TOF_DAYS   = [30.0, 60.0, 90.0, 180.0]
const MAX_CROSS  = 400          # directed plane-pair cap (random sample if more)
const SEED       = 42

feasible(dv) = isfinite(dv) && dv < INFEAS && dv >= 0.0

sim = load_sim()
sats   = findall(startswith("sat"),   sim.names)
depots = findall(startswith("depot"), sim.names)
isempty(depots) && error("No depot in simulation.h5")
depot  = only(depots)

plane_groups = get_plane_groups(sim, sats)
plane_ids    = sort(collect(keys(plane_groups)))
rep_of       = Dict(pid => plane_groups[pid][1] for pid in plane_ids)
n_planes     = length(plane_ids)
@info "Geometry" n_sats=length(sats) n_planes=n_planes n_depots=length(depots)

rep_idxs = unique(vcat(depot, [rep_of[pid] for pid in plane_ids]))
oe_data  = load_oe_data(sim, rep_idxs)

directed = Tuple{Int,Int,String}[]   # (from_idx, to_idx, kind)
for pid in plane_ids
    push!(directed, (depot, rep_of[pid], "depot_to_sat"))
    push!(directed, (rep_of[pid], depot, "sat_to_depot"))
end
cross = Tuple{Int,Int,String}[(rep_of[pi], rep_of[pj], "sat_to_sat")
                              for pi in plane_ids, pj in plane_ids if pi != pj]
if length(cross) > MAX_CROSS
    rng = MersenneTwister(SEED)
    cross = cross[randperm(rng, length(cross))[1:MAX_CROSS]]
    @info "Sampled sat↔sat plane pairs" n=MAX_CROSS of=n_planes*(n_planes-1)
end
append!(directed, cross)
@info "Directed pairs × (dep,tof)" n_pairs=length(directed) n_eval=length(directed)*length(DEP_DAYS)*length(TOF_DAYS)

# rows: lu, eb, tof, dinc_deg, draan_deg, kind_code
# kind_code: 1 depot_to_sat, 2 sat_to_depot, 3 sat_to_sat
kind_code = Dict("depot_to_sat"=>1, "sat_to_depot"=>2, "sat_to_sat"=>3)
n_eval = length(directed) * length(DEP_DAYS) * length(TOF_DAYS)
lu_v  = fill(NaN, n_eval)
eb_v  = fill(NaN, n_eval)
tof_v = fill(NaN, n_eval)
dinc  = fill(NaN, n_eval)
draan = fill(NaN, n_eval)
kind  = zeros(Int, n_eval)

tasks = Tuple{Int,Int,String,Float64,Float64}[]
for (fr, to, k) in directed, dep in DEP_DAYS, tof in TOF_DAYS
    push!(tasks, (fr, to, k, dep, tof))
end

prog = Progress(n_eval; desc="  Lu vs Edelbaum: ", barlen=40, showspeed=true)
Threads.@threads for i in eachindex(tasks)
    fr, to, k, dep, tof = tasks[i]
    ki = nearest_idx(sim.times, dep * 86400.0)
    kj = nearest_idx(sim.times, (dep + tof) * 86400.0)
    oi = oe_data[fr];  oj = oe_data[to]
    a0, I0, R0 = oi[1,ki], oi[2,ki], oi[3,ki]
    af, If, Rf = oj[1,kj], oj[2,kj], oj[3,kj]

    lu_v[i]  = LT_spiral_only_cached(a0, I0, R0, af, If, Rf, tof; oracle=:lu)
    eb_v[i]  = LT_spiral_only_cached(a0, I0, R0, af, If, Rf, tof; oracle=:edelbaum)
    tof_v[i] = tof
    dinc[i]  = abs(If - I0) * 180 / π
    draan[i] = abs(wrap_pi(Rf - R0)) * 180 / π
    kind[i]  = kind_code[k]
    next!(prog)
end
finish!(prog)

lu_ok = feasible.(lu_v)
eb_ok = feasible.(eb_v)
both  = lu_ok .& eb_ok
n_both = count(both)

@printf("\nFeasibility\n")
@printf("  both feasible     %6d  (%.1f%%)\n", n_both, 100*n_both/n_eval)
@printf("  Lu only           %6d\n", count(lu_ok .& .!eb_ok))
@printf("  Edelbaum only     %6d\n", count(eb_ok .& .!lu_ok))
@printf("  both infeasible   %6d\n", count(.!lu_ok .& .!eb_ok))
@printf("  total evaluated   %6d\n", n_eval)

if n_both == 0
    error("No overlapping feasible transfers — cannot compute a difference.")
end

δ    = eb_v[both] .- lu_v[both]
absδ = abs.(δ)
rel  = absδ ./ max.(lu_v[both], 1.0)   # avoid /0 on free J2 waits

@printf("\nΔV difference  Edelbaum − Lu   [m/s]  (n=%d both feasible)\n", n_both)
@printf("  mean signed      %10.1f\n", mean(δ))
@printf("  mean |diff|      %10.1f\n", mean(absδ))
@printf("  median |diff|    %10.1f\n", median(absδ))
@printf("  p90 |diff|       %10.1f\n", quantile(absδ, 0.90))
@printf("  rms              %10.1f\n", sqrt(mean(absδ.^2)))
@printf("  mean relative    %10.2f %%\n", 100*mean(rel))
@printf("  median relative  %10.2f %%\n", 100*median(rel))

@printf("\nBy pair type  (mean |diff| m/s, mean rel %%, n)\n")
for (code, name) in ((1,"depot→sat"), (2,"sat→depot"), (3,"sat↔sat"))
    msk = both .& (kind .== code)
    n = count(msk)
    n == 0 && continue
    ad = abs.(eb_v[msk] .- lu_v[msk])
    rl = ad ./ max.(lu_v[msk], 1.0)
    @printf("  %-12s  %8.1f   %6.1f %%   n=%d\n", name, mean(ad), 100*mean(rl), n)
end

@printf("\nBy |Δi|  (mean |diff| m/s, n)\n")
for (lo, hi, lab) in ((0.0, 2.0, "Δi < 2°"),
                      (2.0, 20.0, "2° ≤ Δi < 20°"),
                      (20.0, 90.0, "Δi ≥ 20°"))
    msk = both .& (dinc .>= lo) .& (dinc .< hi)
    hi == 90.0 && (msk = both .& (dinc .>= lo))
    n = count(msk)
    n == 0 && continue
    @printf("  %-16s  %8.1f   n=%d\n", lab, mean(abs.(eb_v[msk] .- lu_v[msk])), n)
end

mkpath("outputs")
header = ["kind","tof_days","dinc_deg","draan_deg","lu_m_s","eb_m_s","diff_m_s"]
kind_name = Dict(1=>"depot_to_sat", 2=>"sat_to_depot", 3=>"sat_to_sat")
open("outputs/lu_edelbaum_compare.csv", "w") do io
    println(io, join(header, ","))
    for i in 1:n_eval
        diff = (lu_ok[i] && eb_ok[i]) ? (eb_v[i] - lu_v[i]) : NaN
        @printf(io, "%s,%.1f,%.3f,%.3f,%.4f,%.4f,%.4f\n",
                kind_name[kind[i]], tof_v[i], dinc[i], draan[i],
                lu_v[i], eb_v[i], diff)
    end
end
@info "Wrote outputs/lu_edelbaum_compare.csv"
