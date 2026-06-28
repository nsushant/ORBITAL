"""
export_cost_table.jl

Run once to convert the adaptive JLD2 cost table into a flat HDF5 file
readable directly by h5py in Python.

Usage:
    julia export_cost_table.jl

Output: outputs/cost_table.h5  (five flat arrays: from_idx, to_idx, dep, arr, cost)
"""

using JLD2, HDF5

const SRC = length(ARGS) >= 1 ? ARGS[1] : "outputs/cost_table_adaptive.jld2"
const DST = length(ARGS) >= 2 ? ARGS[2] : "outputs/cost_table.h5"

println("Loading cost table from $SRC ...")
d = jldopen(SRC, "r") do f
    # adaptive table uses key "d"; normal table uses key "CostTable"
    haskey(f, "d") ? f["d"] : f["CostTable"]
end

n = length(d)
println("  $n entries found.")

from_v  = Vector{Int64}(undef, n)
to_v    = Vector{Int64}(undef, n)
dep_v   = Vector{Float64}(undef, n)
arr_v   = Vector{Float64}(undef, n)
cost_v  = Vector{Float64}(undef, n)

for (i, (k, v)) in enumerate(d)
    from_v[i] = k[1]
    to_v[i]   = k[2]
    dep_v[i]  = k[3]
    arr_v[i]  = k[4]
    cost_v[i] = v
end

println("Writing to $DST ...")
h5open(DST, "w") do f
    f["from_idx"] = from_v
    f["to_idx"]   = to_v
    f["dep"]      = dep_v
    f["arr"]      = arr_v
    f["cost"]     = cost_v
end

println("Done. $(filesize(DST) ÷ 1024) KB written.")
