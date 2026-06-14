# setup_env.jl
# Run once to create/populate the Julia environment for this project.
# Usage:  julia setup_env.jl
#   or:   julia --project=. setup_env.jl

using Pkg
Pkg.activate(dirname(abspath(@__FILE__)))

Pkg.add([
    PackageSpec(name="HDF5"),
    PackageSpec(name="JLD2"),
    PackageSpec(name="ProgressMeter"),
    PackageSpec(name="Distributions"),
    PackageSpec(name="CairoMakie"),
])

Pkg.resolve()
Pkg.instantiate()

println("\nEnvironment ready. Project.toml and Manifest.toml written to project root.")
