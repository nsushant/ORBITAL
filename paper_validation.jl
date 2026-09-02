# paper_validation.jl
#
# Consistency check of the low-thrust (Lu) fuel-cost model against the stated
# values in:
#   S. Lu et al., "Low-thrust transfer solution between circular orbits based on
#   yaw switch steering and analytical propagation", Acta Astronautica 245 (2026)
#   924-936.
#
# `calculate_transfer_cost` (cost/lu_transfer.jl) implements the *impulsive
# coasting-orbit estimate* of Section 4.2 (Eqs. 49-53, 57-60). The matching
# benchmark is Table 2 (NOT Tables 4/6/8, which are the full continuous-thrust
# NLP results using analytical dynamics not implemented here).
#
# Run from the basic_project root:
#   julia --project=. paper_validation.jl

include(joinpath(@__DIR__, "cost", "lu_transfer.jl"))   # brings in `using NLsolve`

using Test
using Printf

const RE  = 6378.137          # km, Earth radius used in the paper
deg(x) = x * pi / 180.0
rad2deg_(x) = x * 180.0 / pi

cases = [
    (name = "Case 1  (800->800 km, 98->98 deg, RAAN 0->30)",
     a0 = RE + 800.0, I0 = 98.0, R0 = 0.0,
     af = RE + 800.0, If = 98.0, Rf = 30.0, Tf = 100.0,
     J = 484.664, Jt = 242.332, Ja = 242.332, ac = 6853.309, Ic = 99.318),

    (name = "Case 2  (800->900 km, 98->99 deg, RAAN 0->30)",
     a0 = RE + 800.0, I0 = 98.0, R0 = 0.0,
     af = RE + 900.0, If = 99.0, Rf = 30.0, Tf = 100.0,
     J = 550.915, Jt = 304.428, Ja = 246.487, ac = 6873.711, Ic = 100.010),

    (name = "Case 3  (700->1200 km, 49.9->50 deg, RAAN 0->-80)",
     a0 = RE + 700.0, I0 = 49.9, R0 = 0.0,
     af = RE + 1200.0, If = 50.0, Rf = -80.0, Tf = 100.0,
     J = 251.9, Jt = 36.2, Ja = 215.7, ac = 7147.626, Ic = 49.914),
]

# Tolerances. Cases 1 and 3 reproduce the paper to <0.02 km in the coasting
# semi-major axis. The remaining slack is for Case 2, whose published Δa
# (-304.426 km) matches neither the a0- nor the ā-normalised form of Eq. 44
# exactly; the resulting ΔV difference is 0.27%.
const ATOL_DV = 2.0     # m/s
const ATOL_A  = 2.5     # km
const ATOL_I  = 0.10    # deg

@printf("\n%-52s %10s %10s %10s\n", "field", "computed", "paper", "abs.err")
println("-"^86)

@testset "Paper Table 2 consistency (Lu impulsive estimate)" begin
    for c in cases
        res = calculate_transfer_cost(c.a0, deg(c.I0), deg(c.R0),
                                      c.af, deg(c.If), deg(c.Rf), c.Tf)
        Jc  = res["deltaV_total"]
        Jtc = res["deltaV_transfer"]
        Jac = res["deltaV_adjust"]
        acc = res["coasting_orbit"]["a"]
        Icc = rad2deg_(res["coasting_orbit"]["I"])

        println("\n", c.name)
        @printf("  %-50s %10.3f %10.3f %10.3f\n", "J total   [m/s]", Jc,  c.J,  abs(Jc  - c.J))
        @printf("  %-50s %10.3f %10.3f %10.3f\n", "Jt        [m/s]", Jtc, c.Jt, abs(Jtc - c.Jt))
        @printf("  %-50s %10.3f %10.3f %10.3f\n", "Ja        [m/s]", Jac, c.Ja, abs(Jac - c.Ja))
        @printf("  %-50s %10.3f %10.3f %10.3f\n", "coast a   [km] ", acc, c.ac, abs(acc - c.ac))
        @printf("  %-50s %10.3f %10.3f %10.3f\n", "coast I   [deg]", Icc, c.Ic, abs(Icc - c.Ic))

        @testset "$(c.name)" begin
            @test isapprox(Jc,  c.J;  atol = ATOL_DV)
            @test isapprox(Jtc, c.Jt; atol = ATOL_DV)
            @test isapprox(Jac, c.Ja; atol = ATOL_DV)
            @test isapprox(acc, c.ac; atol = ATOL_A)
            @test isapprox(Icc, c.Ic; atol = ATOL_I)
        end
    end
end
