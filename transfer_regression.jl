# transfer_regression.jl
#
# Regression tests for three defects in the Section 4.2 coasting-orbit estimate
# (`calculate_transfer_cost`, cost/lu_transfer.jl). None of these are exercised by
# paper_validation.jl: all three of the paper's Table 2 cases have small RAAN gaps
# and high coasting orbits, so the wrap and the altitude floor are both no-ops
# there. That is intentional — the paper cases must not move — but it means the
# fixes need their own coverage.
#
#   1. RAAN branch.  Orbital elements come back from atan2 in (-pi, pi], so the raw
#      difference RAANf - RAAN0 can demand a 340 deg plane change where 20 deg
#      would do. Cost must depend only on the geometry, not on where the pair
#      happens to sit relative to the +/-180 deg seam.
#
#   2. Altitude floor (Eq. 55).  The Case 2 solve clamps the coasting orbit to
#      a_min. Short transfer times must not buy a cheap sub-surface drift orbit.
#
#   3. Case 1 validity guard.  Eq. 51 is a closed form with no g2 constraint, and
#      it is the common path for a Walker shell (identical a and I, RAAN-only
#      differences). Below a_min the estimate is out of its domain and must be
#      reported as unreachable rather than as a cheap transfer.
#
# Run from the basic_project root:
#   julia --project=. transfer_regression.jl

include(joinpath(@__DIR__, "cost", "lu_transfer.jl"))   # brings in `using NLsolve`

using Test

const A_MIN_REG = A_MIN_LT          # R_E + 200 km
const RE_REG    = R_E_LT
const UNREACHABLE = 1e7                             # infeasible sentinel floor [m/s]

# 550 km / 53 deg shell — the repo's README constellation.
const A_SHELL = RE_REG + 550.0
const I_SHELL = deg2rad(53.0)

dv(a0, I0, R0, af, If, Rf, Tf) =
    calculate_transfer_cost(a0, I0, R0, af, If, Rf, Tf)["deltaV_total"]
coast_a(a0, I0, R0, af, If, Rf, Tf) =
    calculate_transfer_cost(a0, I0, R0, af, If, Rf, Tf)["coasting_orbit"]["a"]

# Two orbits in the same shell, offset in RAAN, differing only microscopically in
# semi-major axis so the Case 2 branch is taken.
const A_NEAR = A_SHELL + 1e-4

@testset "Transfer cost regressions" begin

    @testset "RAAN branch invariance" begin
        # Identical geometry (-20 deg gap), expressed three ways. The third
        # straddles the +/-180 deg seam, which is what atan2 output actually does.
        base = dv(A_SHELL, I_SHELL, 0.0,             A_NEAR, I_SHELL, deg2rad(-20.0), 30.0)
        seam = dv(A_SHELL, I_SHELL, deg2rad(170.0),  A_NEAR, I_SHELL, deg2rad(150.0), 30.0)
        wrapd= dv(A_SHELL, I_SHELL, deg2rad(-170.0), A_NEAR, I_SHELL, deg2rad(170.0), 30.0)

        @test base ≈ seam  rtol = 1e-9
        @test base ≈ wrapd rtol = 1e-9

        # Sanity: the unwrapped reading of the seam case would be +340 deg, which
        # is far more expensive. Guards against the test passing vacuously.
        long_way = dv(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL, deg2rad(340.0), 30.0)
        @test long_way ≈ base rtol = 1e-9      # 340 deg IS -20 deg, mod 2pi

        # A genuinely large gap must still cost more than a small one.
        small = dv(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL, deg2rad(-5.0),   90.0)
        large = dv(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL, deg2rad(-120.0), 90.0)
        @test large > small
    end

    @testset "Coasting orbit respects the Eq. 55 floor (Case 2 path)" begin
        for Tf in (5.0, 10.0, 30.0, 90.0)
            res = calculate_transfer_cost(A_SHELL, I_SHELL, 0.0,
                                          A_NEAR, I_SHELL, deg2rad(-20.0), Tf)
            ac = res["coasting_orbit"]["a"]
            # Either clamped at or above the floor, or flagged unreachable.
            @test ac >= A_MIN_REG - 1e-6 || res["deltaV_total"] >= UNREACHABLE
        end

        # Squeezing the transfer time must not make it cheaper.
        @test dv(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL, deg2rad(-20.0),  5.0) >
              dv(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL, deg2rad(-20.0), 90.0)
    end

    @testset "Case 1 guard rejects sub-surface coasting orbits" begin
        # Identical a and I, RAAN-only: exactly the Eq. 51 closed-form path.
        # A -36 deg gap drives the coasting orbit below the surface at short TOF.
        for Tf in (5.0, 10.0, 30.0)
            res = calculate_transfer_cost(A_SHELL, I_SHELL, 0.0,
                                          A_SHELL, I_SHELL, deg2rad(-36.0), Tf)
            @test res["deltaV_total"] >= UNREACHABLE
            @test res["coasting_orbit"]["a"] < A_MIN_REG   # and we say why
        end

        # With enough time the same transfer is legitimate and cheap.
        res = calculate_transfer_cost(A_SHELL, I_SHELL, 0.0,
                                      A_SHELL, I_SHELL, deg2rad(-36.0), 90.0)
        @test res["coasting_orbit"]["a"] >= A_MIN_REG
        @test 0.0 < res["deltaV_total"] < 1000.0
    end

    @testset "a_min is honoured when overridden" begin
        # A stricter floor can only make a transfer more expensive, never less.
        loose = calculate_transfer_cost(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL,
                                        deg2rad(-20.0), 30.0; a_min = RE_REG + 200.0)
        tight = calculate_transfer_cost(A_SHELL, I_SHELL, 0.0, A_NEAR, I_SHELL,
                                        deg2rad(-20.0), 30.0; a_min = RE_REG + 500.0)
        @test tight["deltaV_total"] >= loose["deltaV_total"]
        @test tight["coasting_orbit"]["a"] >= RE_REG + 500.0 - 1e-6 ||
              tight["deltaV_total"] >= UNREACHABLE
    end

    @testset "Sane behaviour on the trivial and degenerate cases" begin
        # Zero RAAN gap, same orbit: free (or as near as makes no difference).
        @test dv(A_SHELL, I_SHELL, 0.0, A_SHELL, I_SHELL, 0.0, 30.0) < 1.0

        # Near-polar: J2 drift vanishes, the coasting method degenerates and must
        # report unreachable rather than dividing through by ~zero.
        polar = calculate_transfer_cost(A_SHELL, deg2rad(90.0), 0.0,
                                        A_NEAR, deg2rad(90.0), deg2rad(20.0), 30.0)
        @test isfinite(polar["deltaV_total"])
        @test polar["deltaV_total"] >= UNREACHABLE
    end
end
