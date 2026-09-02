# Servicer of Section 3.2.1: Exotrail spacevan LEO v2 class.
#
#   wet mass   300 kg
#   thrust      28 mN   (Hall-effect, ExoMG-nano class)
#   Isp       1100 s
#   delta-v    500 m/s  usable
#
# Thrust T is kg*km/s^2 (1 N = 1e-3). THRUST_N is in Newtons.
# There is no thrust sweep: the vehicle is a single fixed design point.
# Do not use Lu paper fmax=3.5e-6 here.

const ISP_S      = 1100.0
const THRUST_N   = parse(Float64, get(ENV, "THRUST_N", "0.028"))  # 28 mN
const T_FORCE    = THRUST_N * 1e-3
const M_SERVICER = 300.0
const DV_BUDGET  = 500.0

function thrust_tag(T_N::Float64=THRUST_N)
    mN = T_N * 1e3
    return mN >= 1000 - 1e-6 ? string(round(Int, T_N), "N") :
                               string(round(Int, mN), "mN")
end

const THRUST_TAG = thrust_tag()
