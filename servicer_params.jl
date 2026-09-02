# Edelbaum BCR servicer: Isp 2500 s, 7000 m/s tank.
# Thrust T is kg·km/s² (1 N = 1e-3). Override with ENV THRUST_N (Newtons).
# Do not use Lu paper fmax=3.5e-6 here.

const ISP_S      = 2500.0
const THRUST_N   = parse(Float64, get(ENV, "THRUST_N", "0.01"))  # default 10 mN
const T_FORCE    = THRUST_N * 1e-3
const M_SERVICER = 300.0
const DV_BUDGET  = 7000.0

function thrust_tag(T_N::Float64=THRUST_N)
    mN = T_N * 1e3
    return mN >= 1000 - 1e-6 ? string(round(Int, T_N), "N") :
                               string(round(Int, mN), "mN")
end

const THRUST_TAG = thrust_tag()
