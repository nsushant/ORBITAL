"""The servicer of Section 3.2.1, in one place.

Exotrail spacevan LEO v2 class. A single fixed design point: the paper does not
sweep thrust, so nothing here is a free parameter of the study.

    wet mass    300 kg
    thrust       28 mN   (ExoMG-nano class Hall-effect thruster)
    Isp        1100 s
    delta-v     500 m/s  usable

Units follow the rest of the package: km, s, kg. Thrust is therefore in
kg*km/s^2, which is 1e-3 N, and the rocket equation in oos.edelbaum consumes
MASS/ISP/THRUST in exactly these units.
"""

MASS = 300.0            # kg, wet
ISP = 1100.0            # s
THRUST_N = 0.028        # N
THRUST = THRUST_N * 1e-3    # kg*km/s^2
DV_BUDGET = 0.500       # km/s

# Initial thrust acceleration, the quantity Section 3.2.1 quotes in SI.
ACCEL_0 = THRUST / MASS         # km/s^2
ACCEL_0_SI = ACCEL_0 * 1e3      # m/s^2  -> 9.333e-5

STATE_KWARGS = dict(mass=MASS, isp=ISP, thrust=THRUST)
