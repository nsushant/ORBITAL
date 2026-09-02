"""Physical constants. Units are km, s, rad throughout the package."""

J2 = 1.0825267e-3
R_E = 6378.137          # km
MU = 398600.4418        # km^3/s^2
G0 = 9.80665e-3         # km/s^2
DAY = 86400.0           # s

# Appears in the averaged RAAN dynamics of Lu et al. Section 3.
K_J2 = 3.0 * J2 * R_E**2 / (2.0 * MU**3)   # s^6/km^7
