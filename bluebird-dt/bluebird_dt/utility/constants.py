# Radius of Earth (m)
R_E = 6325766.0
# Or the WGS84 version
R_E_WGS84 = 6378137.0
# Flattening of Earth
F = 1 / 298.257223563

# Default origin for stereographic projection
THETA0 = -(27 / 60 + 41 / 3600)
PHI0 = 55

# Adiabatic index of air (dimensionless)
KAPPA = 1.4

# Real gas constant of air (m^2/[Ks^2])
R_GAS = 287.05287

# Gravitational acceleration (m/s^2)
G_ACC = 9.80665

# ISA temperature gradient with altitude below tropopause (K/m)
BETA_T = -0.0065

# ISA temperature at Mean Sea Level (assumed to be where geopotential
# pressure altitude is zero). (K)
T_ZERO = 288.15

# ISA pressure at Mean Sea Level (assumed to be where geopotential
# pressure altitude is zero). (Pa)
P_ZERO = 101325.0

# ISA density at Mean Sea Level (assumed to be where geopotential
# altitude is zero). (kg/m^3)
RHO_ZERO = 1.225

# Geopotential pressure altitude location of the tropopause (m).
HP_TROP = 11000.0

# ISA speed of sound at sea level (m/s)
A_ZERO = 340.294

# Derived constants
MU = (KAPPA - 1.0) / KAPPA
MU_INV = 1.0 / MU
P_OVER_RHO_ZERO = P_ZERO / RHO_ZERO
EXPO = -1.0 * G_ACC / (BETA_T * R_GAS)
EXPO_INV = 1.0 / EXPO

# Approximate nautical miles per degree on the earth
NM_PER_DEGREE = 60

# Correction factor used to convert barometric altitudes
# (see Appendix A, MATS Part 1)
QNE = 1013.25
H_PA_CORRECTION_FACTOR = 27.3
