from __future__ import annotations

import math
import typing
from datetime import UTC, datetime
from functools import lru_cache

import dateutil.parser
import pandas as pd
from pydantic import BeforeValidator

from bluebird_dt.utility import constants

if typing.TYPE_CHECKING:
    from bluebird_dt.core.wind import WindVector


# Lengths
# Multiply to convert meters [m] to feet [ft]
M_TO_FT = 10000.0 / 3048.0  # 3.2808398950131235
# Multiply to convert feet [ft] to meters [m]
FT_TO_M = 1.0 / M_TO_FT

# Multiply to convert nautical miles [nmi] to meters [m]
NMI_TO_M = 1852.0
# Multiply to convert meters [m] to nautical miles [nmi]
M_TO_NMI = 1.0 / NMI_TO_M

# Multiply to convert nautical miles [nmi] to feet [ft]
NMI_TO_FT = NMI_TO_M * M_TO_FT
# Multiply to convert feet [ft] to nautical miles [nmi]
FT_TO_NMI = 1.0 / NMI_TO_FT

# Multiply to convert flight levels [FL] to feet [ft]
FL_TO_FT = 100.0
# Multiply to convert ft [ft] to flight levels [FL]
FT_TO_FL = 1.0 / FL_TO_FT
# Multiply to convert flight levels [FL] to meters [m]
FL_TO_M = FL_TO_FT * FT_TO_M
# Multiply to convert meters [m] to nautical miles [nmi]
M_TO_FL = 1.0 / FL_TO_M

# Multiply to convert flight levels [FL] to nautical miles [nmi]
FL_TO_NMI = FL_TO_M * M_TO_NMI
# Multiply to convert nautical miles [nmi] to flight levels [nmi]
NMI_TO_FL = 1.0 / FL_TO_NMI

# Speeds
# Multiply to convert meters per second [m/s] to knots [kt]
MPS_TO_KT = 3600.0 / 1852.0  # 1.9438444924406046
# Multiply to convert knots [kt] to meters per second [m/s]
KT_TO_MPS = 1.0 / MPS_TO_KT

# Multiply to convert feet per minute [ft/min] to metres per second [m/s]
FPM_TO_MPS = FT_TO_M / 60.0
# Multiply to convert metres per second [m/s] to feet per minute [ft/min]
MPS_TO_FPM = 1.0 / FPM_TO_MPS

# Angles
# Multiply to convert degrees [deg] to radians [rad]
DEG_TO_RAD = math.pi / 180.0
# Multiply to convert radians [rad] to degrees [deg]
RAD_TO_DEG = 1.0 / DEG_TO_RAD

# Time
# Multiply to convert hours to seconds
HRS_TO_SEC = 3600
SEC_TO_NANOSEC = 1e9


# explicitly define the overloaded function signatures for string_to_timestamp and timestamp_to_string so that we type
# hint correctly for both None and float inputs
@typing.overload
def string_to_timestamp(s: None) -> None: ...


@typing.overload
def string_to_timestamp(s: str) -> float: ...


def string_to_timestamp(s: str | None) -> float | None:
    """
    Convert a time string in ISO format to seconds in UNIX/Posix format.

    Parameters
    ----------
    s : str
        A datetime string in ISO format "YYYY-MM-DDTHH:MM:SS.ssssss" representing UTC time
        e.g., "2019-02-15T20:12:50". Note up to 6 digits can be included in the ".ssssss" part (up to microsecond
        granularity) - any extra digits are ignored.

    Returns
    -------
    float or None
        The time in seconds (UNIX/Posix time), with decimal digits for sub-second time component.
        When s is None, the function returns None.
    """

    if s is None:
        return None

    return dateutil.parser.isoparse(s).replace(tzinfo=UTC).timestamp()


@typing.overload
def timestamp_to_string(t: None) -> None: ...


@typing.overload
def timestamp_to_string(t: float) -> str: ...


def timestamp_to_string(t: float | None) -> str | None:
    """
    Convert a timestamp in UNIX/Posix format (seconds) to an ISO format string.

    Parameters
    ----------
    t : float
        A timestamp in UNIX/Posix seconds, with decimal digits representing sub-second time component.

    Returns
    -------
    str or None
        A datetime string in ISO format "YYYY-MM-DDTHH:MM:SS.ssssss", representing UTC time
        e.g., "2019-02-15T20:12:50.000000". When t has more than 6 decimal digits, the decimal component is
        rounded to 6 digits in the string output. When t is None, the function returns None.
    """

    if t is None:
        return None

    return datetime.fromtimestamp(t, tz=UTC).isoformat(timespec="microseconds")[:-6]


@lru_cache(maxsize=10000)
def pressure_from_fl(flight_level: float) -> float:
    """
    Calculate the pressure in an atmosphere at a given flight level.
    Note here we first convert the flight level to a pressure altitude (Hp)
    See equations 3.26 in EEC Technical/Scientific Report No. 2010-001.

    Parameters
    ----------
    flight_level : float
        A flight level

    Returns
    -------
    float
        The atmospheric pressure at the corresponding flight level height.
    """
    Hp = flight_level * FL_TO_M

    if Hp <= constants.HP_TROP:
        pressure = constants.P_ZERO * (1.0 + constants.BETA_T * Hp / constants.T_ZERO) ** constants.EXPO

    else:
        T_isa_trop = constants.T_ZERO + constants.BETA_T * constants.HP_TROP
        P_trop = constants.P_ZERO * (1.0 + constants.BETA_T * constants.HP_TROP / constants.T_ZERO) ** constants.EXPO
        pressure = P_trop * math.exp(-1.0 * constants.G_ACC * (Hp - constants.HP_TROP) / (constants.R_GAS * T_isa_trop))

    return pressure


@lru_cache(maxsize=10000)
def temperature_from_fl(flight_level: float, delta_T: float) -> float:
    """
    Calculate the temperature in an atmosphere at a given flight level.
    Note here we first convert the flight level to a pressure altitude (Hp)
    See equations 3.28 in EEC Technical/Scientific Report No. 2010-001.

    Parameters
    ----------
    flight_level : float
        A flight level
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)

    Returns
    -------
    float
        The bada atmospheric temperature at the corresponding flight level
        height.
    """
    Hp = flight_level * FL_TO_M

    if Hp <= constants.HP_TROP:
        temperature = constants.T_ZERO + delta_T + constants.BETA_T * Hp

    else:
        T_isa_trop = constants.T_ZERO + constants.BETA_T * constants.HP_TROP
        T_trop = T_isa_trop + delta_T
        temperature = T_trop

    return temperature


@lru_cache(maxsize=10000)
def density_from_fl(flight_level: float, delta_T: float, fl_pressure: float | None = None) -> float:
    """
    Calculate the density in an atmosphere at a given flight level.

    Parameters
    ----------
    flight_level : float
        A flight level. The value is not used to calculate pressure from flight
        level if `fl_pressure` is not None.
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)
    fl_pressure : float, optional
        The atmospheric pressure at the corresponding flight level height.
        If this is not specified, then it is calculated using the given flight level.

    Returns
    -------
    float
        The bada atmospheric density at the corresponding flight level
        height.
    """
    if fl_pressure is None:
        fl_pressure = pressure_from_fl(flight_level)

    return fl_pressure / (constants.R_GAS * temperature_from_fl(flight_level, delta_T=delta_T))


@lru_cache(maxsize=10000)
def geopot_altitude_from_fl(flight_level: float, delta_T: float, delta_P: float) -> float:
    """
    Calculate the geopotential height at a given flight level, by
    first converting the flight level to pressure altitude (Hp) and
    then using equations 3-31 in ECC Technical/Scientific Report
    No. 2010-001.

    Parameters
    ----------
    flight_level : float
        A flight level
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)
    delta_P : float
        The BADA atmosphere pressure differential at a pressure
        altitude of zero (in Pa)

    Returns
    -------
    float
        The geopotential height in the bada atmosphere at the given flight
        level height.
    """
    Hp = flight_level * FL_TO_M

    if Hp <= constants.HP_TROP:
        if delta_P != 0.0:
            P_msl = constants.P_ZERO + delta_P
            Hp_msl = (constants.T_ZERO / constants.BETA_T) * ((P_msl / constants.P_ZERO) ** constants.EXPO_INV - 1.0)

        else:
            Hp_msl = 0.0

        T_isa_msl = constants.T_ZERO + constants.BETA_T * Hp_msl

        geopot_altitude = (
            Hp
            - Hp_msl
            + (delta_T / constants.BETA_T) * math.log((constants.T_ZERO + constants.BETA_T * Hp) / T_isa_msl)
        )

    else:
        T_isa_trop = constants.T_ZERO + constants.BETA_T * constants.HP_TROP
        T_trop = T_isa_trop + delta_T
        geopot_altitude = constants.HP_TROP + (T_trop / T_isa_trop) * (Hp - constants.HP_TROP)

    return geopot_altitude


@lru_cache(maxsize=10000)
def geodet_altitude_from_fl(flight_level: float, delta_T: float, delta_P: float) -> float:
    """
    Calculate the geodetic height at a given flight level. Here we
    first calculate the geopotential height and then convert to
    geodetic height using equation 2-7 in ECC Technical/Scientific Report
    No. 2010-001.

    Parameters
    ----------
    flight_level : float
        A flight level
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)
    delta_P : float
        The BADA atmosphere pressure differential at a pressure
        altitude of zero (in Pa)

    Returns
    -------
    float
        The geodetic height in the bada atmosphere at the given flight level
        height.
    """
    H = geopot_altitude_from_fl(flight_level, delta_T=delta_T, delta_P=delta_P)

    return constants.R_E * H / (constants.R_E - H)


@lru_cache(maxsize=10000)
def cas_to_tas(flight_level: float, calibrated_air_speed: float, delta_T: float) -> float:
    """
    Convert calibrated air speed (CAS) to true air speed (TAS).
    See equation 3.1-23 in EEC Technical/Scientific Report No. 2010-003.

    Parameters
    ----------
    flight_level : float
        A flight level. The value is not used to calculate pressure from flight
        level if `fl_pressure` is not None.
    calibrated_air_speed : float
        The calibrated air speed of the aircraft (in m/s)
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)

    Returns
    -------
    float
        The true air speed of the aircraft (in m/s)
    """
    fl_pressure = pressure_from_fl(flight_level)

    p_over_rho = fl_pressure / density_from_fl(flight_level, delta_T=delta_T, fl_pressure=fl_pressure)
    p_ratio = constants.P_ZERO / fl_pressure

    x = 1.0 + 0.5 * constants.MU * (calibrated_air_speed**2) / constants.P_OVER_RHO_ZERO
    x = (1.0 + p_ratio * (x ** (1.0 / constants.MU) - 1.0)) ** constants.MU
    return math.sqrt(2.0 * p_over_rho * (x - 1.0) / constants.MU)


@lru_cache(maxsize=10000)
def tas_to_cas(flight_level: float, true_air_speed: float, delta_T: float) -> float:
    """
    Convert true air speed (TAS) to calibrated air speed (CAS).
    See equation 3.1-24 in EEC Technical/Scientific Report No. 2010-003.

    Parameters
    ----------
    flight_level : float
        A flight level
    true_air_speed : float
        The true air speed of the aircraft (in m/s)
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)

    Returns
    -------
    float
        The calibrated air speed of the aircraft (in m/s)
    """
    fl_pressure = pressure_from_fl(flight_level)

    p_over_rho = fl_pressure / density_from_fl(flight_level, delta_T=delta_T, fl_pressure=fl_pressure)
    p_ratio = fl_pressure / constants.P_ZERO

    x = 1.0 + 0.5 * constants.MU * (true_air_speed**2) / p_over_rho
    x = (1.0 + p_ratio * (x ** (1.0 / constants.MU) - 1.0)) ** constants.MU
    return math.sqrt(2.0 * constants.P_OVER_RHO_ZERO * (x - 1.0) / constants.MU)


@lru_cache(maxsize=10000)
def tas_to_mach(flight_level: float, true_air_speed: float, delta_T: float) -> float:
    """
    Converts the true airspeed (TAS) to a mach number.
    See equation 3.1-26 in EEC Technical/Scientific Report No. 2010-003.

    Parameters
    ----------
    flight_level : float
        A flight level
    true_air_speed : float
        The true air speed of the aircraft (in m/s)
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)

    Returns
    -------
    float
        The aircraft's speed as a mach number (dimensionless)
    """
    return true_air_speed / math.sqrt(
        constants.KAPPA * constants.R_GAS * temperature_from_fl(flight_level, delta_T=delta_T)
    )


@lru_cache(maxsize=10000)
def mach_to_tas(flight_level: float, mach: float, delta_T: float) -> float:
    """
    Converts the mach number to a true airspeed (TAS).
    See equation 3.1-26 in EEC Technical/Scientific Report No. 2010-003.

    Parameters
    ----------
    flight_level : float
        A flight level
    mach : float
        The mach number of the aircraft (dimensionless)
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)

    Returns
    -------
    float
        The aircraft's true air speed (in m/s)
    """
    return mach * math.sqrt(constants.KAPPA * constants.R_GAS * temperature_from_fl(flight_level, delta_T=delta_T))


@lru_cache(maxsize=10000)
def mach_cas_trans_altitude(mach: float, calibrated_air_speed: float, delta_T: float) -> float:
    """
    For given mach number and calibrated air speed find the pressure altitude at which these two quantities represent
    the same true air speed. See equations 3.1-27 to 3.1-29 in EIH Technical/Scientific Report No. 22/05/12-45.
    Note that the equations are augmented with the case where the transition altitude is above the tropopause.

    Parameters
    ----------
    mach : float
        The given mach number
    calibrated_air_speed : float
        The given calibrated air speed (in m/s)
    delta_T : float
        The BADA atmosphere temperature differential at a pressure
        altitude of zero (in K)

    Returns
    -------
    float
        The transition pressure altitude (in m)
    """
    kappa_prime = (constants.KAPPA - 1.0) / 2.0
    pressure_ratio = (
        (1.0 + kappa_prime * (calibrated_air_speed / constants.A_ZERO) ** 2) ** constants.MU_INV - 1.0
    ) / ((1.0 + kappa_prime * mach**2) ** constants.MU_INV - 1.0)

    P_trop = constants.P_ZERO * (1.0 + constants.BETA_T * constants.HP_TROP / constants.T_ZERO) ** constants.EXPO
    P_trans = constants.P_ZERO * pressure_ratio

    if P_trans < P_trop:
        T_isa_trop = constants.T_ZERO + constants.BETA_T * constants.HP_TROP
        transition_altitude = constants.HP_TROP - (constants.R_GAS * T_isa_trop / constants.G_ACC) * math.log(
            P_trans / P_trop
        )
    else:
        temperature_ratio = pressure_ratio ** (-1.0 * constants.BETA_T * constants.R_GAS / constants.G_ACC)
        transition_altitude = (temperature_ratio - 1.0) * (constants.T_ZERO + delta_T) / constants.BETA_T

    return transition_altitude


def horizontal_tas(tas: float, vertical_speed: float) -> float:
    """
    Give the total true air speed of an aircraft (in knots) and its vertical speed
    (in feet/min) return the horizontal true air speed (in knots)

    Parameters
    ----------
    tas : float
        Aircraft's total true air speed (in knots)
    vertical_speed : float
        Aircraft's vertical speed (in feet/min)

    Returns
    -------
    horizontal_tas : float
        Aircraft's horizontal true air speed (in knots)
    """
    return math.sqrt(tas**2 - (vertical_speed * FT_TO_NMI * 60.0) ** 2)


def ground_speed_from_tas(horizontal_tas: float, heading: float, wind_vector: WindVector | None) -> tuple[float, float]:
    """
    Returns aircraft's ground speed and ground track angle based on true airspeed and
    and current heading.

    The method used here is simple vector addition of the eastward and northward components
    of the wind vector and the aircraft velocity in air vector.

    Parameters
    ----------
    horizontal_tas : float
        Aircraft's horizontal true air speed (in knots)
    heading : float
        Aircraft's heading (in degrees east of north)
    wind_vector: WindVector
        A wind vector

    Returns
    -------
    ground_speed : float
        Ground speed of the aircraft (in knots)
    ground_track_angle : float
        Ground track angle of the aircraft (in degrees east of north)
    """
    if wind_vector is None:
        ground_speed = horizontal_tas
        ground_track_angle = heading

    else:
        ground_speed_eastward = wind_vector.u_comp * MPS_TO_KT + horizontal_tas * math.sin(math.radians(heading))
        ground_speed_northward = wind_vector.v_comp * MPS_TO_KT + horizontal_tas * math.cos(math.radians(heading))
        ground_speed = math.sqrt(ground_speed_eastward**2 + ground_speed_northward**2)
        ground_track_angle = math.degrees(math.atan2(ground_speed_eastward, ground_speed_northward)) % 360

    return ground_speed, ground_track_angle


def tas_from_ground_speed(ground_speed: float, ground_track_angle: float, wind_vector: WindVector | None) -> float:
    """
    Returns aircraft's true_airspeed based on the ground_speed,
    using simple vector subtraction of the wind from the ground speed

    Parameters
    ----------
    ground_speed: float
        The ground speed of the aircraft in knots.
    ground_track_angle: float
        The ground track angle of the aircraft in degrees.
    wind_vector: WindVector
        A wind vector

    Returns
    -------
    true_airspeed : float
        True airspeed of the aircraft (in knots)
    """
    if wind_vector is None:
        true_airspeed = ground_speed
    else:
        true_airspeed_eastward = (
            ground_speed * math.sin(math.radians(ground_track_angle)) - wind_vector.u_comp * MPS_TO_KT
        )
        true_airspeed_northward = (
            ground_speed * math.cos(math.radians(ground_track_angle)) - wind_vector.v_comp * MPS_TO_KT
        )

        true_airspeed = math.sqrt(true_airspeed_eastward**2 + true_airspeed_northward**2)
    return true_airspeed


def heading_from_ground_track(
    ground_track_angle: float,
    horizontal_tas: float,
    wind_vector: WindVector | None,
) -> float:
    """
    Returns an aircraft's required heading based on a required ground track angle. The derivation
    of the equation used here takes the following definitions:

    - G = the ground velocity vector
    - V = the aircraft horizontal velocity in air vector
    - W = the wind vector
    - u = eastward wind component
    - v = northward wind component
    - correction_angle = heading - ground_track_angle
    - x is the vector cross product operator
    - * is the usual scalar multiplication symbol/operator

    Now taking the vector cross product of G with the LHS and RHS of the vector equation:

    G = V + W    -- (Eq. 1)

    we have:

    0 = G x V + G x W   or   G x V = - G x W

    rewriting then gives:
    |G| * |V| sin(correction_angle) = v * |G| * sin(ground_track_angle) -  u * |G| * cos(ground_track_angle)

    finally simplifying gives:
    sin(correction_angle) = (v * sin(ground_track_angle) -  u * cos(ground_track_angle)) / |V|   -- (Eq. 2)

    Note: Equation 2 is only valid of course if Equation 1 holds. For a given ground track angle it is not
    always physically possible to satisfy equation 1 if the wind speed is too large compared to the
    true air speed. This case correspond to the magnitude of the RHS of Equation 2 being greater than 1.
    Here the default behaviour in this case is to set the heading equal to the wind direction, i.e. the
    aircraft is to fly directly into the wind.

    Parameters
    ----------
    ground_track_angle : float
        The desired ground track angle for the aircraft (in degrees east of north)
    horizontal_tas : float
        Aircraft's true air speed (in knots)
    wind_vector: WindVector
        A wind vector

    Returns
    -------
    Float :
        The required heading for a given requested ground track angle
    """
    if wind_vector is None:
        required_heading = ground_track_angle

    else:
        u_comp, v_comp = wind_vector.u_comp, wind_vector.v_comp
        sine_of_correction_angle = (
            v_comp * math.sin(math.radians(ground_track_angle)) - u_comp * math.cos(math.radians(ground_track_angle))
        ) / (horizontal_tas * KT_TO_MPS)

        # Wind speed is too large to achieve required ground track angle, then fly directly into the wind
        if abs(sine_of_correction_angle) > 1.0:
            required_heading = wind_vector.direction_wind_from

        else:
            correction_angle = math.degrees(math.asin(sine_of_correction_angle))
            required_heading = (ground_track_angle + correction_angle) % 360

    return required_heading


T = typing.TypeVar("T")


def nan_to_none(value: T) -> T | None:  # noqa: ANN001, ANN201
    return None if pd.isna(value) else value


# A shadow type annotation to convert a pandas NaN representation to None during serialisation or initialisation. See https://github.com/pydantic/pydantic/issues/1779#issuecomment-1524062596.
CouldBePandasNone = typing.Annotated[T | None, BeforeValidator(nan_to_none)]
