import functools
import json

import numpy as np

from bluebird_dt.core.aircraft import FlightState
from bluebird_dt.logger import logger
from bluebird_dt.utility.paths import (
    AIRCRAFT_WEIGHT_MAPPING_FILE,
    SIMPLE_PERFORMANCE_PROFILE_FILE,
    SIMPLE_PERFORMANCE_UNCERTAINTY_FILE,
)
from bluebird_dt.utility.sample import apply_rocd_uncertainty, apply_speed_uncertainty


@functools.cache
def get_performance_table(
    path: str | None,
) -> dict[str, dict[str, list[float]]]:
    """
    Load calibrated airspeed, rate of climb or descend and associated uncertainty tables
    for all aircraft

    Parameters
    ----------
    path: str
        Path to speed table data

    Return
    ------
    Dict:
        Dictionary containing speed profile data
    """
    if path is None:
        path = SIMPLE_PERFORMANCE_PROFILE_FILE

    try:
        with open(path) as aircraft_speed_profile:
            return json.load(aircraft_speed_profile)["aircraft"]

    except FileNotFoundError as e:
        raise FileNotFoundError("Speed profile data file could not be found!") from e


@functools.cache
def get_performance_uncertainty_table(
    path: str | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Load speed uncertainty tables for all aircraft

    Parameters
    ----------
    path: str
        Path to speed table data

    Return
    ------
    Dict:
        Dictionary containing speed uncertainty data
    """
    if path is None:
        path = SIMPLE_PERFORMANCE_UNCERTAINTY_FILE

    try:
        with open(path) as aircraft_speed_uncertainty:
            return json.load(aircraft_speed_uncertainty)["aircraft"]

    except FileNotFoundError as e:
        raise FileNotFoundError("Speed uncertainty data file could not be found!") from e


@functools.cache
def get_aircraft_key_mapping(path: str | None = None) -> dict[str, str]:
    """
    Load aircraft synonym type data. If no path supplied, fall back to a default aircraft weight mapping
    that matches the fallback simple performance and uncertainty files.

    The synonym file will be a 1:1 mapping of the ~1800 ICAO aircraft codes and the
    fallback file is a simplified version which maps aircraft types to weight categories (and in some cases
    weight categories to weight categories).

    Parameters
    ----------
    path: str
        Path to file that maps aircraft type to a lookup key for performance data.

    Return
    ------
    Dict:
        Dictionary containing aircraft type mapping
    """
    if path is None:
        path = AIRCRAFT_WEIGHT_MAPPING_FILE
    try:
        with open(path) as synonym_database:
            return json.load(synonym_database)

    except FileNotFoundError as e:
        raise FileNotFoundError("Aircraft synonym data file could not be found!") from e


def cas_and_mach_from_table(
    aircraft_speed_table: dict[str, list[float | None] | float],
    speed_uncertainty_table: dict[str, dict[str, float]] | None,
    percentile_rank_table: dict[str, float | None] | None,
    aircraft_fl: float,
    aircraft_flight_state: FlightState,
) -> tuple[float, float]:
    """
    Returns the predicted aircraft CAS and mach number using linear interpolation of the CAS and mach table data.

    Parameters
    ----------
    aircraft_speed_table: dict[str, list[float | None] | float]
        Speed table of the aircraft
    speed_uncertainty_table: dict[str, dict[str, float]] | None
        Uncertainty table of the aircraft
    percentile_rank_table: dict[str, float | None] | None
        Percentile rank table of the aircraft
    airacrft_fl: float
        Flight level of the aircraft
    aircraft_flight_state: FlightState
        Flight state of the aircraft

    Returns
    -------
    Tuple:
        Tuple of two floats which are the predicted CAS (in knots) and the mach number, in that order
    """

    flight_levels = aircraft_speed_table["flight_level"]
    min_fl = flight_levels[0]
    max_fl = flight_levels[-1]

    # behave appropriately for each edge case of fl vs available speed profile flight levels
    # NOTE: fl_index corresponds to the lower of the two indices in the flight level array that bounds
    # the flight level, and interp_weight to how far along the fl is between the two flight levels indexed
    # by fl_index and fl_index + 1
    # 1. the aircraft is at a level greater than (or equal to) the table, so use the last table value
    if aircraft_fl >= max_fl:
        # idx -2 -> -1 (penultimate -> last), weighted fully at -1
        fl_index = -2
        interp_weight = 1.0

    # 2. fl that is below or equal to the table, so use the first value
    elif aircraft_fl <= min_fl:
        # idx 0 -> 1, weighted fully at 0
        fl_index = 0
        interp_weight = 0.0

    # 3. within the min/max fl bounds we have data for
    else:
        fl_index = np.where(np.array(flight_levels) <= aircraft_fl)[0][-1]
        interp_weight = (aircraft_fl - flight_levels[fl_index]) / (
            flight_levels[fl_index + 1] - flight_levels[fl_index]
        )

    state_labels = {FlightState.CRUISE: "cr", FlightState.CLIMB: "cl", FlightState.DESCEND: "des"}
    label = state_labels[aircraft_flight_state]
    cas_key = f"cas_{label}"
    mach_key = f"mach_{label}"

    cas_lo = aircraft_speed_table[cas_key][fl_index]
    cas_hi = aircraft_speed_table[cas_key][fl_index + 1]
    mach_values = aircraft_speed_table.get(mach_key)
    if isinstance(mach_values, list):
        mach_lo = mach_values[fl_index]
        mach_hi = mach_values[fl_index + 1]
    else:
        mach_lo = None
        mach_hi = None

    # work out the two edge cases. NOTE: Nones only occur before valid values, like:
    # [None, None, ..., None, float, float, ..., float]
    # 1. we're at the lower boundary, so set cas to be the high value
    if cas_lo is None and cas_hi is not None:
        cas = cas_hi

    # 2. else they are both None, so just choose the lowest value for which we have data
    elif cas_lo is None and cas_hi is None:
        cas = next(value for value in aircraft_speed_table[cas_key] if value is not None)
        logger.debug(
            f"No CAS data available at given flight level {aircraft_fl:.1f}. "
            f"Will use value ({cas}) from closest upper flight level.",
            stacklevel=2,
        )
    else:
        cas = (1.0 - interp_weight) * cas_lo + interp_weight * cas_hi

        # If percentile rank has been specified, use the uncertainty data to draw a speed score
        # from the speed probability distribution.
        if (
            percentile_rank_table is not None
            and cas_key in percentile_rank_table
            and percentile_rank_table[cas_key] is not None
            and speed_uncertainty_table is not None
            and cas_key in speed_uncertainty_table
            and speed_uncertainty_table[cas_key] is not None
        ):
            speed_uncertainty = speed_uncertainty_table[cas_key]
            percentile_rank = percentile_rank_table[cas_key]

            cas = apply_speed_uncertainty(cas, speed_uncertainty, percentile_rank)

    # same edge cases as for cas
    if mach_lo is None and mach_hi is not None:
        mach = mach_hi

    elif mach_lo is None and mach_hi is None:
        if isinstance(mach_values, list):
            mach = next((value for value in mach_values if value is not None), None)
            if mach is not None:
                logger.debug(
                    f"No mach data available for at given flight level {aircraft_fl:.1f}. "
                    f"Will use value ({mach}) from closest upper flight level.",
                    stacklevel=2,
                )
        else:
            mach = None

    else:
        mach = (1.0 - interp_weight) * mach_lo + interp_weight * mach_hi

    if (
        mach is not None
        and percentile_rank_table is not None
        and speed_uncertainty_table is not None
        and mach_key in speed_uncertainty_table
        and speed_uncertainty_table[mach_key] is not None
    ):
        percentile_rank = percentile_rank_table.get(mach_key)
        if percentile_rank is None:
            percentile_rank = percentile_rank_table.get(cas_key)

        mach = apply_speed_uncertainty(mach, speed_uncertainty_table[mach_key], percentile_rank)

    return cas, mach


def rocd_from_table(
    aircraft_speed_table: dict[str, list[float | None]],
    speed_uncertainty_table: dict[str, dict[str, float]] | None,
    percentile_rank_table: dict[str, float | None] | None,
    aircraft_fl: float,
    aircraft_flight_state: FlightState,
) -> float:
    """
    Return the predicted aircraft vertical speed using linear interpolation on the ROCD table data

    Parameters
    ----------
    aircraft_speed_table: dict[str, list[float | None]]
        Speed table of the aircraft
    speed_uncertainty_table: dict[str, dict[str, float]] | None
        Uncertainty table of the aircraft
    percentile_rank_table: dict[str, float | None] | None
        Percentile rank table of the aircraft
    airacrft_fl: float
        Flight level of the aircraft
    aircraft_flight_state: FlightState
        Flight state of the aircraft

    Returns
    -------
    Float:
        The predicted vertical speed (in feet per minute)
    """

    flight_levels = aircraft_speed_table["flight_level"]
    min_fl = flight_levels[0]
    max_fl = flight_levels[-1]

    # behave appropriately for each edge case of fl vs available speed profile flight levels
    # NOTE: fl_index corresponds to the lower of the two indices in the flight level array that bounds
    # the flight level, and interp_weight to how far along the fl is between the two flight levels indexed
    # by fl_index and fl_index + 1
    # 1. the aircraft is at a level greater than (or equal to) the table, so use the last table value
    if aircraft_fl >= max_fl:
        # idx -2 -> -1 (penultimate -> last), weighted fully at -1
        fl_index = -2
        interp_weight = 1.0

    # 2. fl that is below or equal to the table, so use the first value
    elif aircraft_fl <= min_fl:
        # idx 0 -> 1, weighted fully at 0
        fl_index = 0
        interp_weight = 0.0

    # 3. within the min/max fl bounds we have data for
    else:
        fl_index = np.where(np.array(flight_levels) <= aircraft_fl)[0][-1]
        interp_weight = (aircraft_fl - flight_levels[fl_index]) / (
            flight_levels[fl_index + 1] - flight_levels[fl_index]
        )

    state_labels = {FlightState.CRUISE: "cr", FlightState.CLIMB: "cl", FlightState.DESCEND: "des"}
    label = state_labels[aircraft_flight_state]
    rocd_key = f"rocd_{label}"

    if rocd_key != "rocd_cr":
        rocd_data = aircraft_speed_table[rocd_key]

        if isinstance(rocd_data, list):
            vspeed_lo = rocd_data[fl_index]
            vspeed_hi = rocd_data[fl_index + 1]

            if vspeed_lo is None and vspeed_hi is not None:
                vertical_speed = vspeed_hi

            elif vspeed_lo is None and vspeed_hi is None:
                vertical_speed = next(value for value in rocd_data if value is not None)
                logger.debug(
                    f"No ROCD data available at given flight level {aircraft_fl:.1f}. "
                    f"Will use value ({vertical_speed}) from closest upper flight level.",
                    stacklevel=2,
                )

            else:
                vertical_speed = (1.0 - interp_weight) * vspeed_lo + interp_weight * vspeed_hi

        else:
            vertical_speed = float(rocd_data)

        # If percentile rank has been specified, use the uncertainty data to draw a rocd score
        # from the rocd probability distribution.
        if (
            percentile_rank_table is not None
            and rocd_key in percentile_rank_table
            and percentile_rank_table[rocd_key] is not None
            and speed_uncertainty_table is not None
            and rocd_key in speed_uncertainty_table
            and speed_uncertainty_table[rocd_key] is not None
        ):
            rocd_uncertainty = speed_uncertainty_table[rocd_key]
            percentile_rank = percentile_rank_table[rocd_key]

            vertical_speed = apply_rocd_uncertainty(vertical_speed, rocd_uncertainty, percentile_rank)

        # Vertical speed during descend must be negative
        if rocd_key == "rocd_des":
            vertical_speed = -1.0 * vertical_speed

    else:
        vertical_speed = 0

    return vertical_speed
