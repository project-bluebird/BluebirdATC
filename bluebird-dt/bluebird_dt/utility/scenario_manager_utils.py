import math

import numpy as np
from numpy.random import Generator

from bluebird_dt.core import Airspace, Pos2D, Route
from bluebird_dt.utility import geometry


def allowed_flight_levels(airspace: Airspace, sector_name: str, fl_limits: tuple[int, int]) -> np.ndarray:
    """
    Return the flight levels at which aircraft can enter or exit a sector: the multiples of 10 within both
    the sector's range, over all its Volumes, and fl_limits.

    Parameters
    ==========
    airspace: Airspace
        the current airspace.
    sector_name: str
        the name of the sector in the airspace.
    fl_limits: tuple[int, int]
        min and max flight levels at which aircraft can enter or exit.

    Returns
    =======
    flight_levels: np.ndarray, the allowed flight levels as floats, in ascending order.

    Raises
    ======
    ValueError
        if no multiple of 10 lies within both the sector's range and fl_limits.
    """
    # Sector.get_bounds returns the lower and upper [lon, lat, fl] bounds over all the sector's Volumes.
    sector_min_fl, sector_max_fl = [b[2] for b in airspace.sectors[sector_name].get_bounds()]
    # Clip the sector's range to fl_limits, then round the lower bound up and the upper bound down to the nearest
    # multiple of 10 so that both stay inside the clipped range, e.g. FL215-FL395 becomes FL220-FL390.
    min_fl = 10 * math.ceil(max(sector_min_fl, fl_limits[0]) / 10)
    max_fl = 10 * math.floor(min(sector_max_fl, fl_limits[1]) / 10)
    if min_fl > max_fl:
        raise ValueError(
            f"No flight levels for sector {sector_name}: its range FL{sector_min_fl:g}-FL{sector_max_fl:g} "
            f"and fl_limits {fl_limits} have no multiple of 10 in common."
        )
    return np.arange(min_fl, max_fl + 10, 10, dtype="float")


def laterally_offset_start_point(
    airspace: Airspace,
    route: Route,
    offset_range: tuple[float, float],
    rng: Generator,
) -> Pos2D:
    """
    Given a route, return a position laterally offset by a random amount from the first fix.

    Parameters
    ==========
    airspace: Airspace
        the current airspace, including fix locations.
    route: Route
        an aircraft route (a series of Fixes)
    offset_range: tuple[float, float]
        min and max lateral offsets from route centre-line
    rng: Generator
        the random number generator to use, which may have random seed set for reproducibility.

    Returns
    =======
    spawn_point: Pos2D, a position laterally offset from the first fix of a route.
    """
    first_fix_name, second_fix_name = route.filed[:2]
    first_fix = airspace.fixes.places[first_fix_name]
    second_fix = airspace.fixes.places[second_fix_name]
    # calculate headings perpendicular to the line from first to second fix.
    heading_1, heading_2 = geometry.get_perpendicular_headings(first_fix, second_fix, airspace.geo_helper)
    # pick one of the two directions
    offset_heading = float(rng.choice((heading_1, heading_2)))
    # random offset distance
    offset_distance = rng.uniform(offset_range[0], offset_range[1])

    lon, lat = airspace.geo_helper.forward(
        first_fix.lon,
        first_fix.lat,
        heading=offset_heading,
        distance=offset_distance,
    )
    return Pos2D(lat=lat, lon=lon)
