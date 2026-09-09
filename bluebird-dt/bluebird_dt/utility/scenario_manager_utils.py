from numpy.random import Generator

from bluebird_dt.core import Airspace, Pos2D, Route
from bluebird_dt.utility import geometry


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
    # geometry.get_perpendicular_line(a,b) returns start and end points of a line
    # perpendicular to the line from a to b, centred on b.
    start, _, _ = geometry.get_perpendicular_line(second_fix, first_fix)
    # Calculate the heading from the outer (spawning) fix along this perpendicular line in one direction
    heading_1 = airspace.geo_helper.bearing_to(
        lat=start[0],
        lon=start[1],
        lat_origin=first_fix.lat,
        lon_origin=first_fix.lon,
    )
    # calculate the opposite heading
    heading_2 = (heading_1 + 180.0) % 360.0

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
