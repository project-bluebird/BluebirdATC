from typing import Literal, TypeVar

from numpy.random import Generator

from bluebird_dt.core import Aircraft, Airspace, Coordination, FlightPlan, Pos2D, Pos3D, Route
from bluebird_dt.utility import geometry

TAircraft = TypeVar("TAircraft", bound=Aircraft)


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


def create_aircraft_with_coordinations(
    callsign: str,
    pos: Pos3D,
    heading: float,
    speed: float,
    route: Route,
    sector_name: str,
    entry_fl: float,
    exit_fl: float,
    airspace: Airspace,
    on_route: bool = False,
    prev_sector: str = "background",
    next_sector: str = "background",
    coord_direction: Literal["Horizontal", "Down", "Up"] = "Horizontal",
    typeof_aircraft: type[TAircraft] = Aircraft,
) -> tuple[TAircraft, Coordination, Coordination]:
    """
    Create an Aircraft instance, and entry and exit coordinations,
    given the necessary input parameters.

    Parameters
    ----------
    callsign: str
        callsign of the aircraft being created.
    pos: Pos3D
        position (lat, lon, FL) at which aircraft will be created.
    heading: float
        initial heading for the aircraft
    speed: float
        initial speed for the aircraft
    route: Route
        sequence of fixes on aircraft's route.
    sector_name: str
        name of the sector for entry and exit coordinations
    entry_fl: float
        Flight Level for entry coordination
    exit_fl: float
        Flight Level for exit coordination
    airspace: Airspace
        specify the airspace containing the sector being controlled.
    on_route: bool
        Setting for  "on_route" flag on created aircraft. Default is False.
    prev_sector: str
        "from_sector" for entry coordination. Default is "background".
    next_sector: str
        "to_sector" for exit coordination. Default is "background".
    coord_direction: Literal["Horizontal", "Down", "Up"]
        whether the coordination is horizontal or vertical.  Default is "Horizontal".
    typeof_aircraft: type[TAircraft]
        if derived class of Aircraft is to be created, specify here.

    Returns
    =======
    tuple[TAircraft, Coordination, Coordination]
        Newly instantiated Aircraft, with Entry Coordination and Exit Coordination.
    """
    # find the entry and exit fixes for the coordinations.
    entry_fix, exit_fix = find_entry_exit_fixes(airspace, route, sector_name)

    # create entry and exit coordinations
    coordination_entry = Coordination(
        callsign=callsign,
        from_sector=prev_sector,
        to_sector=sector_name,
        fl=entry_fl,
        fix=entry_fix,
        direction=coord_direction,
    )

    coordination_exit = Coordination(
        callsign=callsign,
        from_sector=sector_name,
        to_sector=next_sector,
        fl=exit_fl,
        fix=exit_fix,
        direction=coord_direction,
    )

    # generate the Aircraft instance
    flight_plan = FlightPlan(route)
    aircraft = typeof_aircraft(
        pos.lat,
        pos.lon,
        pos.fl,
        heading,
        flight_plan,
        callsign,
        selected_fl=int(pos.fl),
        current_sector=sector_name,
    )
    aircraft.speed_tas = speed
    aircraft.selected_instructions.cas = speed
    aircraft.on_route = on_route
    aircraft.simulated = True

    return aircraft, coordination_entry, coordination_exit


def find_entry_exit_fixes(airspace: Airspace, route: Route, sector_name: str) -> tuple[str, str]:
    """
    Given a route and a sector name, find the fixes to use
    for entry and exit coordinations.

    Parameters
    ==========
    airspace: Airspace
        the airspace under consideration
    route: Route
        the aircraft's route.
    sector_name: str
        name of the controlling sector
    Returns
    =======
    entry_fix, exit_fix: tuple[str, str]
        The fix names for entry and exit coordinations
    """
    sector = airspace.sectors[sector_name]
    entry_fix, exit_fix = None, None

    # If route starts inside the sector, take entry_fix to be first fix on route
    if sector.contains_laterally(airspace.fixes.places[route.filed[0]]):
        entry_fix = route.filed[0]
    # If route ends inside the sector, take exit_fix to be last fix on route
    if sector.contains_laterally(airspace.fixes.places[route.filed[-1]]):
        exit_fix = route.filed[-1]

    if entry_fix and exit_fix:
        return entry_fix, exit_fix
    # If we start or end outside the sector, find fixes on the boundary
    sector_boundary = sector.boundary()

    if not entry_fix:
        # start from the beginning of the route
        for fix in route.filed:
            fix_pos = airspace.fixes.places[fix]
            if sector_boundary.on_boundary(fix_pos):
                entry_fix = fix
                break
        # if no match, revert to first fix in route
        entry_fix = entry_fix if entry_fix is not None else route.filed[0]
    if not exit_fix:
        # start from the end of the route
        for fix in reversed(route.filed):
            fix_pos = airspace.fixes.places[fix]
            if sector_boundary.on_boundary(fix_pos) and fix is not entry_fix:
                exit_fix = fix
                break
        # if no match, use last fix in route
        exit_fix = exit_fix if exit_fix is not None else route.filed[-1]
    return entry_fix, exit_fix
