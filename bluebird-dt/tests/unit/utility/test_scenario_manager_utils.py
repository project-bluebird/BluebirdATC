import numpy as np
import pytest

from bluebird_dt.core import Aircraft, Coordination, Pos3D, Route
from bluebird_dt.utility.scenario_manager_utils import (
    laterally_offset_start_point,
    create_aircraft_with_coordinations,
    find_entry_exit_fixes,
)

def test_laterally_offset_start_point(generate_i):
    airspace, routes = generate_i
    rng = np.random.default_rng()
    offset_range = (5, 10)
    for route in routes:
        orig_start_point = airspace.fixes.places[route.filed[0]]
        next_fix = airspace.fixes.places[route.filed[1]]
        orig_heading = orig_start_point.bearing_to(next_fix)
        for _ in range(10):
            new_start_point = laterally_offset_start_point(
                airspace=airspace,
                route=route,
                offset_range=(5,10),
                rng=rng
            )
            # check that distance from orig to offset start point is in range
            d = orig_start_point.distance(new_start_point)
            assert offset_range[0] < d < offset_range[1]
            # check that the heading is +/- 90 degrees from orig_heading
            perp_heading = new_start_point.bearing_to(orig_start_point)
            angle = (perp_heading - orig_heading) % 180
            assert abs(angle - 90) < 1 # allow 1 degree variation


@pytest.mark.parametrize("callsign, pos, heading, speed, route, sector_name, entry_fl, exit_fl", [
    ("AIR-0", Pos3D(51.0, 0.5, 260.), 180., 350., Route(filed=["FIX1", "FIX2"], current=["FIX1", "FIX2"]), "test_1", 260, 300),
    ("AIR-1", Pos3D(0.0, 0.0, 300.), 300., 250., Route(filed=["FIX2", "FIX3"], current=["FIX2", "FIX3"]), "test_2", 260, 300),
])
def test_create_aircraft_with_coordinations(callsign, pos, heading, speed, route, sector_name, entry_fl, exit_fl):
    """
    Test the function that creates aircraft and entry and exit coordinations.
    """
    aircraft, coord_entry, coord_exit = create_aircraft_with_coordinations(
        callsign=callsign,
        pos=pos,
        heading=heading,
        speed=speed,
        route=route,
        sector_name=sector_name,
        entry_fl=entry_fl,
        exit_fl=exit_fl
    )
    assert isinstance(aircraft, Aircraft)
    assert aircraft.callsign == callsign
    assert aircraft.lat == pos.lat
    assert aircraft.lon == pos.lon
    assert aircraft.heading == heading
    assert aircraft.speed_tas == speed
    assert aircraft.selected_instructions.cas == speed
    assert aircraft.flight_plan.route == route
    assert isinstance(coord_entry, Coordination)
    assert coord_entry.fl == entry_fl
    assert coord_entry.from_sector == "background"
    assert coord_entry.to_sector == sector_name
    assert isinstance(coord_exit, Coordination)
    assert coord_exit.fl == exit_fl
    assert coord_exit.from_sector == sector_name
    assert coord_exit.to_sector == "background"


def test_find_entry_exit_fixes(generate_i):
    """
    Test the function that returns fixes nearest the entry and exit of 
    a route through a sector.
    """
    airspace, routes = generate_i
    # first route goes from "FIRE" to "SPIRIT", should enter sector at "EARTH"
    # and exit at "AIR"
    entry_fix, exit_fix = find_entry_exit_fixes(airspace, routes[0], "sector_i")
    assert entry_fix == "EARTH"
    assert exit_fix == "AIR"
    # second route should be the reverse
    entry_fix, exit_fix = find_entry_exit_fixes(airspace, routes[1], "sector_i")
    assert entry_fix == "AIR"
    assert exit_fix == "EARTH"