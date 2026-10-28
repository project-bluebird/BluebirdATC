import numpy as np
import pytest
import random

from bluebird_dt.airspace_generator import SectorI
from bluebird_dt.core import Aircraft, Airspace, Coordination, Pos3D, Route
from bluebird_dt.utility import geometry
from bluebird_dt.utility.scenario_manager_utils import (
    allowed_flight_levels,
    laterally_offset_start_point,
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
            assert abs(angle - 90) < 1e-2


def test_lateral_headings_reject_coincident_fixes(generate_i: tuple[Airspace, list[Route]]):
    airspace, routes = generate_i
    first, second = (airspace.fixes.places[name] for name in routes[0].filed[:2])
    second.lat, second.lon = first.lat, first.lon
    with pytest.raises(ValueError, match="start and end points must be different"):
        _, _ = geometry.get_perpendicular_headings(first, second, airspace.geo_helper)


@pytest.mark.parametrize(
    "sector_limits, fl_limits, expected",
    [
        ((215, 660), (50, 400), range(220, 410, 10)),
        ((45, 395), (50, 400), range(50, 400, 10)),
        ((60, 400), (55, 395), range(60, 400, 10)),
    ],
)
def test_allowed_flight_levels(sector_limits, fl_limits, expected):
    """
    The levels should be the multiples of 10 within both the sector's range and fl_limits.
    """
    airspace, _ = SectorI(60, 120, sector_limits).generate_airspace()
    assert list(allowed_flight_levels(airspace, "sector_i", fl_limits)) == list(expected)


def test_no_allowed_flight_levels(generate_i):
    """
    fl_limits outside the sector's range should raise an error naming the sector and the limits.
    """
    airspace, _ = generate_i
    with pytest.raises(ValueError, match=r"sector_i.*\(410, 500\)"):
        allowed_flight_levels(airspace, "sector_i", (410, 500))
