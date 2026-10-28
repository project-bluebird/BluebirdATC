import numpy as np
import pytest
import random

from bluebird_dt.core import Aircraft, Airspace, Coordination, Pos3D, Route
from bluebird_dt.utility import geometry
from bluebird_dt.utility.scenario_manager_utils import (
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
