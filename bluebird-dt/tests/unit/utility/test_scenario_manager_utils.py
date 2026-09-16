import numpy as np
import pytest
import random

from bluebird_dt.core import Aircraft, Coordination, Pos3D, Route
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
            assert abs(angle - 90) < 1 # allow 1 degree variation
