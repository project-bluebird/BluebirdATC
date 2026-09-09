import numpy as np
import pandas as pd
import pytest

from bluebird_dt.airspace_generator import SectorI, SectorX, SectorY, Thunderdome
from bluebird_dt.core import Airspace, Environment, Fixes, Pos2D, Route

@pytest.fixture
def generate_i():
    width = 60
    height = 120
    alt_limits = (60, 400)
    airspace, routes = SectorI(width, height, alt_limits).generate_airspace()

    return airspace, routes


@pytest.fixture
def generate_i_fl300_to_fl310():
    width = 20
    height = 120
    alt_limits = (300, 310)
    airspace, routes = SectorI(width, height, alt_limits).generate_airspace()

    return airspace, routes


@pytest.fixture
def generate_i_bent_route():
    width = 20
    height = 120
    alt_limits = (300, 310)
    airspace, routes = SectorI(width, height, alt_limits).generate_airspace()

    # shift boundary fixes a bit to the right/left to get a route that isn't straight line
    airspace.fixes.places["EARTH"] = Pos2D(*airspace.fixes.places["EARTH"].location - np.array([0, 0.1]))
    airspace.fixes.places["AIR"] = Pos2D(*airspace.fixes.places["AIR"].location - np.array([0, -0.1]))

    return airspace, routes


@pytest.fixture
def generate_x():
    width = 40
    height = 120
    alt_limits = (60, 400)
    alpha = 45
    airspace, routes = SectorX(width, height, alt_limits, alpha).generate_airspace()

    return airspace, routes


@pytest.fixture
def generate_y():
    width = 60
    height = 120
    alt_limits = (60, 400)
    alpha = 45
    airspace, routes = SectorY(width, height, alt_limits, alpha).generate_airspace()

    return airspace, routes

@pytest.fixture
def generate_thunderdome():
    radius = 60
    alt_limits = (60, 400)
    num_inner = 4
    num_outer = 5
    airspace, routes = Thunderdome(radius, alt_limits, num_inner, num_outer).generate_airspace()

    return airspace, routes

@pytest.fixture
def generate_two_sector():
    """
    Example I shaped two Sector Airspace with varying FL limits in each.
    One Sector is narrower than the other.
    """

    # airspace1 span: lon = [-0.5, 0.5], lats = [-2, 0], fl = [60, 400]
    # airspace2 span: lon = [-0.5, 0.5], lats = [0, 2], fl = [100, 300]
    width = 60.0
    height = 120.0
    airspace1, _ = SectorI(width, height, (60, 400)).generate_airspace()
    airspace2, _ = SectorI(width, height, (100, 300)).generate_airspace()

    # 0.495 here ensures that the polygons still touch; 0.5 causes their union to be disjoint
    airspace1.forward(distance=height * 0.495, heading=180)
    airspace2.forward(distance=height * 0.495, heading=0)

    # combine and rename fixes, get 7 in total
    # - FIX0 and FIX6 are outside the airspace
    # - FIX1 and FIX5 are on the boundary (entry/exit)
    # - FIX3 is in the centre
    # - FIX2 and FIX4 are 1/4 or 3/4 inside the airspace
    fixes1 = {f"FIX{i}": fix[1] for i, fix in enumerate(airspace2.fixes.places.items()) if i <= 2}
    fixes2 = {f"FIX{i + 2}": fix[1] for i, fix in enumerate(airspace1.fixes.places.items()) if i != 0}
    fixes = Fixes({**fixes2, **fixes1})

    # goes through airspace2 then airspace1 (lat move 2.3 to -2.3 )
    route1 = [f"FIX{i}" for i in range(7)]
    # goes through airspace1 then airspace 2 (lat move from -2.3 to 2.3)
    route2 = route1[::-1]

    routes = [Route(route1), Route(route2)]

    new_airspace = Airspace(
        sectors={"sector_i1": airspace1.sectors["sector_i"], "sector_i2": airspace2.sectors["sector_i"]}, fixes=fixes
    )

    return new_airspace, routes


@pytest.fixture
def generate_two_sector_offset():
    """
    Example I shaped two Sector Airspace with varying FL limits in each.
    One Sector FL range goes higher than the other (with 50FL overlap).
    """

    width = 60.0
    height = 120.0
    airspace1, _ = SectorI(width, height, (60, 300)).generate_airspace()
    airspace2, _ = SectorI(width, height, (250, 400)).generate_airspace()

    # 0.495 here ensures that the polygons still touch; 0.5 causes their union to be disjoint
    airspace1.forward(distance=height * 0.495, heading=180)
    airspace2.forward(distance=height * 0.495, heading=0)

    # combine and rename fixes, get 7 in total
    # - FIX0 and FIX6 are outside the sector
    # - FIX1 and FIX5 are on the boundary (entry/exit)
    # - FIX3 is in the centre
    # - FIX1 and FIX5 are 1/4 or 3/4 inside the sector
    fixes1 = {f"FIX{i}": fix[1] for i, fix in enumerate(airspace2.fixes.places.items()) if i <= 2}
    fixes2 = {f"FIX{i + 2}": fix[1] for i, fix in enumerate(airspace1.fixes.places.items()) if i != 0}
    all_fixes = {**fixes2, **fixes1}
    fixes = Fixes(all_fixes)

    route1 = list(all_fixes.keys())
    route2 = route1[::-1]
    routes = [Route(route1), Route(route2)]

    new_airspace = Airspace(
        sectors={"sector_i1": airspace1.sectors["sector_i"], "sector_i2": airspace2.sectors["sector_i"]}, fixes=fixes
    )

    return new_airspace, routes


@pytest.fixture
def generate_simple_environment():
    """
    Environment with an I sector and two Aircraft.
    """

    environment = Environment.from_json(
        """
        {
            "time": 0.0,
            "start_time": 0.0,
            "time_str": "1970-01-01T00:00:00",
            "airspace": {
                "sectors": {
                    "sector_i": {
                        "volumes": [
                            {
                                "area": [
                                  [ 0.8, -0.16666666666666666 ],
                                  [ -0.8, -0.16666666666666666 ],
                                  [ -0.8, 0.16666666666666666 ],
                                  [ 0.8, 0.16666666666666666 ]
                                ],
                                "min_fl": 70,
                                "max_fl": 400
                            }
                        ],
						"area_of_responsibility": null
                    }
                },
                "fixes": { 
                    "places": {
                          "SPIRIT": { "lat": 1.0, "lon": 0.0 },
                          "AIR": { "lat": 0.8, "lon": 0.0 },
                          "WATER": { "lat": 0, "lon": 0 },
                          "EARTH": { "lat": -0.8, "lon": 0.0 },
                          "FIRE": { "lat": -1.0, "lon": 0.0 }
                        }
                    },
                "airspace_configuration": { "sector_i": ["sector_i"] },
                "individual_sectors": {
                    "sector_i": {
                        "volumes": [
                            {
                                "area": [
                                  [0.8, -0.16666666666666666 ],
                                  [-0.8, -0.16666666666666666 ],
                                  [-0.8, 0.16666666666666666 ],
                                  [0.8, 0.16666666666666666 ]
                                ],
                                "min_fl": 70,
                                "max_fl": 400
                            }
                        ],
						"area_of_responsibility": null
                    }
		}
            },
            "aircraft": {
                "AIR0": {
                    "flight_plan": {
                        "route": {
                                "current": ["FIRE", "EARTH", "AIR", "SPIRIT"],
                                "filed": ["FIRE", "EARTH", "AIR", "SPIRIT"]
                        },
                        "unexpanded_route": "FIRE-EARTH-AIR-SPIRIT",
                        "origin": "EGSS",
                        "dest": "EGPF",
                        "milcivil": "C",
                        "sector_crossing_seq": "None-sector_i-None",
                        "requested_flight_level": 340,
                        "filed_true_airspeed": 400
                    },
                    "callsign": "AIR0",
                    "lat": -1.0,
                    "lon": 0.0,
                    "fl": 150,
                    "heading": 0.0,
                    "speed_tas": 360,
                    "vertical_speed": 0,
                    "aircraft_type": "B738",
                    "curr_sector": "sector_i",
                    "pilot": {
                        "pilot_type": "Pilot",
                        "callsign": "AIR0"
                    }
                },
                "AIR1": {
                    "flight_plan": {
                        "route": {
                            "current": ["SPIRIT", "AIR", "EARTH", "FIRE"],
                            "filed": ["SPIRIT", "AIR", "EARTH", "FIRE"]
                        },
                        "unexpanded_route": "SPIRIT-AIR-EARTH-FIR",
                        "origin": "EGPF",
                        "dest": "EGSS",
                        "milcivil": "C",
                        "sector_crossing_seq": "sector2",
                        "requested_flight_level": 340,
                        "filed_true_airspeed": 400
                    },
                    "callsign": "AIR1",
                    "lat": 1.0,
                    "lon": 0.0,
                    "fl": 200,
                    "heading": 180.0,
                    "speed_tas": 360,
                    "vertical_speed": 0,
                    "aircraft_type": "NULL",
                    "current_sector": "sector_i",
                    "pilot": {
                        "pilot_type": "Pilot",
                        "callsign": "AIR1"
                    }
                }
            },
            "wind_field": null,
            "coordinations": [
                {
                     "callsign": "AIR0",
                     "from_sector": null,
                     "to_sector": "sector_i",
                     "fl": 200,
                     "fix": "SPIRIT",
                     "direction": "Horizontal",
                     "level_by": false,
                     "level_by_details": null,
                     "secondary_coord_conditions": null,
                     "datetime": "1970-01-01 00:02:00"
                },
                {
                     "callsign": "AIR0",
                     "from_sector": "sector_i",
                     "to_sector": null,
                     "fl": 200,
                     "fix": "FIRE",
                     "direction": "Horizontal",
                     "level_by": false,
                     "level_by_details": null,
                     "secondary_coord_conditions": null,
                     "datetime": "1970-01-01 00:18:00"
                },
                {
                     "callsign": "AIR1",
                     "from_sector": null,
                     "to_sector": "sector_i",
                     "fl": 150,
                     "fix": "FIRE",
                     "direction": "Horizontal",
                     "level_by": false,
                     "level_by_details": null,
                     "secondary_coord_conditions": null,
                     "datetime": "1970-01-01 00:02:00"
                },
                {
                     "callsign": "AIR1",
                     "from_sector": "sector_i",
                     "to_sector": null,
                     "fl": 300,
                     "fix": "SPIRIT",
                     "direction": "Horizontal",
                     "level_by": false,
                     "level_by_details": null,
                     "secondary_coord_conditions": null,
                     "datetime": "1970-01-01 00:18:00"
                }
            ]
        }
    """
    )
    return environment

@pytest.fixture
def conditional_volumes(generate_two_sector):
    init_airspace, routes = generate_two_sector
    sector_1 = init_airspace.sectors["sector_i1"]
    sector_2 = init_airspace.sectors["sector_i2"]
    vol1 = sector_1.volumes[0]
    vol1.description = "AIR_SPIRIT"
    vol2 = sector_2.volumes[0]
    vol2.description = "FIRE_WATER"
    return vol1, vol2

@pytest.fixture
def stereographic_projection_test_vals():
    return np.asarray([
        #   LAT         LON     X       Y
        [51.01133, 0.02758, 34357.5, -443946.0],
        [52.68470, 0.20144, 44842.0, -257523.0],
        [54.38651, 0.32763, 51260.5, -68004.0],
        [53.23376, 0.34443, 53819.5, -196303.5],
        [50.94335, 0.25139, 50159.0, -451385.5],
        [51.94852, 0.25331, 49174.5, -339448.5],
        [51.40415, 0.64326, 76942.0, -399718.0],
        [52.94173, 0.66254, 75581.5, -228520.0],
        [52.23031, 0.59021, 71890.5, -307789.0],
        [52.13222, 0.81240, 87274.5, -318460.5],
        [55.45223, -12.33230, -748352.0, 114254.0],
        [48.61844, -9.00104, -630569.0, -673707.5]
    ])