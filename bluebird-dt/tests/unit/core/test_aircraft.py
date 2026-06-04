import random
import logging

import pytest

from bluebird_dt.core import Aircraft, FlightPlan, Pos2D, Pilot, Route
from bluebird_dt.utility.convert import FL_TO_FT


def test_init_exceptions():
    """
    Test Aircraft initialization raises ValueError when invalid parameter values are used.
    """

    lat, lon = (50.7351, -3.4153)
    fl = 120
    heading = 66.7  # [deg]
    callsign = "AIR0"
    pilot = Pilot(callsign)
    flight_plan = FlightPlan(Route(["a", "b", "c"]))

    with pytest.raises(ValueError):
        Aircraft(lat, lon, fl, heading, flight_plan, callsign, pilot="incorrect_pilot")

    aircraft_callsign = "AIR0"
    incorrect_callsign = "incorrect_callsign"
    pilot_with_incorrect_callsign = Pilot(incorrect_callsign)
    assert pilot_with_incorrect_callsign.callsign == incorrect_callsign

    with pytest.raises(ValueError):
        Aircraft(
            lat,
            lon,
            fl,
            heading,
            flight_plan,
            aircraft_callsign,
            pilot=pilot_with_incorrect_callsign,
        )


def test_data():
    """
    Test correct data is returned (e.g., handling of default parameters).
    """

    lat, lon = (50.7351, -3.4153)
    flight_level = 120
    heading = 66.7  # [deg]
    operation_params = {}
    random_seed = 5

    route = Route(["ALFA", "BRAVO", "CHAR"])
    intention_code = "IM"
    flight_plan = FlightPlan(route, intention_code=intention_code)
    callsign = "AIR0"

    a = Aircraft(lat, lon, flight_level, heading, flight_plan, callsign)
    data = a.data()

    # data() should have a field for each class parameter
    assert len(data.keys()) == len(a.__dict__.keys())

    # check default values for not-specified parameters are correct
    assert data["squawk"] is None
    assert data["wake_vortex"] == "M"
    assert data["heading_changing_to"] is None
    assert data["next_fix_index"] == 1
    assert data["ufid"] is None
    assert len(data["operation_params"]) == 0
    assert len(data["predictor_params"]) == 0
    assert data["current_sector"] == "background"
    assert data["previous_sector"] == "background"
    assert data["rate_of_turn"] is None
    assert data["aircraft_type"] == "B753"
    assert data["controllable"] is True
    assert data["simulated"] is True

    assert data["selected_instructions"]["heading"] is None
    assert data["selected_instructions"]["fl"] == flight_level
    assert data["selected_instructions"]["on_route"] is True
    assert data["selected_instructions"]["cas"] is None
    assert data["selected_instructions"]["mach"] is None
    assert data["selected_instructions"]["vertical_speed"] is None
    assert data["cleared_instructions"]["heading"] is None
    assert data["cleared_instructions"]["fl"] == data["fl"]
    assert data["cleared_instructions"]["on_route"] is True
    assert data["cleared_instructions"]["cas"] is None
    assert data["cleared_instructions"]["mach"] is None
    assert data["cleared_instructions"]["vertical_speed"] is None
    assert data["speed_tas"] is None
    assert data["vertical_speed"] == pytest.approx(0.0)
    assert data["ground_speed"] is None
    assert data["ground_track_angle"] is None
    assert data["random_seed"] is None
    assert all(value is None for value in data["percentile_rank_dict"].values())
    assert data["pilot"]["pilot_type"] == "Pilot"
    assert data["pilot"]["callsign"] == "AIR0"

    aircraft_type = "A"
    cleared_heading = 60
    cleared_fl = 200
    cleared_cas = 500.0
    cleared_mach = 0.7
    cleared_vertical_speed = 15.0 * FL_TO_FT
    vertical_speed = 9.0
    on_route = False
    controllable = False
    simulated = False
    selected_fl = 220
    ufid = "20170101-45624-AIR0"
    rate_of_turn = 2.2
    current_sector = "07"
    squawk = "5563"
    wake_vortex = "M"

    a = Aircraft(
        lat=lat,
        lon=lon,
        fl=flight_level,
        heading=heading,
        flight_plan=flight_plan,
        callsign=callsign,
        selected_fl=selected_fl,
        ufid=ufid,
        rate_of_turn=rate_of_turn,
        aircraft_type=aircraft_type,
        operation_params=operation_params,
        controllable=controllable,
        simulated=simulated,
        random_seed=random_seed,
        current_sector=current_sector,
        squawk=squawk,
        wake_vortex=wake_vortex,
    )

    a.cleared_fl = cleared_fl
    a.selected_fl = cleared_fl
    a.cleared_instructions.heading = cleared_heading
    a.selected_instructions.heading = cleared_heading
    a.cleared_instructions.cas = cleared_cas
    a.selected_instructions.cas = cleared_cas
    a.cleared_instructions.mach = cleared_mach
    a.selected_instructions.mach = cleared_mach
    a.cleared_instructions.vertical_speed = cleared_vertical_speed
    a.selected_instructions.vertical_speed = cleared_vertical_speed
    a.cleared_instructions.on_route = on_route
    a.on_route = on_route
    a.vertical_speed = vertical_speed

    data = a.data()

    # check set parameter values
    assert data["aircraft_type"] == "A" # Aircraft type should be as supplied
    assert data["controllable"] == controllable
    assert data["simulated"] == simulated
    assert data["squawk"] == squawk
    assert data["wake_vortex"] == wake_vortex
    assert data["ufid"] == ufid
    assert data["current_sector"] == current_sector
    assert data["rate_of_turn"] == rate_of_turn

    # check attributes
    assert data["cleared_instructions"]["heading"] == cleared_heading
    assert data["cleared_instructions"]["fl"] == cleared_fl
    assert data["cleared_instructions"]["cas"] == cleared_cas
    assert data["cleared_instructions"]["mach"] == cleared_mach
    assert data["cleared_instructions"]["vertical_speed"] == cleared_vertical_speed
    assert data["cleared_instructions"]["on_route"] == on_route
    assert data["selected_instructions"]["heading"] == cleared_heading
    assert data["selected_instructions"]["fl"] == cleared_fl
    assert data["selected_instructions"]["cas"] == cleared_cas
    assert data["selected_instructions"]["mach"] == cleared_mach
    assert data["selected_instructions"]["vertical_speed"] == cleared_vertical_speed
    assert data["selected_instructions"]["on_route"] == on_route
    assert data["speed_tas"] is None
    assert data["vertical_speed"] == vertical_speed
    assert data["ground_speed"] is None
    assert data["ground_track_angle"] is None

    # Check randomly initialised attributes
    random.seed(random_seed)
    cas_percentile_rank = random.uniform(0, 100.0)
    rocd_percentile_rank = random.uniform(0, 100.0)
    assert data["random_seed"] == random_seed
    lateral_keys = ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl"]
    assert all(data["percentile_rank_dict"][key] == cas_percentile_rank for key in lateral_keys)
    assert all(data["percentile_rank_dict"][key] == rocd_percentile_rank for key in ["rocd_des", "rocd_cl"])


def test_aircraft_from_json(example_aircraft_json_no_pilot):
    """
    Test default Pilot is instantiated when "pilot" key not in json string.
    """

    json_string = example_aircraft_json_no_pilot

    aircraft = Aircraft.from_json(json_string)

    assert hasattr(aircraft, "pilot")
    assert isinstance(aircraft.pilot, Pilot)
    assert aircraft.pilot.callsign == aircraft.callsign


def test_aircraft_pilot_callsigns(example_aircraft_pilot):
    """
    Test that Aircraft callsigns are equal to the Pilot callsigns.
    """

    json_string = example_aircraft_pilot

    aircraft = Aircraft.from_json(json_string)
    callsign = "AIR0"
    assert aircraft.callsign == callsign
    assert aircraft.pilot.callsign == callsign

def test_set_attributes(caplog: pytest.LogCaptureFixture):
    """
    Test that set_attributes only allows setting 'fl', 'heading', and 'controllable'.
    """

    lat, lon = (50.7351, -3.4153)
    flight_level = 120
    heading = 66.7  # [deg]
    callsign = "AIR0"
    flight_plan = FlightPlan(Route(["ALFA", "BRAVO", "CHAR"]))

    aircraft = Aircraft(lat, lon, flight_level, heading, flight_plan, callsign)

    # Valid attributes
    attributes = {"fl": 130, "heading": 90.0, "controllable": False}
    aircraft.set_attributes(attributes)

    assert aircraft.fl == 130
    assert aircraft.heading == 90.0
    assert aircraft.controllable is False

    # Invalid attribute
    invalid_attributes = {"ground_speed": 500.0}
    with caplog.at_level(logging.WARN):
        aircraft.set_attributes(invalid_attributes)

        # Check that a warning was raised
        assert caplog.record_tuples[0] == (
            "bluebird_dt.logger", 
            logging.WARN, 
            "Attribute ground_speed is not allowed to be set via set_attributes."
        )
    
    # Ensure the invalid attribute was not set
    assert aircraft.ground_speed is None
    
def test_randomise_performance():
    """
    Test randomised performance method has the correct behaviour
    """

    lat, lon = (50.7351, -3.4153)
    flight_level = 120
    heading = 66.7  # [deg]
    random_seed = 5

    route = Route(["ALFA", "BRAVO", "CHAR"])
    flight_plan = FlightPlan(route)
    callsign = "AIR0"

    # Create aircraft with non-randomised performance characteristics
    a = Aircraft(lat, lon, flight_level, heading, flight_plan, callsign)

    # Check random seed and performance characteristic values are set to None for this
    # aircraft
    assert a.random_seed is None
    assert all([value is None for value in a.percentile_rank_dict.values()])

    # Call randomise speed performance method and check
    a.randomise_performance(random_seed)
    assert a.random_seed == random_seed
    lateral_keys = ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl"]
    assert all([a.percentile_rank_dict[key] == 62.29016948897019 for key in lateral_keys])
    assert all([a.percentile_rank_dict[key] == 74.17869892607294 for key in ["rocd_des", "rocd_cl"]])

    # De-randomise speed performance and check the performance characteristic values and
    # random seed is set back to None
    a.randomise_performance(None)
    assert a.random_seed is None
    assert all([value is None for value in a.percentile_rank_dict.values()])


def test_set_performance():
    """
    This speed performance setter method
    """
    lat, lon = (50.7351, -3.4153)
    flight_level = 120
    heading = 66.7  # [deg]

    route = Route(["ALFA", "BRAVO", "CHAR"])
    flight_plan = FlightPlan(route)
    callsign = "AIR0"

    # Create aircraft with default speed performance characteristics
    a = Aircraft(lat, lon, flight_level, heading, flight_plan, callsign)

    # Call setter method with no arguments, and check all speed percentile ranks are
    # not set
    a.set_performance()
    assert all([value is None for value in a.percentile_rank_dict.values()])

    # Call setter method to only set CAS percentile rank, and check values as correctly set
    cas_pr = 20.0
    a.set_performance(cas_pr=cas_pr)
    lateral_keys = ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl"]
    assert all([a.percentile_rank_dict[key] == cas_pr for key in lateral_keys])
    assert all([a.percentile_rank_dict[key] is None for key in ["rocd_des", "rocd_cl"]])

    # Call setter method to only set ROCD percentile rank, and check values as correctly set
    rocd_pr = 80.0
    a.set_performance(rocd_pr=rocd_pr)
    assert all([a.percentile_rank_dict[key] is None for key in lateral_keys])
    assert all([a.percentile_rank_dict[key] == rocd_pr for key in ["rocd_des", "rocd_cl"]])

    # Call setter method to set all percentile ranks, and check values as correctly set
    cas_pr = 95.0
    rocd_pr = 45.0
    a.set_performance(cas_pr=cas_pr, rocd_pr=rocd_pr)
    assert all([a.percentile_rank_dict[key] == cas_pr for key in lateral_keys])
    assert all([a.percentile_rank_dict[key] == rocd_pr for key in ["rocd_des", "rocd_cl"]])

    # Check an exception is raised when trying to use an invalid percentile rank
    with pytest.raises(ValueError):
        a.set_performance(cas_pr=-34.5)
    with pytest.raises(ValueError):
        a.set_performance(rocd_pr=150.5)


def test_distance_to_abeam(example_aircraft_pilot):
    """
    Test distance_to_abeam method
    """
    aircraft = Aircraft.from_json(example_aircraft_pilot)

    aircraft.lat = 50.0
    aircraft.lon = 0.0
    aircraft.ground_track_angle = 0.0

    location1 = Pos2D(50.8, 0.0)  # Directly to the north
    distance_to_loc1 = aircraft.distance(location1)

    # If the aircraft is heading directly towards the location, the distance to abeam should be the same as the distance
    assert aircraft.distance_to_abeam(location1) == pytest.approx(distance_to_loc1)

    # If the aircraft is heading directly away from the location, the distance_to_abeam should be the same, but negative
    aircraft.ground_track_angle = 180.0

    ret = aircraft.distance_to_abeam(location1)
    assert ret is not None and ret < 0
    assert aircraft.distance_to_abeam(location1) == pytest.approx(-distance_to_loc1)

    location2 = Pos2D(50.8, 0.3)  # Directly to the east of location1
    distance_to_loc2 = aircraft.distance(location2)
    loc1_to_loc2 = location1.distance(location2)

    aircraft.ground_track_angle = 0.0

    # Distance to abeam location2 should be approx the same as the distance to location1
    # Need to increase the tolerance as distance_to_abeam doesn't properly take into account the curvature of the earth
    assert aircraft.distance_to_abeam(location2) == pytest.approx(distance_to_loc1, rel=1e-3)

    # Use Pythagoras to check the three distances are consistent
    ret = aircraft.distance_to_abeam(location2)
    assert ret is not None
    assert ret**2 + loc1_to_loc2**2 == pytest.approx(distance_to_loc2**2, rel=1e-3)

    # If the aircraft won't pass closer than 20NM from the point, distance_to_abeam should return None
    location3 = Pos2D(50.8, 0.6)
    loc1_to_loc3 = location1.distance(location3)

    assert loc1_to_loc3 > 20.0
    assert aircraft.distance_to_abeam(location3) is None


def test_set_current_sector(example_aircraft_pilot):
    """
    Check that setting the current sector sets the previous sector to the old current sector
    """
    aircraft = Aircraft.from_json(example_aircraft_pilot)

    # Set an initial sector
    aircraft.current_sector = initial_sector = "S1"

    # Update current sector
    aircraft.current_sector = "S2"

    assert aircraft.previous_sector == initial_sector
