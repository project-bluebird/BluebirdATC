import copy
import logging
import random

import numpy as np
import pytest

from bluebird_dt.core import Action, Environment, Fixes, FlightState, Pos2D, Aircraft, FlightPlan, Route
from bluebird_dt.core.wind import WindField
from bluebird_dt.predictor import SimplePredictor
from bluebird_dt.scenario_manager.regular import Regular
from bluebird_dt.simulator.simulator import Simulator
from bluebird_dt.utility.convert import (
    FL_TO_FT,
    FT_TO_FL,
)
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS


def test_init_exceptions():
    """
    Test ValueError is raised if invalid arguments are used when initiating class.
    """

    with pytest.raises(ValueError):
        SimplePredictor(0, 1, Fixes({}))

    with pytest.raises(ValueError):
        SimplePredictor(1, 0, Fixes({}))


@pytest.mark.parametrize("callsign, fix_name", [("AIR0", "EARTH"), ("AIR1", "AIR")])
def test_get_target_pos(callsign: str, fix_name: str, generate_simple_environment: Environment):
    """
    Test method returns position of the next Fix on Route.
    """

    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    selected_aircraft = environment.aircraft[callsign]
    selected_aircraft.selected_instructions.cas = random.randint(300,600)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    _ = predictor.predict_aircraft(selected_aircraft, 1.0, environment_time=environment.time)

    pos = predictor.get_target_pos(selected_aircraft)
    assert pos.__str__() == environment.airspace.fixes.places[fix_name].__str__()


@pytest.mark.parametrize(
    "callsign, action_kind, action_value, heading, cleared_heading, check_heading",
    [
        ("AIR0", "change_heading_by", -90, 0, 270, 358.5),
        ("AIR0", "maintain_current_heading", 0, 0, 0, 0),
        ("AIR1", "maintain_current_heading", 0, 180, 180, 180),
        ("AIR1", "change_heading_to", 45, 180, 45, 178.5),
        ("AIR1", "route_direct_to", "AIR", 180, None, 180),
    ],
)
def test_process_lateral_actions(
    callsign: str,
    action_kind: str,
    action_value: float | str,
    heading: float | None,
    cleared_heading: float | None,
    check_heading: float,
    generate_simple_environment: Environment,
):
    """
    Test `heading` and `selected_instructions.heading` are appropriately updated. Also, check `on_route` behaviour is
    tracked correctly when lateral Actions are issued.
    """

    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    # Get an aircraft and check it is on_route
    aircraft = environment.aircraft[callsign]
    aircraft.selected_instructions.cas = random.randint(300,600)
    aircraft.rate_of_turn = 1.5  # fix in case default value changes
    pilot = aircraft.pilot

    assert aircraft.on_route

    # Create and process an action
    action = Action(callsign, action_kind, action_value)
    pilot.process_lateral_actions(action, environment)

    # Check the heading of the initial aircraft is correct
    assert aircraft.selected_instructions.heading == cleared_heading
    assert aircraft.heading == heading

    # Check that the on_route flags have been set correctly
    if action_kind == "route_direct_to" or action_kind == "change_cas_to":
        assert aircraft.on_route
    else:
        assert not aircraft.on_route

    # Call predict_aircraft and check returned aircraft
    predictor_aircraft = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)

    assert predictor_aircraft.heading == check_heading
    assert predictor_aircraft.selected_instructions.heading == cleared_heading

@pytest.mark.parametrize(
    "callsign, fl, cleared_fl, check_flight_state, cleared_vspeed, check_vspeed",
    [
        ("AIR1", 200, 200, "cruise", None, 0.0),
        ("AIR1", 210, 210, "cruise", 21 * FL_TO_FT, 0.0),
        ("AIR1", 200, 300, "climb", None, 2000.0),
        ("AIR1", 200, 250, "climb", 23 * FL_TO_FT, 2300),
        ("AIR1", 200, 170, "descend", None, -2000.0),
        ("AIR1", 200, 180, "descend", 25 * FL_TO_FT, -2500.0),
    ],
)

def test_update_aircraft_vertical_speeds_method(
    callsign: str,
    fl: float,
    cleared_fl: float,
    check_flight_state: str,
    cleared_vspeed: float,
    check_vspeed: float,
    generate_simple_environment: Environment,
):
    """
    Test that the correct vertical speed is set, either from aircraft.selected_instructions.vertical_speed, or from the
    speed profile data tables. Test the cruise, climb and descend cases. When read from the table, look for the key
    "rocd" for climb or descent cases ("rocd_cl" or rocd_des). Remember that table values are used when cleared values
    are None.
    """

    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)
    aircraft = environment.aircraft[callsign]
    aircraft.aircraft_type = "B753"

    aircraft.fl = fl
    aircraft.selected_fl = cleared_fl
    aircraft.selected_instructions.vertical_speed = cleared_vspeed
    aircraft.selected_instructions.cas = random.randint(300,600)
    flight_state = aircraft.flight_state
    assert flight_state.value == check_flight_state

    predictor_aircraft = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)

    predictor.update_vertical_speeds(predictor_aircraft)
    assert predictor_aircraft.vertical_speed == pytest.approx(check_vspeed)

@pytest.mark.parametrize(
    "callsign, action_kind, action_value",
    [
        ("AIR0", "change_cas_to", 180.0),
        ("AIR0", "change_cas_to", 410.0),
        ("AIR1", "change_cas_to", 356.0),
        ("AIR1", "change_cas_to", 550.0),
    ],
)
def test_update_cas_as_tas_speeds(
    callsign: str,
    action_kind: str,
    action_value: float,
    generate_simple_environment: Environment,
):
    """
    Test cas_as_tas functions correctly when the cas_as_tas flag is set to True.
    """

    # Generate simple environment
    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)

    # Check fixture aircraft in the environment have speeds as expected on instantiation
    aircraft = environment.aircraft[callsign]
    pilot = aircraft.pilot

    # Set up a predictor using the cas_is_tas speed modelling
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    # Check that an exception is raised when cleared values are not set
    with pytest.raises(ValueError):
        _ = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time, deepcopy_aircraft=False)

    # Create and process an action
    action = Action(callsign, action_kind, action_value)
    pilot.process_speed_actions(action, environment)

    predictor_aircraft = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)

    assert predictor_aircraft.speed_tas == pytest.approx(action_value)


@pytest.mark.parametrize(
    "callsign, action_kind, action_value, vspeed, change_fl",
    [
        ("AIR0", "change_vertical_speed_to", 26 * FL_TO_FT, -26 * FL_TO_FT, -50),
        ("AIR0", "change_vertical_speed_to", 26 * FL_TO_FT, 0.0, 0),
        ("AIR0", "change_vertical_speed_to", 26 * FL_TO_FT, 26 * FL_TO_FT, 50),
    ],
)
def test_update_vertical_speeds(
    callsign: str,
    action_kind: str,
    action_value: float,
    vspeed: float,
    change_fl: float,
    generate_simple_environment: Environment,
):
    """
    Test that vertical speeds are updated when vertical actions are processed, and the update_speeds method is used.
    """

    # Generate a simple environment
    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    aircraft = environment.aircraft[callsign]
    aircraft.selected_instructions.cas = random.randint(300,600)
    pilot = aircraft.pilot

    # Create two actions: change_vertical_speed_to and a change_flight_level_by action
    action_1 = Action(callsign, action_kind, action_value)
    action_2 = Action(callsign, "change_flight_level_by", change_fl)
    action1_value = action_1.value
    action2_value = action_2.value

    assert isinstance(action1_value, (float, int))
    assert isinstance(action2_value, (float, int))

    # Get initial values for speeds
    initial_fl = aircraft.fl
    initial_vertical_speed = aircraft.vertical_speed
    assert initial_vertical_speed == 0.0
    assert aircraft.selected_instructions.vertical_speed is None
    assert aircraft.cleared_instructions.vertical_speed is None

    # Process a pair of vertical speed actions
    pilot.process_vertical_speed_actions(action_1, environment)
    assert aircraft.selected_instructions.vertical_speed == action1_value
    assert aircraft.cleared_instructions.vertical_speed == action1_value

    pilot.process_vertical_actions(action_2, environment)
    assert aircraft.selected_fl == initial_fl + action2_value
    assert aircraft.cleared_fl == initial_fl + action2_value

    # Call predict_aircraft and check the vertical speed of the aircraft has been updated
    _ = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time, deepcopy_aircraft=False)
    assert aircraft.vertical_speed == vspeed


@pytest.mark.parametrize(
    "callsign, action_kind, action_value, n_step",
    [
        ("AIR1", "change_heading_to", 45, 10),
        ("AIR0", "change_heading_by", -5, 15),
        ("AIR1", "maintain_current_heading", 0, 10),
        ("AIR0", "route_direct_to", "AIR", 30),
        ("AIR1", "change_heading_to", 45, 10),
        ("AIR0", "change_heading_by", -5, 15),
        ("AIR1", "maintain_current_heading", 0, 10),
        ("AIR0", "route_direct_to", "AIR", 30),
    ],
)
def test_update_position(
    callsign: str,
    action_kind: str,
    action_value: float | str,
    n_step: float,
    generate_simple_environment: Environment,
):
    """
    Test update position flag values are correct, considering turn_model_flag.
    """

    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    aircraft = environment.aircraft[callsign]
    aircraft.selected_instructions.cas = random.randint(300,600)
    pilot = aircraft.pilot

    action = Action(callsign, action_kind, action_value)
    pilot.process_lateral_actions(action, environment)

    if aircraft.heading != aircraft.selected_instructions.heading:
        assert aircraft.heading_changing_to is not None

    predictor_aircraft = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)

    if predictor_aircraft.heading == predictor_aircraft.selected_instructions.heading:
        assert predictor_aircraft.heading_changing_to is None


@pytest.mark.parametrize(
    "action_kind, action_value",
    [
        ("change_heading_to", 100),
        ("route_direct_to", "CCC"),  # route to NOT last fix. No turn required.
        ("route_direct_to", "DDD"),  # route to NOT last fix. Needs to turn.
        ("route_direct_to", "EEE"),  # route to last fix. Needs to turn.
    ],
)
def test_turn_radius_set(
    action_kind: str,
    action_value: float | str,
    generate_simple_environment: Environment,
):
    """
    Test the predictor_params turn_radius is set correctly.
    """
    callsign = "AIR0"
    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)

    fixes = Fixes({
        "AAA": Pos2D(lat=0.0, lon=-1.0),
        "BBB": Pos2D(lat=0.0, lon=-0.25),
        "CCC": Pos2D(lat=0.0, lon=0.0),
        "DDD": Pos2D(lat=0.5, lon=0.0),
        "EEE": Pos2D(lat=1.0, lon=0.0),
    })

    flight_plan = FlightPlan(route=Route(["AAA", "BBB", "CCC", "DDD", "EEE"]))
    environment.aircraft = {callsign: Aircraft(lat=0.0, lon=-0.5, fl=200, heading=90,
                                            flight_plan=flight_plan, callsign=callsign, aircraft_type="B753")}
    environment.airspace.fixes = fixes
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    aircraft = environment.aircraft[callsign]
    aircraft.selected_instructions.cas = 300

    # evolve once up update aircraft state. Evolve in place so that the environment updates
    predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time,
                                          deepcopy_aircraft=False)

    action = Action(callsign, action_kind, action_value)
    aircraft.pilot.process_lateral_actions(action, environment)

    aircraft = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time,
                                          deepcopy_aircraft=False)

    if action_kind == "change_heading_to":
        # turn_radius should be None
        assert aircraft.predictor_params["turn_radius"] is None
    elif action_kind == "route_direct_to":
        if action_value == "CCC":
            # no need to turn. Turn radius should be None.
            assert aircraft.predictor_params["turn_radius"] is None
        if action_value == "DDD":
            # needs to turn. Turn radius should not be None.
            assert aircraft.predictor_params["turn_radius"] is not None
        if action_value == "EEE":
            # needs to turn. Turn radius should not be None.
            assert aircraft.predictor_params["turn_radius"] is not None
    else:
        raise ValueError(f"Action kind {action_kind} is not valid.")


@pytest.mark.parametrize(
    "callsign, action_kind, action_value",
    [
        ("AIR0", "descend_when_ready,level_by_fix", (100, "AIR")),
        ("AIR0", "descend_now,level_by_fix", (120, "AIR")),
        ("AIR1", "descend_now,level_by_fix", (80, "EARTH")),
        ("AIR1", "descend_when_ready,level_by_fix", (50, "EARTH")),
    ],
)
def test_update_flight_level_with_level_by_fix_actions(
    callsign: str, action_kind: str, action_value: tuple[int, str], generate_simple_environment: Environment
):
    """
    Test level_by_fix actions are handled correctly
    """
    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    aircraft = environment.aircraft[callsign]
    pilot = aircraft.pilot
    aircraft.selected_instructions.cas = random.randint(300,600)
    action = Action(callsign, action_kind, action_value)
    assert action.kind in SUPPORTED_ACTIONS["vertical"]

    pilot.process_vertical_actions(action, environment)

    start_fl = aircraft.fl

    # Move aircraft and check fl has or hasn't changed, depending on action kind.
    delta_t = 6.0
    predictor.predict_aircraft(aircraft, delta_t, environment_time=environment.time, deepcopy_aircraft=False)

    if action.kind == "descend_now,level_by_fix":
        assert aircraft.fl == pytest.approx(start_fl + predictor.slow_descent_rate * FT_TO_FL * delta_t / 60)
    elif action.kind == "descend_when_ready,level_by_fix":
        assert aircraft.fl == start_fl
    else:
        # Raise error here so that if new actions are added or names changed, the test will get updated
        raise ValueError(f"Action is not a valid kind: {action.kind}")

    # Move aircraft until past the level_by fix and check fl is the target fl
    assert aircraft.next_fix_index is not None
    assert aircraft.flight_plan is not None
    while aircraft.flight_plan.route.current[aircraft.next_fix_index] not in ["SPIRIT", "FIRE"]:
        predictor.predict_aircraft(aircraft, delta_t, environment_time=environment.time, deepcopy_aircraft=False)

    assert aircraft.fl == pytest.approx(action_value[0])


@pytest.mark.parametrize(
    "callsign, action_kind, action_value, action_kind_2, action_value_2, check_cleared_fl, n_step",
    [
        ("AIR0", "change_flight_level_by", 0, "change_vertical_speed_to", 0.0, 150, 10),
        ("AIR1", "change_flight_level_by", -10, "change_vertical_speed_to", 0.0, 190, 10),
        ("AIR0", "change_flight_level_to", 160, "change_vertical_speed_to", 30.0 * FL_TO_FT, 160, 10),
        ("AIR1", "change_flight_level_to", 190, "change_vertical_speed_to", 25.0 * FL_TO_FT, 190, 10),
    ],
)
def test_update_flight_level(
    callsign: str,
    action_kind: str,
    action_value: float,
    action_kind_2: str,
    action_value_2: float,
    check_cleared_fl: float,
    n_step: float,
    generate_simple_environment: Environment,
):
    """
    Test flight level is updated correctly.
    """

    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)

    aircraft = environment.aircraft[callsign]
    aircraft.selected_instructions.cas = random.randint(300,600)
    pilot = aircraft.pilot

    action_1 = Action(callsign, action_kind, action_value)
    action_2 = Action(callsign, action_kind_2, action_value_2)

    for a in [action_1, action_2]:
        if a.kind in SUPPORTED_ACTIONS["vertical"]:
            pilot.process_vertical_actions(a, environment)
        elif a.kind in SUPPORTED_ACTIONS["vertical_speed"]:
            pilot.process_vertical_speed_actions(a, environment)

    # Check aircraft.flight_state is correct
    if aircraft.fl == aircraft.selected_fl:
        assert aircraft.flight_state is FlightState.CRUISE
    elif aircraft.fl < aircraft.selected_fl:
        assert aircraft.flight_state is FlightState.CLIMB
    else:
        assert aircraft.flight_state is FlightState.DESCEND

    start_fl = aircraft.fl

    # Call predict_aircraft and get predictor aircraft
    predictor_aircraft = predictor.predict_aircraft(aircraft, n_step, environment_time=environment.time)

    # Get values to check against
    climb_rate = (action_value_2 * FT_TO_FL) / 60
    check_vertical_speed = action_value_2

    flight_state = predictor_aircraft.flight_state
    if flight_state is FlightState.DESCEND:
        climb_rate *= -1
        check_vertical_speed *= -1

    check_fl = start_fl + n_step * climb_rate

    assert predictor_aircraft.selected_fl == pytest.approx(check_cleared_fl)
    assert predictor_aircraft.selected_instructions.vertical_speed == pytest.approx(action_value_2)
    assert predictor_aircraft.vertical_speed == pytest.approx(check_vertical_speed)

    if predictor_aircraft.flight_state is not FlightState.CRUISE:
        assert predictor_aircraft.fl == pytest.approx(check_fl)
    else:
        assert predictor_aircraft.fl == pytest.approx(start_fl)


@pytest.mark.parametrize(
    "callsign, action_kind, action_value",
    [
        ("AIR0", "change_heading_by", -90),
        ("AIR0", "maintain_current_heading", 0),
        ("AIR0", "maintain_current_heading", -10),  # value should make no difference
        ("AIR0", "maintain_current_heading", ""),  # value should make no difference
        ("AIR1", "change_heading_to", 45),
        ("AIR1", "route_direct_to", "AIR"),
        ("AIR0", "change_cas_to", 340.0),
        ("AIR1", "change_cas_to", 205.0),
        ("AIR0", "change_flight_level_by", -50),
        ("AIR1", "change_flight_level_to", 40),
        ("AIR0", "change_vertical_speed_to", 30.0 * FL_TO_FT),
        ("AIR1", "change_vertical_speed_to", 28.0 * FL_TO_FT),
        ("AIR1", "descend_when_ready,level_by_fix", (100, "EARTH")),
        ("AIR1", "descend_now,level_by_fix", (120, "AIR")),
        ("AIR0", "descend_now,level_by_fix", (80, "AIR")),
        ("AIR0", "descend_when_ready,level_by_fix", (50, "EARTH")),
    ],
)
def test_predict_aircraft(
    callsign: str,
    action_kind: str,
    action_value: float | str | tuple[int, str],
    generate_simple_environment: Environment,
):
    """
    Test correct trajectory is returned, taking actions into account.
    """

    # Randomly select a dt and n_step
    dt = random.choice([1.0, 3.0, 5.0])
    n_step = random.choice([15, 30, 45])

    # Set up environment, predictor and aircraft
    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(dt, 2.0, fixes=environment.airspace.fixes)

    selected_aircraft = environment.aircraft[callsign]
    pilot = selected_aircraft.pilot

    # Get initial values
    start_pos = selected_aircraft.pos2d()

    # Create and process action
    action = Action(callsign, action_kind, action_value)
    pilot.receive_actions([action], environment)
    pilot.process_actions(environment)

@pytest.mark.parametrize(
    "callsign, action_kind, action_value, delta_t, n_step, next_fix_index, on_route",
    [
        ("AIR1", "route_direct_to", "AIR", 5.0, 45, 0, True),
        ("AIR0", "change_cas_to", 340.0, 5.0, 200, 2, True),
        ("AIR1", "change_cas_to", 205.0, 5.0, 150, 1, True),
        ("AIR0", "change_flight_level_by", -50, 5.0, 70, 1, True),
        ("AIR1", "change_flight_level_to", 40, 5.0, 2000, None, False),
        ("AIR0", "change_vertical_speed_to", 30.0 * FL_TO_FT, 5.0, 300, 2, True),
        ("AIR1", "change_vertical_speed_to", 28.0 * FL_TO_FT, 5.0, 65, 1, True),
        ("AIR1", "descend_when_ready,level_by_fix", (100, "EARTH"), 5.0, 300, 2, True),
        ("AIR1", "descend_now,level_by_fix", (120, "AIR"), 5.0, 65, 1, True),
        ("AIR0", "descend_now,level_by_fix", (80, "AIR"), 5.0, 2000, None, False),
        ("AIR0", "descend_when_ready,level_by_fix", (50, "EARTH"), 5.0, 150, 2, True),
    ],
)
def test_predict_aircraft_on_route(
    callsign: str,
    action_kind: str,
    action_value: float | str | tuple[int, str],
    delta_t: float,
    n_step: int,
    next_fix_index: int | None,
    on_route: bool,
    generate_simple_environment: Environment,
):
    """
    Test if aircraft has correct next fix index and on_route flag
    at end of the trajectory, taking different actions into account.
    """

    # Set up environment, predictor and aircraft
    environment = generate_simple_environment
    environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
    predictor = SimplePredictor(delta_t, 2.0, fixes=environment.airspace.fixes)

    selected_aircraft = environment.aircraft[callsign]
    selected_aircraft.selected_instructions.cas = random.randint(300,600)
    pilot = selected_aircraft.pilot

    # Create and process action
    action = Action(callsign, action_kind, action_value)
    pilot.receive_actions([action], environment)
    pilot.process_actions(environment)

    # Test predictor aircraft has correct next_fix_index and on_route settings
    # at end of trajectory roll out
    predictor_aircraft = predictor.predict_aircraft(selected_aircraft, n_step, environment_time=environment.time)
    assert predictor_aircraft.next_fix_index == next_fix_index
    assert predictor_aircraft.on_route == on_route

def test_predict_aircraft_change_heading_to_by_direction():
    sim = Simulator.from_category(category="Two Aircraft", scenario_name="I-Sector")
    
    callsign = list(sim.manager.environment.aircraft.keys())[0]
    aircraft = sim.manager.environment.aircraft[callsign]
    aircraft.heading = 90

    sim.manager.receive_actions([Action(aircraft.callsign, "change_heading_to_by_direction", (10, 'right'))])

    sim.evolve(6)
    sim.evolve(6)
    sim.evolve(6)
    
    assert aircraft.heading > 90

    
def test_predict_route_direct_of_onroute_aircraft_with_uninitialised_parameters():
    """
    Test calling predict on an aircraft on route being instructed to fly route direct to a point, which is handled differently, doesn't raise an exception.
    """
    predictor = SimplePredictor(5, 2.0, Fixes({f"FIX{i}": Pos2D(0, 0) for i in [0, 2, 4]}))

    aircraft = Aircraft(0, 0, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [0, 2, 4]])), "AIR0", aircraft_type="B753")
    
    aircraft.selected_instructions.cas = 250
    aircraft.cleared_instructions.lateral_action = Action("AIR0", "route_direct_to", "FIX4")
    aircraft.on_route = True

    predictor.predict_aircraft(aircraft, 20)
    
