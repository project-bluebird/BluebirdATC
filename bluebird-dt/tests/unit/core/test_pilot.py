import logging
import pytest

from bluebird_dt.core import Action, Aircraft, Pilot, Pos2D, QueueItem


def test_init():
    """
    Test Pilot instantiates with correct given and default values.
    """

    callsign = "AIR0"
    pilot = Pilot(callsign)
    assert pilot.callsign == callsign
    assert pilot.action_queue == []

    with pytest.raises(TypeError):
        Pilot()


def test_pilot_queue_item():
    """
    Test the QueueItem class functions correctly (using a TypedDict).
    """

    action = Action("AIR0", "change_heading_by", 5)
    time = 0.5

    queue_item: QueueItem = {
        "receipt_time": time,
        "process_time": time,
        "action": action,
    }

    assert queue_item["action"].kind == "change_heading_by"
    assert queue_item["process_time"] == queue_item["receipt_time"]
    assert queue_item["process_time"] == 0.5


def test_pilot_instantiation(generate_simple_environment):
    """
    Check Pilots are instantiated when reading from json with no Pilot fields.
    """

    environment = generate_simple_environment
    for callsign, aircraft in environment.aircraft.items():
        # Check every aircraft has a Pilot
        assert hasattr(aircraft, "pilot")

        # Check pilot is a valid Pilot
        pilot = aircraft.pilot
        assert isinstance(pilot, Pilot)

        # Check every Pilot has the correct callsign
        assert pilot.callsign == callsign


def test_pilot_receive_valid_actions(generate_simple_environment):
    """
    Test Pilots correctly receive valid actions correctly, and they get added to a queue.
    """

    environment = generate_simple_environment
    callsign = "AIR0"
    pilot = environment.aircraft[callsign].pilot

    # Check action queue starts off empty
    assert len(pilot.action_queue) == 0

    # Create action and issue to the pilot
    new_action = Action("AIR0", "change_heading_by", 5)
    ignored_actions = pilot.receive_actions([new_action], environment)

    # Check queue is now 1
    assert len(pilot.action_queue) == 1

    # Check queue item has correct attributes
    queue_item = pilot.action_queue[0]
    assert queue_item["action"].kind == "change_heading_by"
    assert queue_item["receipt_time"] == queue_item["process_time"]

    # Check that the action was not ignored
    assert len(ignored_actions) == 0

    # Add two more actions to the queue and check queue length
    more_actions = [Action("AIR0", "change_heading_by", 27), Action("AIR0", "change_flight_level_to", 147)]
    pilot.receive_actions(more_actions, environment)
    assert len(pilot.action_queue) == 3

    # Check all actions have the same receipt_time
    action_receipt_times = [action["receipt_time"] for action in pilot.action_queue]
    assert len(set(action_receipt_times)) == 1


def test_check_actions_pilot_callsign(generate_simple_environment):
    """
    Test that Pilots must have a callsign before processing actions.
    """

    environment = generate_simple_environment
    callsign = "AIR0"
    selected_aircraft = environment.aircraft[callsign]

    # Check aircraft has a Pilot and it has the correct callsign
    assert hasattr(selected_aircraft, "pilot")
    assert selected_aircraft.pilot.callsign == callsign


def test_check_callsign_of_actions(generate_simple_environment):
    """
    Test that an error is thrown if Pilots receive actions intended for other callsigns.
    """

    environment = generate_simple_environment
    callsign = "AIR0"
    selected_aircraft = environment.aircraft[callsign]

    # Create action for another callsign and check that an error is thrown
    action = Action(callsign="another_aircraft", kind="change_heading_by", value=10)

    with pytest.raises(ValueError):
        selected_aircraft.pilot.check_actions([action], environment)


def test_check_controllable_action_processing(generate_simple_environment, caplog: pytest.LogCaptureFixture):
    """
    Check that Pilots ignore actions if Aircraft.controllable flag is False.
    """

    environment = generate_simple_environment
    callsign = "AIR0"
    selected_aircraft = environment.aircraft[callsign]

    # Ensure controllable flag is set to False, action is ignored, and warning is given
    selected_aircraft.controllable = False
    new_action = Action(callsign, "change_heading_by", 10)

    with caplog.at_level(logging.DEBUG):
        valid_actions, ignored_actions = selected_aircraft.pilot.check_actions([new_action], environment)
        assert len(ignored_actions) == 1
        assert len(valid_actions) == 0
        assert len(caplog.record_tuples) == 1

    # Check the opposite case with the flag == True
    selected_aircraft.controllable = True
    valid_actions, ignored_actions = selected_aircraft.pilot.check_actions([new_action], environment)
    assert len(valid_actions) == 1
    assert len(ignored_actions) == 0


def test_receive_check_process_single_action(generate_simple_environment):
    """
    Tests actions are received, checked, and processed correctly.
    """

    environment = generate_simple_environment
    callsign = "AIR1"
    aircraft = environment.aircraft[callsign]
    pilot = aircraft.pilot

    # HEADING
    new_action = Action(callsign, "change_heading_by", 95)
    ignored_actions = pilot.receive_actions([new_action], environment)

    assert len(ignored_actions) == 0
    assert len(pilot.action_queue) == 1

    # Process action. Check that the Aircraft attributes are modified correctly
    pilot.process_actions(environment)
    assert len(pilot.action_queue) == 0
    assert aircraft.cleared_instructions.heading == (aircraft.heading + 95) % 360
    assert aircraft.selected_instructions.heading == (aircraft.heading + 95) % 360
    assert aircraft.cleared_instructions.on_route is False
    assert aircraft.selected_instructions.on_route is False
    assert aircraft.cleared_instructions.lateral_action == new_action
    assert aircraft.selected_instructions.lateral_action == new_action

    # ROUTE DIRECT
    initial_length_of_route = len(aircraft.flight_plan.route.filed)
    third_fix_on_route = aircraft.flight_plan.route.filed[2]
    new_action = Action(callsign, "route_direct_to", [third_fix_on_route])  # central FIX
    ignored_actions = pilot.receive_actions([new_action], environment)

    assert len(ignored_actions) == 0
    assert len(pilot.action_queue) == 1

    # Process action. Check that the Aircraft attributes are modified correctly
    pilot.process_actions(environment)
    assert len(pilot.action_queue) == 0
    # filed route remains unchanged while current route includes ABYSS and fixes that follow it
    assert aircraft.flight_plan.route.current[0] == third_fix_on_route
    assert len(aircraft.flight_plan.route.filed) == initial_length_of_route
    # as we route_directed to the third fix, current flight plan should be 2 shorter
    assert len(aircraft.flight_plan.route.current) == initial_length_of_route - 2
    assert aircraft.next_fix_index == 0
    assert aircraft.cleared_instructions.on_route is True
    assert aircraft.selected_instructions.on_route is True
    assert aircraft.cleared_instructions.lateral_action == new_action
    assert aircraft.selected_instructions.lateral_action == new_action

    # VERTICAL
    new_action = Action(callsign, "change_flight_level_by", 50)
    ignored_actions = pilot.receive_actions([new_action], environment)

    assert len(ignored_actions) == 0
    assert len(pilot.action_queue) == 1

    # Process action. Check that the Aircraft attributes are modified correctly
    pilot.process_actions(environment)
    assert len(pilot.action_queue) == 0
    assert aircraft.cleared_instructions.fl == (aircraft.fl + 50)
    assert aircraft.selected_instructions.fl == (aircraft.fl + 50)
    assert aircraft.cleared_instructions.vertical_action == new_action
    assert aircraft.selected_instructions.vertical_action == new_action
    # Check can access flight level instructions correctly via properties
    assert aircraft.cleared_fl == aircraft.cleared_instructions.fl
    assert aircraft.selected_fl == aircraft.selected_instructions.fl

    # SPEED
    # own speed on mach
    new_action = Action(callsign, "change_mach_to", None)
    ignored_actions = pilot.receive_actions([new_action], environment)

    assert len(ignored_actions) == 0
    assert len(pilot.action_queue) == 1

    # set cas
    new_action = Action(callsign, "change_cas_to", 270)
    ignored_actions = pilot.receive_actions([new_action], environment)

    assert len(ignored_actions) == 0
    assert len(pilot.action_queue) == 2

    # Process action. Check that the Aircraft attributes are modified correctly
    pilot.process_actions(environment)
    assert len(pilot.action_queue) == 0
    assert aircraft.cleared_instructions.mach is None
    assert aircraft.selected_instructions.mach is None
    assert aircraft.cleared_instructions.cas == 270
    assert aircraft.selected_instructions.cas == 270

    # OUTCOMM, using
    initial_sector = aircraft.current_sector
    new_action = Action(callsign, "outcomm", "background")
    ignored_actions = pilot.receive_actions([new_action], environment)

    assert len(ignored_actions) == 0
    assert len(pilot.action_queue) == 1

    # Process action. Check that the Aircraft attributes are modified correctly
    pilot.process_actions(environment)
    assert len(pilot.action_queue) == 0
    assert aircraft.previous_sector == initial_sector
    # only one named sector in generate_simple_environment so new current sector is "background"
    assert aircraft.current_sector == "background"


def test_hold_action_and_lateral_cancellation(generate_simple_environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    hold_fix = next(iter(environment.airspace.fixes.places))
    hold_action = Action(
        aircraft.callsign,
        "route_direct_to,hold_at_location",
        {"fix": hold_fix, "outbound_time": 45, "turn_direction": "left"},
    )

    aircraft.pilot.receive_actions([hold_action], environment)
    aircraft.pilot.process_actions(environment)

    assert aircraft.predictor_params["hold"] == {
        "fix": hold_fix,
        "location": [
            environment.airspace.fixes.places[hold_fix].lat,
            environment.airspace.fixes.places[hold_fix].lon,
        ],
        "outbound_time": 45.0,
        "turn_direction": "left",
        "phase": "direct_to_location",
        "phase_elapsed": 0.0,
        "inbound_track": None,
    }
    assert aircraft.on_route is False

    aircraft.pilot.receive_actions([Action(aircraft.callsign, "maintain_current_heading", 0)], environment)
    aircraft.pilot.process_actions(environment)

    assert "hold" not in aircraft.predictor_params


def test_hold_rejects_unknown_fix(generate_simple_environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    action = Action(aircraft.callsign, "route_direct_to,hold_at_location", {"fix": "NOT_A_FIX"})

    aircraft.pilot.receive_actions([action], environment)
    with pytest.raises(ValueError, match="Unknown holding fix"):
        aircraft.pilot.process_actions(environment)


def test_hold_at_coordinate(generate_simple_environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    location = (51.25, -1.75)
    action = Action(aircraft.callsign, "route_direct_to,hold_at_location", {"location": location})

    aircraft.pilot.process_lateral_actions(action, environment)

    assert aircraft.predictor_params["hold"]["fix"] is None
    assert aircraft.predictor_params["hold"]["location"] == list(location)
    assert aircraft.heading_changing_to == pytest.approx(aircraft.pos2d().bearing_to(Pos2D(*location)))


def test_pilot_from_complete_json():
    """
    Test the Pilot from_json method on a complete Aircraft json string.
    """

    json_string = """
        {
            "pilot_type": "Pilot",
            "callsign": "AIR0"
        }
    """

    pilot = Pilot.from_json(json_string)

    assert isinstance(pilot, Pilot)
    assert pilot.callsign == "AIR0"


def test_pilot_from_aircraft_json():
    """
    Test Pilots are instantiated correctly using Aircraft.from_json().
    """

    json_string = """ {
        "flight_plan": {
            "route": {"current": ["ABC", "DEF", "GHI"], "filed": ["ABC", "DEF", "GHI"]},
            "sectors": [
                "first_sector_name",
                "second_sector_name"
            ]
        },
        "callsign": "AIR0",
        "lat": 51.4702,
        "lon": -0.4479,
        "fl": 120,
        "heading": 249.68,
        "speed": 200,
        "vertical_speed": 0,
        "aircraft_type": "DEFAULT",
        "pilot": {
            "pilot_type": "Pilot",
            "callsign": "AIR0"
        }
    } 
    """

    aircraft = Aircraft.from_json(json_string)
    pilot = aircraft.pilot
    assert pilot.callsign == "AIR0"
    assert isinstance(pilot, Pilot)


def test_pilot_from_incomplete_json():
    """
    Test the default Pilot is instantiated when no Pilot data is found in the Aircraft data string.
    """

    json_string = """ {
        "flight_plan": {
            "route": {"current": ["ABC", "DEF", "GHI"], "filed": ["ABC", "DEF", "GHI"]},
            "sectors": [
                "first_sector_name",
                "second_sector_name"
            ]
        },
        "callsign": "AIR0",
        "lat": 51.4702,
        "lon": -0.4479,
        "fl": 120,
        "heading": 249.68,
        "speed": 200,
        "vertical_speed": 0,
        "aircraft_type": "DEFAULT"
    } 
    """

    aircraft = Aircraft.from_json(json_string)

    assert aircraft.pilot.callsign == "AIR0"
