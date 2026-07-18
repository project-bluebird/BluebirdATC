from itertools import pairwise

import pytest

from bluebird_dt.core import Action, Environment
from bluebird_dt.predictor import SimplePredictor


def test_repeating_standard_rate_hold(generate_simple_environment: Environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    aircraft.selected_instructions.cas = 360
    aircraft.cleared_instructions.cas = 360
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)
    action = Action(
        aircraft.callsign,
        "route_direct_to,hold_at_location",
        {"fix": "EARTH", "outbound_time": 30, "turn_direction": "right"},
    )
    aircraft.pilot.process_lateral_actions(action, environment)

    transitions = []
    turn_headings = []
    previous_phase = aircraft.predictor_params["hold"]["phase"]

    for elapsed in range(1, 501):
        predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=False)
        phase = aircraft.predictor_params["hold"]["phase"]
        if phase != previous_phase:
            transitions.append((phase, elapsed, aircraft.pos2d()))
            previous_phase = phase
        if phase in ["turn_outbound", "turn_inbound"]:
            turn_headings.append(aircraft.heading)

    phases = [transition[0] for transition in transitions]
    assert phases[:6] == [
        "turn_outbound",
        "outbound",
        "turn_inbound",
        "inbound",
        "turn_outbound",
        "outbound",
    ]

    transition_times = [transition[1] for transition in transitions]
    assert transition_times[1] - transition_times[0] == 60
    assert transition_times[2] - transition_times[1] == 30
    assert transition_times[3] - transition_times[2] == 60
    assert transition_times[5] - transition_times[4] == 60

    hold_fix = environment.airspace.fixes.places["EARTH"]
    first_turn_position = transitions[0][2]
    second_turn_position = transitions[4][2]
    assert first_turn_position.distance(hold_fix) <= 2.2
    assert second_turn_position.distance(hold_fix) <= 2.2

    turn_deltas = [(second - first) % 360 for first, second in pairwise(turn_headings)]
    assert max(delta for delta in turn_deltas if delta < 10.0) == pytest.approx(3.0, abs=0.01)


def test_hold_state_survives_aircraft_json_roundtrip(generate_simple_environment: Environment):
    aircraft = generate_simple_environment.aircraft["AIR0"]
    aircraft.predictor_params["hold"] = {
        "fix": "EARTH",
        "outbound_time": 90.0,
        "turn_direction": "left",
        "phase": "outbound",
        "phase_elapsed": 12.0,
        "inbound_track": 0.0,
    }

    restored = aircraft.from_json(aircraft.to_json())

    assert restored.predictor_params["hold"] == aircraft.predictor_params["hold"]


def test_coordinate_hold_does_not_require_predictor_fixes(generate_simple_environment: Environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    aircraft.selected_instructions.cas = 360
    aircraft.cleared_instructions.cas = 360
    hold_position = environment.airspace.fixes.places["EARTH"]
    predictor = SimplePredictor(1.0, 2.0, fixes=None)
    action = Action(
        aircraft.callsign,
        "route_direct_to,hold_at_location",
        {"location": (hold_position.lat, hold_position.lon), "outbound_time": 30},
    )
    aircraft.pilot.process_lateral_actions(action, environment)

    for _ in range(300):
        predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=False)

    assert aircraft.predictor_params["hold"]["phase"] != "direct_to_location"
