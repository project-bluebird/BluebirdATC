from itertools import pairwise

import pytest

from bluebird_dt.core import Action, Environment, HoldAtFixParameters, HoldAtLocationParameters
from bluebird_dt.predictor import SimplePredictor


def test_repeating_standard_rate_hold(generate_simple_environment: Environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    aircraft.selected_instructions.cas = 360
    aircraft.cleared_instructions.cas = 360
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)
    hold_fix = environment.airspace.fixes.places["EARTH"]
    inbound_course_deg = aircraft.pos2d().bearing_to(hold_fix)
    action = Action(
        aircraft.callsign,
        "route_direct_to,hold_at_location",
        HoldAtFixParameters(
            fix="EARTH",
            hold_orientation_deg=(inbound_course_deg + 180.0) % 360.0,
            outbound_time_s=30,
            turn_direction="right",
        ),
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
        "location": None,
        "inbound_course_deg": 0.0,
        "outbound_time_s": 90.0,
        "turn_direction": "left",
        "phase": "outbound",
        "phase_elapsed": 12.0,
        "entry_type": "direct",
    }

    restored = aircraft.from_json(aircraft.to_json())

    assert restored.predictor_params["hold"] == aircraft.predictor_params["hold"]


def test_coordinate_hold_does_not_require_predictor_fixes(generate_simple_environment: Environment):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    aircraft.selected_instructions.cas = 360
    aircraft.cleared_instructions.cas = 360
    hold_position = environment.airspace.fixes.places["EARTH"]
    inbound_course_deg = aircraft.pos2d().bearing_to(hold_position)
    predictor = SimplePredictor(1.0, 2.0, fixes=None)
    action = Action(
        aircraft.callsign,
        "route_direct_to,hold_at_location",
        HoldAtLocationParameters(
            location=(hold_position.lat, hold_position.lon),
            hold_orientation_deg=(inbound_course_deg + 180.0) % 360.0,
            outbound_time_s=30,
        ),
    )
    aircraft.pilot.process_lateral_actions(action, environment)

    for _ in range(300):
        predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=False)

    assert aircraft.predictor_params["hold"]["phase"] != "direct_to_location"


@pytest.mark.parametrize(
    ("turn_direction", "arrival_track_deg", "expected_entry"),
    [
        ("right", 0.0, "direct"),
        ("right", 225.0, "parallel"),
        ("right", 135.0, "teardrop"),
        ("left", 0.0, "direct"),
        ("left", 135.0, "parallel"),
        ("left", 225.0, "teardrop"),
        ("right", 110.0, "direct"),
        ("right", 180.0, "parallel"),
        ("right", 290.0, "direct"),
    ],
)
def test_select_hold_entry(
    turn_direction: str,
    arrival_track_deg: float,
    expected_entry: str,
):
    assert SimplePredictor.select_hold_entry(arrival_track_deg, 0.0, turn_direction) == expected_entry


@pytest.mark.parametrize(
    ("turn_direction", "arrival_track_deg", "expected_entry", "expected_phase"),
    [
        ("right", 0.0, "direct", "turn_outbound"),
        ("right", 225.0, "parallel", "parallel_turn_outbound"),
        ("right", 135.0, "teardrop", "teardrop_turn_outbound"),
        ("left", 135.0, "parallel", "parallel_turn_outbound"),
        ("left", 225.0, "teardrop", "teardrop_turn_outbound"),
    ],
)
def test_hold_starts_selected_entry(
    generate_simple_environment: Environment,
    turn_direction: str,
    arrival_track_deg: float,
    expected_entry: str,
    expected_phase: str,
):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    hold_fix = environment.airspace.fixes.places["EARTH"]
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)
    aircraft.pilot.process_lateral_actions(
        Action(
            aircraft.callsign,
            "route_direct_to,hold_at_location",
            HoldAtFixParameters(
                fix="EARTH",
                hold_orientation_deg=180.0,
                turn_direction=turn_direction,
            ),
        ),
        environment,
    )
    aircraft.lat = hold_fix.lat
    aircraft.lon = hold_fix.lon
    aircraft.heading = arrival_track_deg
    aircraft.heading_changing_to = None

    predictor.update_hold_guidance(
        aircraft,
        aircraft.predictor_params["hold"],
        arrival_track_deg,
        aircraft.selected_instructions.cas,
        None,
    )

    assert aircraft.predictor_params["hold"]["entry_type"] == expected_entry
    assert aircraft.predictor_params["hold"]["phase"] == expected_phase


@pytest.mark.parametrize(
    ("turn_direction", "relative_arrival_deg", "entry_type"),
    [
        ("right", 225.0, "parallel"),
        ("right", 135.0, "teardrop"),
        ("left", 225.0, "parallel"),
        ("left", 135.0, "teardrop"),
    ],
)
def test_non_direct_entry_joins_repeating_hold(
    generate_simple_environment: Environment,
    turn_direction: str,
    relative_arrival_deg: float,
    entry_type: str,
):
    environment = generate_simple_environment
    aircraft = environment.aircraft["AIR0"]
    aircraft.selected_instructions.cas = 360
    aircraft.cleared_instructions.cas = 360
    hold_fix = environment.airspace.fixes.places["EARTH"]
    arrival_track_deg = aircraft.pos2d().bearing_to(hold_fix)
    direction_sign = 1.0 if turn_direction == "right" else -1.0
    inbound_course_deg = (arrival_track_deg - direction_sign * relative_arrival_deg) % 360.0
    predictor = SimplePredictor(1.0, 2.0, fixes=environment.airspace.fixes)
    aircraft.pilot.process_lateral_actions(
        Action(
            aircraft.callsign,
            "route_direct_to,hold_at_location",
            HoldAtFixParameters(
                fix="EARTH",
                hold_orientation_deg=(inbound_course_deg + 180.0) % 360.0,
                outbound_time_s=30,
                turn_direction=turn_direction,
            ),
        ),
        environment,
    )

    transitions = []
    previous_phase = aircraft.predictor_params["hold"]["phase"]
    for elapsed_s in range(1, 901):
        predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=False)
        phase = aircraft.predictor_params["hold"]["phase"]
        if phase != previous_phase:
            transitions.append((phase, elapsed_s, aircraft.heading))
            previous_phase = phase
        if phase == "outbound":
            break

    phases = [phase for phase, _, _ in transitions]
    assert phases == [
        f"{entry_type}_turn_outbound",
        f"{entry_type}_outbound",
        f"{entry_type}_turn_inbound",
        f"{entry_type}_inbound",
        "turn_outbound",
        "outbound",
    ]
    assert aircraft.predictor_params["hold"]["entry_type"] == entry_type
    outbound_track_deg = (inbound_course_deg + 180.0) % 360.0
    expected_entry_track_deg = (
        outbound_track_deg
        if entry_type == "parallel"
        else (outbound_track_deg - direction_sign * 30.0) % 360.0
    )
    assert transitions[1][2] == pytest.approx(expected_entry_track_deg)
    assert transitions[-1][2] == pytest.approx(outbound_track_deg)
    entry_started_s = transitions[0][1]
    entry_outbound_started_s = transitions[1][1]
    entry_inbound_turn_started_s = transitions[2][1]
    assert entry_inbound_turn_started_s - entry_started_s == 60
    assert entry_inbound_turn_started_s - entry_outbound_started_s < 60
    if entry_type == "parallel":
        parallel_turn_delta_deg = (transitions[2][2] - expected_entry_track_deg + 180.0) % 360.0 - 180.0
        assert parallel_turn_delta_deg * direction_sign < 0.0
