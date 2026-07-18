import pytest

from bluebird_dt.core import Action, HoldAtFixParameters
from bluebird_dt.scenario_manager import TwoAircraft

def test_init_exceptions():
    """
    Test Action initialization raises ValueError when invalid parameter values are used.
    """

    with pytest.raises(ValueError):
        # no callsign
        Action("", "outcomm", "next_sector")

    with pytest.raises(ValueError):
        # incorrect action kind
        Action("AIR0", "incorrect_kind", 10)


def test_init_exceptions_from_str():
    """
    Test Action initialization raises ValueError when invalid string given
    """
    with pytest.raises(ValueError):
        # incorrect set of parameters
        Action.from_str("AIR0 route_direct_to FIX1 FIX2 agent")


@pytest.mark.parametrize(
    "kind, value",
    [
        ("route_direct_to", "FIX"),
        ("route_direct_to,hold_at_location", HoldAtFixParameters(fix="FIX")),
        ("change_heading_to", 120),
        ("change_heading_by", -10),
        ("change_flight_level_to", 250),
        ("change_flight_level_by", -20),
        ("change_vertical_speed_to", 1250),
        ("descend_when_ready,level_by_fix", (100, "EARTH")),
        ("descend_now,level_by_fix", (120, "AIR")),
    ],
)
def test_values_set(kind, value):
    """Test values are set correctly in Action"""

    action = Action(callsign="AIR0", kind=kind, value=value)

    assert action.value == value


def test_str_without_sector():
    """Test Action.__str__ method with no sector"""
    action = Action(callsign="AIR0", kind="route_direct_to", value="FIX")
    expected_str = "AIR0 route_direct_to FIX on all"

    action_str = str(action)

    assert action_str == expected_str


def test_str_with_sector():
    """Test Action.__str__ method when sector specified"""
    action = Action(callsign="AIR0", kind="route_direct_to", value="FIX", sector=["SEC"])

    action_str = str(action)

    assert action_str == "AIR0 route_direct_to FIX on SEC"


def test_from_str():
    """Test Action.from_str class method"""
    action_str = "AIR0 route_direct_to FIX"
    expected_action = Action(callsign="AIR0", kind="route_direct_to", value="FIX")

    action = Action.from_str(action_str)

    assert action == expected_action


def test_from_str_with_sector():
    """Test Action.from_str when sector specified"""
    action_str = "AIR0 route_direct_to FIX on SEC1,SEC2"
    expected_action = Action(callsign="AIR0", kind="route_direct_to", value="FIX", sector=["SEC1","SEC2"])

    action = Action.from_str(action_str)

    assert action == expected_action


def test_hold_parameters_roundtrip():
    action = Action(
        callsign="AIR0",
        kind="route_direct_to,hold_at_location",
        value={"fix": "FIX", "outbound_time": 60, "turn_direction": "left"},
    )

    assert action == Action.from_json(action.to_json())
    assert action == Action.from_str(str(action))


def test_hold_parameters_from_json_string():
    action = Action(
        callsign="AIR0",
        kind="route_direct_to,hold_at_location",
        value='{"fix":"FIX","outbound_time":60,"turn_direction":"left"}',
    )

    assert action.value == HoldAtFixParameters(fix="FIX", outbound_time=60, turn_direction="left")


def test_coordinate_hold_parameters_roundtrip():
    action = Action(
        callsign="AIR0",
        kind="route_direct_to,hold_at_location",
        value={"location": (50.716667, -3.533333), "outbound_time": 60},
    )

    assert action.value.location == (50.716667, -3.533333)
    assert action.value.location_label == "50.7167, -3.53333"
    assert action == Action.from_json(action.to_json())
    assert action == Action.from_str(str(action))


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"fix": ""},
        {"fix": "FIX", "location": (50.0, -3.0)},
        {"location": (91.0, -3.0)},
        {"location": (50.0, -181.0)},
        {"fix": "FIX", "outbound_time": 0},
        {"fix": "FIX", "turn_direction": "straight"},
    ],
)
def test_invalid_hold_parameters(value: dict):
    with pytest.raises(ValueError, match="validation error"):
        Action("AIR0", "route_direct_to,hold_at_location", value)


@pytest.mark.parametrize(
    ("kind", "value", "agent", "sector"),
    [
        ("route_direct_to", "FIX", None, None),
        ("change_heading_to", 120, None, None),
        ("change_heading_by", -10, None, None),
        ("change_flight_level_to", 250, None, None),
        ("change_flight_level_by", -20, None, None),
        ("change_vertical_speed_to", 1250, None, None),
        ("descend_when_ready,level_by_fix", (100, "EARTH"), None, None),
        ("descend_now,level_by_fix", (120, "AIR"), None, None),
        ("route_direct_to", "FIX", "human", None),
        ("change_heading_to", 120, None, ["SEC"]),
        ("change_heading_by", -10, "human", ["SEC"]),
        ("route_direct_to", ["EARTH", "AIR"], None, None),
    ],
)
def test_action_str_roundtrip(kind: str, value: str | int | tuple[int, str], agent: str | None, sector: str | None):
    """Test roundtrip using Action.__str__ and Action.from_str with sector"""
    action = Action(callsign="AIR0", kind=kind, value=value, agent=agent, sector=sector)

    action_from_str = Action.from_str(str(action))

    assert action == action_from_str

def test_outcomm():
    sim = TwoAircraft.setup(scenario_name="Two Sector Two Aircraft", log_filename=None, predictor=None)

    sim.manager.receive_actions([Action("AIR1", "outcomm", "sector_2", "Test", sector=["sector_1"])])
    sim.manager.evolve(10)
    sim.manager.evolve(10)

    triggered_incomms = 0

    for clearance in sim.manager.event_logger.clearances_log:
        if clearance["kind"] == "outcomm":
            assert clearance["value"] == "sector_2"
            assert clearance["sector"] == ["sector_1"]
    
    assert sim.manager.environment.aircraft["AIR1"].current_sector == "sector_2"
