from bluebird_gymnasium.actions import DEFAULT_RELATIVE_HEADING
from bluebird_gymnasium.actions.simple.heading import (
    heading_left,
    heading_right,
    heading_route_parallel,
    heading_maintain_current,
)


def test_heading_left(gym_env):
    """Test `heading_left` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]
    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    if aircraft.selected_instructions.heading is not None:
        prev_selected_heading = aircraft.selected_instructions.heading
    else:
        prev_selected_heading = aircraft.heading

    action = heading_left(callsign, gym_env)
    assert action.kind == "change_heading_to"
    selected_heading = (
        int(round(prev_selected_heading - DEFAULT_RELATIVE_HEADING, 0)) % 360
    )
    assert action.value == selected_heading

    value = 15
    action = heading_left(callsign, gym_env, value=value)
    assert action.kind == "change_heading_to"
    selected_heading = int(round(prev_selected_heading - value, 0)) % 360
    assert action.value == selected_heading


def test_heading_right(gym_env):
    """Test `heading_right` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]
    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    if aircraft.selected_instructions.heading is not None:
        prev_selected_heading = aircraft.selected_instructions.heading
    else:
        prev_selected_heading = aircraft.heading

    action = heading_right(callsign, gym_env)
    assert action.kind == "change_heading_to"
    selected_heading = (
        int(round(prev_selected_heading + DEFAULT_RELATIVE_HEADING, 0)) % 360
    )
    assert action.value == selected_heading

    value = 15
    action = heading_right(callsign, gym_env, value=value)
    assert action.kind == "change_heading_to"
    selected_heading = int(round(prev_selected_heading + value, 0)) % 360
    assert action.value == selected_heading


def test_heading_route_parallel(gym_env):
    """Test `heading_route_parallel` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = heading_route_parallel(callsign, gym_env)
    assert action.kind == "change_heading_to"
    assert 0 <= action.value <= 360


def test_heading_maintain_current(gym_env):
    """Test `heading_maintain_current` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = heading_maintain_current(callsign, gym_env)
    assert action.kind == "maintain_current_heading"
