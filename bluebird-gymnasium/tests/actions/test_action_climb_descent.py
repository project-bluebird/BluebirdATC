from __future__ import annotations

import typing

from bluebird_gymnasium.actions import DEFAULT_RELATIVE_CLIMB_DESCENT
from bluebird_gymnasium.actions.simple.climb_descent import (
    fl_climb,
    fl_descent,
    fl_exit,
    fl_intermediate,
)
from bluebird_gymnasium.utils.types import PositionStatus

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_fl_climb(gym_env: BaseEnv):
    """Test `fl_climb` action function."""

    simulator_env = gym_env.get_simulator_env()
    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))
    aircraft = simulator_env.aircraft[callsign]

    action = fl_climb(callsign, gym_env)
    assert action.kind == "change_flight_level_to"
    assert action.value == aircraft.selected_fl + DEFAULT_RELATIVE_CLIMB_DESCENT

    value = 20
    action = fl_climb(callsign, gym_env, value=value)
    assert action.kind == "change_flight_level_to"
    assert action.value == aircraft.selected_fl + value


def test_fl_descent(gym_env: BaseEnv):
    """Test `fl_descent` action function."""

    simulator_env = gym_env.get_simulator_env()
    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))
    aircraft = simulator_env.aircraft[callsign]

    action = fl_descent(callsign, gym_env)
    assert action.kind == "change_flight_level_to"
    assert action.value == aircraft.selected_fl - DEFAULT_RELATIVE_CLIMB_DESCENT

    value = 20
    action = fl_descent(callsign, gym_env, value=value)
    assert action.kind == "change_flight_level_to"
    assert action.value == aircraft.selected_fl - value


def test_fl_intermediate(gym_env: BaseEnv):
    """Test `fl_intermediate` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    simulator_env = gym_env.get_simulator_env()

    for cs, ac_data in tracked_aircraft.items():
        if ac_data.pos_status == PositionStatus.IN_SECTOR:
            callsign = cs
            break

    aircraft = simulator_env.aircraft[callsign]
    ac_fl = aircraft.fl
    _sector = gym_env.get_active_airspace_sector()
    ac_exit_fl = tracked_aircraft[callsign].exit_coords[_sector].fl

    min_fl, max_fl = simulator_env.airspace.find_fl_lim(aircraft.pos2d())

    action = fl_intermediate(callsign, gym_env)
    if ac_fl <= ac_exit_fl:
        assert action.kind == "change_flight_level_to"
        action_fl = action.value
    else:
        # assert action.kind == "descend_now,level_by_fix"
        # action_fl = action.value[0]

        # option 2: use regular change_flight_level_to for
        # intermediate flight level descent.
        assert action.kind == "change_flight_level_to"
        action_fl = action.value

    assert min_fl <= action_fl <= max_fl


def test_fl_exit(gym_env: BaseEnv):
    """Test `fl_exit` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    simulator_env = gym_env.get_simulator_env()

    for cs, ac_data in tracked_aircraft.items():
        if ac_data.pos_status == PositionStatus.IN_SECTOR:
            callsign = cs
            break

    aircraft = simulator_env.aircraft[callsign]
    ac_fl = aircraft.fl
    _sector = gym_env.get_active_airspace_sector()
    ac_exit_fl = tracked_aircraft[callsign].exit_coords[_sector].fl

    min_fl, max_fl = simulator_env.airspace.find_fl_lim(aircraft.pos2d())

    action = fl_exit(callsign, gym_env)

    if ac_fl <= ac_exit_fl:
        assert action.kind == "change_flight_level_to"
        action_fl = action.value
    else:
        assert action.kind == "descend_now,level_by_fix"
        action_fl = action.value[0]

    assert min_fl <= action_fl <= max_fl
