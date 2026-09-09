from __future__ import annotations

import typing

import pytest
from bluebird_dt.core import Action

from bluebird_gymnasium.actions.simple.route_direct import route_direct

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_route_direct(gym_env: BaseEnv):
    """Test `route_direct` action function."""

    _sector = gym_env.get_active_airspace_sector()
    tracked_aircraft = gym_env.get_tracked_aircraft_data()

    callsign = next(iter(tracked_aircraft.keys()))
    next_fix = tracked_aircraft[callsign].next_fix_fr

    action = route_direct(callsign, gym_env)
    assert action.kind == "route_direct_to"
    assert action.value == next_fix


def test_route_direct_fix_after_next(gym_env: BaseEnv):
    """Test `route_direct` action function."""

    _sector = gym_env.get_active_airspace_sector()
    tracked_aircraft = gym_env.get_tracked_aircraft_data()

    callsign = next(iter(tracked_aircraft.keys()))
    next_fix = tracked_aircraft[callsign].next_fix_fr

    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    route = aircraft.flight_plan.route.filed
    next_fix_idx = route.index(next_fix)

    # route direct to the fix after the next forward fix
    # `value` is set to 2 for this.
    action = route_direct(callsign, gym_env, value=2)
    fix_after_next_fix = route[next_fix_idx + 1]
    assert action.kind == "route_direct_to"
    assert action.value == fix_after_next_fix


def test_route_direct_to_previous_fix_raises(gym_env: BaseEnv):
    """Test that route direct actions cannot target previous fixes."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()

    callsign = next(iter(tracked_aircraft.keys()))
    next_fix = tracked_aircraft[callsign].next_fix_fr

    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    route = aircraft.flight_plan.route.filed
    next_fix_idx = route.index(next_fix)
    assert next_fix_idx > 0

    previous_fix = route[next_fix_idx - 1]
    action = Action(callsign, "route_direct_to", previous_fix)

    with pytest.raises(AssertionError, match="should be a future fix"):
        gym_env.action_p.convert_simulator_action_to_gym_action(action, gym_env)
