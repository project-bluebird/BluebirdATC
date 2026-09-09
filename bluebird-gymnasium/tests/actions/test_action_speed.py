from __future__ import annotations

import typing

from bluebird_gymnasium.actions import DEFAULT_RELATIVE_SPEED
from bluebird_gymnasium.actions.simple.speed import (
    speed_choose_own,
    speed_decrease,
    speed_increase,
    speed_maintain_current,
)
from bluebird_gymnasium.utils.simulator_utils import get_aircraft_selected_cas

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_speed_increase(gym_env: BaseEnv):
    """Test `speed_increase` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))
    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    selected_cas = round(get_aircraft_selected_cas(aircraft))

    action = speed_increase(callsign, gym_env)
    assert action.kind == "change_cas_to"
    assert action.value == selected_cas + DEFAULT_RELATIVE_SPEED

    action = speed_increase(callsign, gym_env, value=15.0)
    assert action.kind == "change_cas_to"
    assert action.value == selected_cas + 15.0


def test_speed_decrease(gym_env: BaseEnv):
    """Test `speed_decrease` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))
    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    selected_cas = round(get_aircraft_selected_cas(aircraft))

    action = speed_decrease(callsign, gym_env)
    assert action.kind == "change_cas_to"
    assert action.value == selected_cas - DEFAULT_RELATIVE_SPEED

    action = speed_decrease(callsign, gym_env, value=15.0)
    assert action.kind == "change_cas_to"
    assert action.value == selected_cas - 15.0


def test_speed_maintain_current(gym_env: BaseEnv):
    """Test `speed_maintain_current` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))
    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    selected_cas = round(get_aircraft_selected_cas(aircraft))

    action = speed_maintain_current(callsign, gym_env)
    assert action.kind == "change_cas_to"
    assert action.value == selected_cas


def test_speed_choose_own(gym_env: BaseEnv):
    """Test `speed_choose_own` action function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = speed_choose_own(callsign, gym_env)
    assert action.kind == "change_cas_to"
    assert action.value is None
