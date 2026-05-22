from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.expeditious import (
    expeditious_const,
    expeditious_exp,
    expeditious_linear,
    expeditious_quad,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_expeditious_const(gym_env: BaseEnv):
    """Test `expeditious_const` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -1.0 <= expeditious_const(gym_env, callsign, action) <= 1.0


def test_expeditious_linear(gym_env: BaseEnv):
    """Test `expeditious_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -1.0 <= expeditious_linear(gym_env, callsign, action) <= 1.0


def test_expeditious_quad(gym_env: BaseEnv):
    """Test `expeditious_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -1.5 <= expeditious_quad(gym_env, callsign, action) <= 1.5


def test_expeditious_exp(gym_env: BaseEnv):
    """Test `expeditious_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert 0.0 <= expeditious_exp(gym_env, callsign, action) <= 1.0
