from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.safety import (
    safety_simple_avoidance_exp,
    safety_simple_avoidance_nvl,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_safety_simple_avoidance_exp(gym_env: BaseEnv):
    """Test `safety_simple_avoidance_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert safety_simple_avoidance_exp(gym_env, callsign, action) <= 0.0


def test_safety_simple_avoidance_nvl(gym_env: BaseEnv):
    """Test `safety_simple_avoidance_nvl` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert safety_simple_avoidance_nvl(gym_env, callsign, action) <= 0.0
