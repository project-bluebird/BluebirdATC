from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.position import (
    position_status_const,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_position_status_const(gym_env: BaseEnv):
    """Test `position_status_const` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert position_status_const(gym_env, callsign, action) == 0.0
