from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.action_penalty import (
    action_penalty_const,
    action_penalty_thresh,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs import BaseEnv


def test_action_penalty_const(gym_env: BaseEnv):
    """Test `action_penalty_const` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 1
    assert action_penalty_const(gym_env, callsign, action) == -1.0
    action = 0
    assert action_penalty_const(gym_env, callsign, action) == 0.0


def test_action_penalty_thresh(gym_env: BaseEnv):
    """Test `action_penalty_thresh` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert action_penalty_thresh(gym_env, callsign, action) == 0.0

    action = 1
    assert action_penalty_thresh(gym_env, callsign, action) <= 0.0
