from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.custom.custom_reward import custom_reward_fn

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs import BaseEnv


def test_custom_reward_fn(gym_env: BaseEnv):
    """Test `custom_reward_fn` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    # this function returns zero until it is implemented
    assert custom_reward_fn(gym_env, callsign, action) == 0.0
