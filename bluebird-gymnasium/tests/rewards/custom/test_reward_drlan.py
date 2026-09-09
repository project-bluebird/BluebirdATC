from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.custom.reward_drlan import reward_drlan

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs import BaseEnv


def test_reward_drlan(gym_env: BaseEnv):
    """Test `reward_drlan` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    # for single aircraft env with noop action taken, a small
    # negative reward is expected.
    action = 0
    assert reward_drlan(gym_env, callsign, action) <= 0.0
