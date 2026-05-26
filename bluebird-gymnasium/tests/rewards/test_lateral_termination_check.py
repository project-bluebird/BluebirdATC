from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.lateral_termination_check import (
    lateral_termination_check_mac,
    lateral_termination_check_sac,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_lateral_termination_check_sac(gym_env: BaseEnv):
    """Test `lateral_termination_check_sac` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert lateral_termination_check_sac(gym_env, callsign, action, gym_env.timestep, gym_env.maxstep) <= 0.0


def test_lateral_termination_check_mac(gym_env: BaseEnv):
    """Test `lateral_termination_check_mac` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert (
        lateral_termination_check_mac(
            gym_env,
            callsign,
            action,
            gym_env.timestep,
            gym_env.maxstep,
            transferred=False,
        )
        <= 0.0
    )
