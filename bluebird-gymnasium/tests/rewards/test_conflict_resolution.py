from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.conflict_resolution import (
    conflict_resolution_exp,
    conflict_resolution_tanh,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_conflict_resolution_exp(gym_env: BaseEnv):
    """Test `conflict_resolution_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -1.0 <= conflict_resolution_exp(gym_env, callsign, action) <= 0.0


def test_conflict_resolution_tanh(gym_env: BaseEnv):
    """Test `conflict_resolution_tanh` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -1.0 <= conflict_resolution_tanh(gym_env, callsign, action) <= 0.0
