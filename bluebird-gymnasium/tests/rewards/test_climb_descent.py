from __future__ import annotations

import typing

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.rewards.climb_descent import (
    climb_target_exp,
    climb_target_linear,
    climb_target_quad,
    descent_target_exp,
    descent_target_linear,
    descent_target_quad,
    overflier_const,
    overflier_exp,
    overflier_linear,
    overflier_quad,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_climb_target_linear(gym_env: BaseEnv):
    """Test `climb_target_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert climb_target_linear(gym_env, callsign, action) <= 0.0


def test_climb_target_quad(gym_env: BaseEnv):
    """Test `climb_target_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert climb_target_quad(gym_env, callsign, action) <= 0.0


def test_climb_target_exp(gym_env: BaseEnv):
    """Test `climb_target_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -5.0 <= climb_target_exp(gym_env, callsign, action) <= 1.0


def test_descent_target_linear(gym_env: BaseEnv):
    """Test `descent_target_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert descent_target_linear(gym_env, callsign, action) <= 0.0


def test_descent_target_quad(gym_env: BaseEnv):
    """Test `descent_target_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert descent_target_quad(gym_env, callsign, action) <= 0.0


def test_descent_target_exp(gym_env: BaseEnv):
    """Test `descent_target_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -5.0 <= descent_target_exp(gym_env, callsign, action) <= 1.0


def test_overflier_const(gym_env: BaseEnv):
    """Test `overflier_const` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert 0.0 <= overflier_const(gym_env, callsign, action) <= 1.0


def test_overflier_linear(gym_env: BaseEnv):
    """Test `overflier_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert overflier_linear(gym_env, callsign, action) <= 0.0


def test_overflier_quad(gym_env: BaseEnv):
    """Test `overflier_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert overflier_quad(gym_env, callsign, action) <= 0.0


def test_overflier_exp(gym_env: BaseEnv):
    """Test `overflier_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert -5.0 <= overflier_exp(gym_env, callsign, action) <= 1.0
