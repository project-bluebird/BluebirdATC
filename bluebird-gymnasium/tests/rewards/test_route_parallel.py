from __future__ import annotations

import typing

from bluebird_gymnasium.rewards.route_parallel import (
    route_parallel_const,
    route_parallel_exp,
    route_parallel_linear,
    route_parallel_quad,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_route_parallel_const(gym_env: BaseEnv):
    """Test `route_parallel_const` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert route_parallel_const(gym_env, callsign, action) <= 0.0

    action_parser = gym_env.get_action_parser()
    left_action = gym_env.get_action_parser().get_heading_left_actions()[0]
    # fly parallel to the current segment in the route.
    route_parallel_action = action_parser.get_heading_route_parallel_actions()[0]

    _ = gym_env.step(left_action)  # no longer parallel to the route
    assert route_parallel_const(gym_env, callsign, left_action) < 0.0

    # the aircraft is now parallel to the route even though
    # it is slightly off the route's centreline.
    _ = gym_env.step(route_parallel_action)
    assert route_parallel_const(gym_env, callsign, route_parallel_action) == 0.0


def test_route_parallel_linear(gym_env: BaseEnv):
    """Test `route_parallel_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert route_parallel_linear(gym_env, callsign, action) <= 0.0

    action_parser = gym_env.get_action_parser()
    left_action = gym_env.get_action_parser().get_heading_left_actions()[0]
    # fly parallel to the current segment in the route.
    route_parallel_action = action_parser.get_heading_route_parallel_actions()[0]

    _ = gym_env.step(left_action)  # no longer parallel to the route
    assert route_parallel_linear(gym_env, callsign, left_action) < 0.0

    # the aircraft is now parallel to the route even though
    # it is slightly off the route's centreline.
    _ = gym_env.step(route_parallel_action)
    assert route_parallel_linear(gym_env, callsign, route_parallel_action) == 0.0


def test_route_parallel_quad(gym_env: BaseEnv):
    """Test `route_parallel_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert route_parallel_quad(gym_env, callsign, action) <= 0.0

    action_parser = gym_env.get_action_parser()
    left_action = gym_env.get_action_parser().get_heading_left_actions()[0]
    # fly parallel to the current segment in the route.
    route_parallel_action = action_parser.get_heading_route_parallel_actions()[0]

    _ = gym_env.step(left_action)  # no longer parallel to the route
    assert route_parallel_quad(gym_env, callsign, left_action) < 0.0

    # the aircraft is now parallel to the route even though
    # it is slightly off the route's centreline.
    _ = gym_env.step(route_parallel_action)
    assert route_parallel_quad(gym_env, callsign, route_parallel_action) == 0.0


def test_route_parallel_exp(gym_env: BaseEnv):
    """Test `route_parallel_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = next(iter(tracked_aircraft.keys()))

    action = 0
    assert route_parallel_exp(gym_env, callsign, action) >= 0.0

    action_parser = gym_env.get_action_parser()
    left_action = action_parser.get_heading_left_actions()[0]
    # fly parallel to the current segment in the route.
    route_parallel_action = action_parser.get_heading_route_parallel_actions()[0]

    _ = gym_env.step(left_action)  # no longer parallel to the route
    assert route_parallel_exp(gym_env, callsign, left_action) < 1.0

    # the aircraft is now parallel to the route even though
    # it is slightly off the route's centreline.
    _ = gym_env.step(route_parallel_action)
    assert route_parallel_exp(gym_env, callsign, route_parallel_action) == 1.0
