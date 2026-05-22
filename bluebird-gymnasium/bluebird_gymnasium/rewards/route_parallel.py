from __future__ import annotations

import typing

import numpy as np

from bluebird_gymnasium.utils.simulator_utils import prev_next_fixes_positions

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv

EPSILON = 0.5


def route_parallel_const(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float: # noqa: ARG001, ANN003
    """Reward for flying parallel to the aircraft's current route.

    Apply a constant penalty of -1.0 when the aircraft is not flying parallel
    to its current route. Otherwise, return 0.0.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: -1.0 to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    tracked_data = gym_env.get_tracked_aircraft_data(callsign)
    forward_fixes_info = gym_env.get_forward_fixes_info()
    aircraft = simulator_env.aircraft[callsign]

    if aircraft.on_route:
        return 0.0

    if forward_fixes_info.use_filed_route:
        prev_fix = tracked_data.previous_fix_fr
        next_fix = tracked_data.next_fix_fr
    else:
        prev_fix = tracked_data.previous_fix_cr
        next_fix = tracked_data.next_fix_cr

    route_start_pos = tracked_data.pos_at_last_route_direct
    previous_fix_pos, next_fix_pos = prev_next_fixes_positions(
        prev_fix, next_fix, simulator_env.airspace, route_start_pos
    )
    bearing_pf_nf = previous_fix_pos.bearing_to(next_fix_pos)

    if aircraft.selected_instructions.heading is not None:
        selected_heading = aircraft.selected_instructions.heading
    else:
        # use the heading as the selected heading
        selected_heading = aircraft.heading

    diff = abs(bearing_pf_nf - selected_heading)
    # apply threshold: if diff < epsilon, clip it to zero
    if diff < EPSILON:
        return 0.0
    return -1.0


def route_parallel_linear(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float: # noqa: ARG001, ANN003
    """Reward for flying parallel to the aircraft's current route.

    Linearly penalize the aircraft for not flying parallel to its
    current route.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    tracked_data = gym_env.get_tracked_aircraft_data(callsign)
    forward_fixes_info = gym_env.get_forward_fixes_info()
    aircraft = simulator_env.aircraft[callsign]

    if aircraft.on_route:
        return 0.0

    if forward_fixes_info.use_filed_route:
        prev_fix = tracked_data.previous_fix_fr
        next_fix = tracked_data.next_fix_fr
    else:
        prev_fix = tracked_data.previous_fix_cr
        next_fix = tracked_data.next_fix_cr

    route_start_pos = tracked_data.pos_at_last_route_direct
    previous_fix_pos, next_fix_pos = prev_next_fixes_positions(
        prev_fix, next_fix, simulator_env.airspace, route_start_pos
    )
    bearing_pf_nf = previous_fix_pos.bearing_to(next_fix_pos)

    if aircraft.selected_instructions.heading is not None:
        selected_heading = aircraft.selected_instructions.heading
    else:
        # use the heading as the selected heading
        selected_heading = aircraft.heading

    diff = abs(bearing_pf_nf - selected_heading)
    # apply threshold: if diff < epsilon, clip it to zero
    diff = max((diff - EPSILON), 0)
    return -1.0 * diff


def route_parallel_quad(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float: # noqa: ARG001, ANN003
    """Reward for flying parallel to the aircraft's current route.

    Quadratically penalize the aircraft for not flying parallel to its
    current route.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    tracked_data = gym_env.get_tracked_aircraft_data(callsign)
    forward_fixes_info = gym_env.get_forward_fixes_info()
    aircraft = simulator_env.aircraft[callsign]

    if aircraft.on_route:
        return 0.0

    if forward_fixes_info.use_filed_route:
        prev_fix = tracked_data.previous_fix_fr
        next_fix = tracked_data.next_fix_fr
    else:
        prev_fix = tracked_data.previous_fix_cr
        next_fix = tracked_data.next_fix_cr

    route_start_pos = tracked_data.pos_at_last_route_direct
    previous_fix_pos, next_fix_pos = prev_next_fixes_positions(
        prev_fix, next_fix, simulator_env.airspace, route_start_pos
    )
    bearing_pf_nf = previous_fix_pos.bearing_to(next_fix_pos)

    if aircraft.selected_instructions.heading is not None:
        selected_heading = aircraft.selected_instructions.heading
    else:
        # use the heading as the selected heading
        selected_heading = aircraft.heading

    diff = abs(bearing_pf_nf - selected_heading)
    # apply threshold: if diff < epsilon, clip it to zero
    diff = max((diff - EPSILON), 0)
    return -1.0 * diff**2


def route_parallel_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float: # noqa: ARG001, ANN003
    """Reward for flying parallel to the aircraft's current route.

    Exponentially reward the aircraft for flying parallel to its
    current route.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: 0.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    tracked_data = gym_env.get_tracked_aircraft_data(callsign)
    forward_fixes_info = gym_env.get_forward_fixes_info()
    aircraft = simulator_env.aircraft[callsign]

    if aircraft.on_route:
        return 1.0

    if forward_fixes_info.use_filed_route:
        prev_fix = tracked_data.previous_fix_fr
        next_fix = tracked_data.next_fix_fr
    else:
        prev_fix = tracked_data.previous_fix_cr
        next_fix = tracked_data.next_fix_cr

    route_start_pos = tracked_data.pos_at_last_route_direct
    previous_fix_pos, next_fix_pos = prev_next_fixes_positions(
        prev_fix, next_fix, simulator_env.airspace, route_start_pos
    )
    bearing_pf_nf = previous_fix_pos.bearing_to(next_fix_pos)

    if aircraft.selected_instructions.heading is not None:
        selected_heading = aircraft.selected_instructions.heading
    else:
        # use the heading as the selected heading
        selected_heading = aircraft.heading

    diff = abs(bearing_pf_nf - selected_heading)
    # apply threshold: if diff < epsilon, clip it to zero
    diff = max((diff - EPSILON), 0)
    return float(np.exp(-1.0 * diff))
