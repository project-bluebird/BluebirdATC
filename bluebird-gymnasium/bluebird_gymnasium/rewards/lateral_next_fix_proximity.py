import warnings

import numpy as np
from bluebird_dt.utility import convert

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.geo_utils import angle_diff
from bluebird_gymnasium.utils.simulator_utils import prev_next_fixes


def lateral_next_fix_proximity_bpfnf(
    gym_env: BaseEnv, callsign: str, action: int, next_fix_idx: int, **kwargs # noqa: ARG001, ANN003
) -> float:
    """Reward for aircraft's proximity to next fix.

    Computed based on the aircraft's lateral distance from the next fix.
    Distance computed based on the angular difference between the previous fix
    to next fix bearing (bpfnf) and the aircraft's heading.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.
        next_fix_idx: index of the next fix based on aircraft's route.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()

    reward = 0
    ac = simulator_env.aircraft[callsign]
    fix0 = simulator_env.airspace.fixes.places[ac.flight_plan.route.filed[next_fix_idx - 1]]
    fix1 = simulator_env.airspace.fixes.places[ac.flight_plan.route.filed[next_fix_idx]]
    fix_parallel_heading = fix0.bearing_to(fix1)
    ac_next_angle = angle_diff(fix_parallel_heading, ac.heading)
    if ac_next_angle > 5:
        reward = ac_next_angle
    return -1.0 * reward


def lateral_next_fix_proximity_bacnf(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for aircraft's proximity to next fix.

    Computed based on the aircraft's lateral distance from the next fix.
    Distance computed based on the angular difference between the aircraft to
    next fix bearing (bacnf) and the aircraft's heading.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()

    reward = 0
    ac = simulator_env.aircraft[callsign]
    str_next_fix = simulator_env.airspace.closest_forward_fix(ac, distance_threshold_NMI=10.0)
    if str_next_fix is None:
        # assume that aircraft has gotten to the end of its planned
        # route in its flight plan. raise a warning for now and
        # set the next fix as the last fix on its current route.
        warnings.warn(
            f"{callsign} has likely gotten to the end of its"
            " planned route. Therefore, there's no next fix. Assuming next"
            " fix as the last fix on its planned route",
            UserWarning,
            stacklevel=0,
        )
        str_next_fix = ac.flight_plan.route.current[-1]
    ac_next_fix = simulator_env.airspace.fixes.places[str_next_fix]
    ac_next_fix_bearing = ac.pos2d().bearing_to(ac_next_fix)
    ac_next_range = ac.pos2d().distance(ac_next_fix)
    ac_next_angle = angle_diff(ac_next_fix_bearing, ac.heading)
    if ac_next_angle < 90:
        ac_fix_proximity = abs(np.sin(ac_next_angle * convert.DEG_TO_RAD)) * ac_next_range
    else:
        ac_fix_proximity = ac.pos2d().distance(ac_next_fix) * (ac_next_angle / 90)
    if ac_fix_proximity > 5:
        reward = ac_fix_proximity
    return -1.0 * reward


def lateral_next_fix_proximity_dist_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for aircraft's proximity to next fix.

    Computed based on the aircraft's lateral distance from the next fix.
    The distance is scaled by the distance between the aircraft's previous
    and next fixes, and then exponentiated.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    aircraft = simulator_env.aircraft[callsign]

    places = simulator_env.airspace.fixes.places
    if ac_tracked_state is None:
        prev_fix, next_fix = prev_next_fixes(callsign, simulator_env, use_filed_route=True)
    else:
        prev_fix = ac_tracked_state.previous_fix_fr
        next_fix = ac_tracked_state.next_fix_fr

    prev_fix = places[prev_fix]
    next_fix = places[next_fix]
    pf_nf_dist = prev_fix.distance(next_fix)
    ac_nf_dist = aircraft.distance(next_fix)

    return np.exp(-ac_nf_dist / pf_nf_dist)
