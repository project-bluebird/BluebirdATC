import numpy as np

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.geo_utils import (
    angle_diff,
    get_centreline_distance,
    left_right_check,
)
from bluebird_gymnasium.utils.simulator_utils import (
    predict_trajectory_simple,
    prev_next_fixes,
)
from bluebird_gymnasium.utils.types import TurnDirection


def lateral_centreline_distance_linear(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for staying close to the aircraft's filed route centreline.

    Computed based on the linear distance of an aircraft from its route's
    centre line.

    If the centre line distance is less than a threshold (i.e.,
    15), then 0 is returned.

    Otherwise, reward is computed as the subtraction
    of 15 from the centre line distance.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    reward = 0
    ac = simulator_env.aircraft[callsign]

    if ac_tracked_state is None or ac_tracked_state.centreline_info_fr is None:
        centre_dist, _, _ = get_centreline_distance(ac.pos2d(), ac.flight_plan.route.filed, simulator_env.airspace)
    else:
        centre_dist, _, _ = ac_tracked_state.centreline_info_fr

    reward = 0 if centre_dist <= 15 else (centre_dist - 15)
    return -1.0 * reward


def lateral_centreline_distance_quad(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for staying close to the aircraft's filed route centreline.

    Computed based on the aircraft's distance from the centreline, formulated
    as a quadratic (`quad`) penalty.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: -infinity to 2.5).
    """

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    ac = simulator_env.aircraft[callsign]

    if ac_tracked_state is None or ac_tracked_state.centreline_info_fr is None:
        centre_dist, _, _ = get_centreline_distance(ac.pos2d(), ac.flight_plan.route.filed, simulator_env.airspace)
    else:
        centre_dist, _, _ = ac_tracked_state.centreline_info_fr

    # return -1.0 * (centre_dist + ((centre_dist**2) / 10))

    # at centre_dist of 5 nautical miles, the reward is 0, < 5 yields
    # a positive reward, and > 5 yields a negative reward. the highest
    # positive reward is 2.5
    return (-1.0 * (centre_dist**2 / 10)) + 2.5


def lateral_centreline_distance_special(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for staying close to the aircraft's filed route centreline.

        Uses a shaped function to incentivise staying within a 20nm wide airway.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: 0.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac = simulator_env.aircraft[callsign]

    centre_dist, _, _ = get_centreline_distance(ac.pos2d(), ac.flight_plan.route.filed, simulator_env.airspace)

    return -1.0 * (1 - np.exp(-((centre_dist / 6) ** 2)))


def lateral_centreline_distance_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward for staying close to the aircraft's filed route centreline.

    Computed based on the negative exponent of the aircraft's distance from
    the centreline.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: 0.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    scale_factor = 10.0  # to control steepness of curve
    ac = simulator_env.aircraft[callsign]

    if ac_tracked_state is None or ac_tracked_state.centreline_info_fr is None:
        centre_dist, _, _ = get_centreline_distance(ac.pos2d(), ac.flight_plan.route.filed, simulator_env.airspace)
    else:
        centre_dist, _, _ = ac_tracked_state.centreline_info_fr

    # reward = centre_dist + (centre_dist ** 2) / 10
    if centre_dist <= 5.0:  # noqa: SIM108
        reward = 1.0
    else:
        reward = float(np.exp(-centre_dist / scale_factor))
    return reward


def lateral_centreline_distance_shaped(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Shaped Reward for staying close to route's centreline

    Penalize for taking lateral actions away from the route's centre and
    give positive reward for staying on the route's centre.

    If the aircraft is away from centreline (based on a threshold), then
    reward the agent if lateral actions are taken to return the
    aircraft to the centreline. Otherwise, penalize the agent.

    Also, if the aircraft is on the route's centreline, then reward the agent
    if no lateral action is taken for the aircraft (e.g., action 0). Otherwise,
    penalize the agent. *Note*: there's an exception to this formulation, which
    is: if the aircraft is on the route's centre but it is close to its
    next fix which would require a change of heading (turn), then reward agent
    if a heading (turn) action is issued because it would keep the agent on the
    route's centreline.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        the computed reward (range: -1.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # set variables to specified arg value, or the default if not specified
    epsilon_centreline = 5.0  # in nautical miles
    # epsilon_centreline = 10.0 # in nautical miles
    epsilon_angle_diff = 5.0  # in degrees

    ac = simulator_env.aircraft[callsign]
    if ac_tracked_state is None:
        centre_dist, centre_turn_dir, _ = get_centreline_distance(
            ac.pos2d(), ac.flight_plan.route.filed, simulator_env.airspace
        )
    else:
        centre_dist, centre_turn_dir, _ = ac_tracked_state.centreline_info_fr

    reward = None
    if action == 0:
        if centre_dist < epsilon_centreline:
            reward = 1.0
        else:
            # two scenarios.
            #
            # *scenario 1:* aircraft might be veering off from centreline and
            # the agent has taken no action to prevent it.
            #
            # *scenario 2:* the agent has instructed a heading action in a
            # previous time step to set the aircraft to navigate back to its
            # route's centre (likely as a result of a next fix that requires
            # a change of heading) and the agent does not need to take any
            # further action (hence NOOP action, action 0) as the aircraft's
            # continued navigation will re-position it at the route's centre.
            #
            # to detect whcih scenario the aircraft is in, simulate a future
            # trajectory of the aircraft based on the its current heading.
            # if future position of the aircraft is farther away from the
            # route's centreline, then it's scenario 1, otherwise, scenario 2
            #
            # if scenario 1, then penalize agent, otherwise, reward agent.

            future_pos = predict_trajectory_simple(ac.pos2d(), ac.heading, distance=1.0, num_control_points=1)[1]
            future_centre_dist, _, _ = get_centreline_distance(
                future_pos, ac.flight_plan.route.filed, simulator_env.airspace
            )
            reward = -1.0 if future_centre_dist >= centre_dist else 1.0

    else:
        # get categories of heading actions
        action_parser = gym_env.get_action_parser()
        actions_decr_hd = action_parser.get_heading_left_actions()
        actions_incr_hd = action_parser.get_heading_left_actions()

        if centre_dist < epsilon_centreline:  # aircraft on centreline
            places = simulator_env.airspace.fixes.places
            if ac_tracked_state is None:
                prev_fix, next_fix = prev_next_fixes(callsign, simulator_env)
            else:
                prev_fix = ac_tracked_state.previous_fix_fr
                next_fix = ac_tracked_state.next_fix_fr

            pf_nf_bearing = places[prev_fix].bearing_to(places[next_fix])
            nf_turn_dir = left_right_check(ac.heading, pf_nf_bearing)

            if angle_diff(ac.heading, pf_nf_bearing) > epsilon_angle_diff:
                # aircraft is approaching a (next) fix where a turn action
                # is required even though its on the route's centreline

                if (nf_turn_dir == TurnDirection.LEFT and action in actions_decr_hd) or (
                    nf_turn_dir == TurnDirection.RIGHT and action in actions_incr_hd
                ):
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                # penalize: agent made an unneccessary turn action for the
                # aircraft even though it's approximately at the route centre
                if (nf_turn_dir == TurnDirection.LEFT and action in actions_decr_hd) or (
                    nf_turn_dir == TurnDirection.RIGHT and action in actions_incr_hd
                ):
                    reward = np.exp(-centre_dist)
                else:
                    reward = -1.0

        else:  # aircraft away from centreline
            if centre_turn_dir == TurnDirection.LEFT and action in actions_decr_hd:
                # left direction and left action (heading left)
                reward = 1.0
            elif centre_turn_dir == TurnDirection.RIGHT and action in actions_incr_hd:
                # right direction and right action (heading left)
                reward = 1.0
            else:
                reward = -1.0
    return reward
