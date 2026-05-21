from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.constants import STEPS_SINCE_ACTION_MAX
from bluebird_gymnasium.utils.geo_utils import get_centreline_distance


def action_penalty_memory(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:
    """Penalize taking actions. Linear decay has the effect of minimising delay
    between necessary actions.

    Penalty computed as a linear decay from max value following an action.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -1.0 to 0.0).
    """

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    if ac_tracked_state is None or ac_tracked_state.steps_since_action is None:
        steps_since_action = STEPS_SINCE_ACTION_MAX
    else:
        steps_since_action = ac_tracked_state.steps_since_action

    reward = steps_since_action / STEPS_SINCE_ACTION_MAX - 1

    return reward


def action_penalty_const(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:
    """Penalize taking action when it is not necessary.

    Penalty computed as a constant (`const`) cost per RL time step.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (constant/fixed value: -1.0 if action is
        not 0 (No action), else returns 0.0).
    """

    if action != 0:
        reward = -1.0
    else:
        reward = 0.0
    return reward


def action_penalty_thresh(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:
    """Penalize taking action when it is not necessary.

    Penalty computed based on the aircraft's distance from the aircraft's
    current route centreline.
    If the distance is above a threshold, then a fixed cost is applied.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -infinity to 0.0).
    """

    simulator_env = gym_env.get_simulator_env()
    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)

    # set variables to specified arg value, or the default if not specified
    penalty = 30
    epsilon_centreline = 1.5

    reward = 0
    if action != 0:
        # if aircraft is away from centreline (thresholded by some epsilon),
        # then don't penalise action taken
        ac = simulator_env.aircraft[callsign]
        if ac_tracked_state is None or ac_tracked_state.centreline_info_fr is None:
            centre_dist, _, _ = get_centreline_distance(
                ac.pos2d(), ac.flight_plan.route.current, simulator_env.airspace
            )
        else:
            centre_dist, _, _ = ac_tracked_state.centreline_info_cr
        reward = penalty if centre_dist < epsilon_centreline else 0
    return -1.0 * reward
