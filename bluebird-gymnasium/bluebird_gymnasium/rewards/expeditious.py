import numpy as np

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.utils.constants import MAX_SPEED_TAS

DIFF_THRESHOLD = 1.0  # nautical miles (nmi)


def expeditious_const(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward or penalize for being navigating efficiently or inefficiently.

    Reward or penalty is computed as a constant.
    For a given aircraft, if its current distance to its exit is less than the
    previous distance to the exit, then a reward of +1.0 is given, else, -1.0
    is given.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -1.0 to 1.0).
    """

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    prev_ac_tracked_state = gym_env.get_tracked_aircraft_data_previous(callsign)

    if prev_ac_tracked_state is None:
        # aircraft tracking just started (either was just spawned or
        # arrived at the point before sector entry where tracking begins).
        # therefore, there's no previous step information to assess
        # its expeditious behaviour. it can be assessed from the next step.
        reward = 0.0
    else:
        reward = 1.0 if ac_tracked_state.track_dist_to_exit_cr < prev_ac_tracked_state.track_dist_to_exit_cr else -1.0

    return reward


def expeditious_linear(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward or penalize for being navigating efficiently or inefficiently.

    Reward is computed using an linear function.

    For a given aircraft, if the difference between the current and previous
    distance to its exit is negative (i.e., the aircraft is now closer to its
    exit than before), then a positive reward is computed with a maximum
    value of 1.0 (the larger the difference the closer the reward is to 1.0).
    Otherwise, a negative reward is computed with a minimum value of
    -1.0 (the larger the difference, the closer the reward is to -1.0) .

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -1.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    aircraft = simulator_env.aircraft[callsign]

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    prev_ac_tracked_state = gym_env.get_tracked_aircraft_data_previous(callsign)

    if prev_ac_tracked_state is None:
        # aircraft tracking just started (either was just spawned or
        # arrived at the point before sector entry where tracking begins).
        # therefore, there's no previous step information to assess
        # its expeditious behaviour. it can be assessed from the next step.
        reward = 0.0

    else:
        curr_dist = ac_tracked_state.track_dist_to_exit_cr
        prev_dist = prev_ac_tracked_state.track_dist_to_exit_cr
        dist_diff = curr_dist - prev_dist

        reward = np.clip(dist_diff, -DIFF_THRESHOLD, DIFF_THRESHOLD)

        # scale the reward based on the aircraft's speed. (i.e., it is
        # more impressive for an aircraft with a slower speed to travel
        # faster towards its sector exit, than a faster aircraft).
        #
        # NOTE, true airspeed (tas) is currently used. should calibrated
        # airspeed (cas) be used instead? if cas is used, consider
        # including a flight level scaler too because aircraft at the same
        # cas fly differently on different flight levels (the higher the
        # flight level, the faster the aircraft).

        speed_scale = 1.0 - (aircraft.speed_tas / MAX_SPEED_TAS)
        # ensure that the scale is not completelly 0.0 for aircraft
        # flying at the max speed.
        speed_scale = max(speed_scale, 0.1)

        reward = reward * speed_scale

    return float(reward)


def expeditious_quad(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward or penalize for being navigating efficiently or inefficiently.

    Reward is computed using a quadratic function.

    For a given aircraft, if the difference between the current and previous
    distance to its exit is negative (i.e., the aircraft is now closer to its
    exit than before), then a positive reward is computed with a maximum
    value of 1.5 (the larger the difference the closer the reward is to 1.5).
    Otherwise, a negative reward is computed with a minimum value of
    -1.5 (the larger the difference, the closer the reward is to -1.5) .

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: -1.5 to 1.5).
    """

    simulator_env = gym_env.get_simulator_env()
    aircraft = simulator_env.aircraft[callsign]

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    prev_ac_tracked_state = gym_env.get_tracked_aircraft_data_previous(callsign)

    if prev_ac_tracked_state is None:
        # aircraft tracking just started (either was just spawned or
        # arrived at the point before sector entry where tracking begins).
        # therefore, there's no previous step information to assess
        # its expeditious behaviour. it can be assessed from the next step.
        reward = 0.0

    else:
        curr_dist = ac_tracked_state.track_dist_to_exit_cr
        prev_dist = prev_ac_tracked_state.track_dist_to_exit_cr
        dist_diff = curr_dist - prev_dist

        reward = np.clip(dist_diff, -DIFF_THRESHOLD, DIFF_THRESHOLD)
        sign = np.sign(reward)
        reward = 1.5 * (reward**2)

        # scale the reward based on the aircraft's speed. (i.e., it is
        # more impressive for an aircraft with a slower speed to travel
        # faster towards its sector exit, than a faster aircraft).
        #
        # NOTE, true airspeed (tas) is currently used. should calibrated
        # airspeed (cas) be used instead? if cas is used, consider
        # including a flight level scaler too because aircraft at the same
        # cas fly differently on different flight levels (the higher the
        # flight level, the faster the aircraft).

        speed_scale = 1.0 - (aircraft.speed_tas / MAX_SPEED_TAS)
        # ensure that the scale is not completelly 0.0 for aircraft
        # flying at the max speed.
        speed_scale = max(speed_scale, 0.1)

        reward = sign * reward * speed_scale

    return float(reward)


def expeditious_exp(gym_env: BaseEnv, callsign: str, action: int, **kwargs) -> float:  # noqa: ARG001, ANN003
    """Reward or penalize for being navigating efficiently or inefficiently.

    Reward is computed using an exponeniated function or a constant value of
    0 (for a penalty).

    For a given aircraft, if the difference between the current and previous
    distance to its exit is negative (i.e., the aircraft is now closer to its
    exit than before), then an exponential reward is computed having a maximum
    value of 1.0 (the bigger the difference the closer the reward to 1.0).
    Otherwise, a negative linear reward is computed.

    Args:
        gym_env: the gymnasium environment.
        callsign: identifier of the aircraft in the simulation.
        action: action taken by the agent.

    Returns:
        float, the computed reward (range: 0.0 to 1.0).
    """

    simulator_env = gym_env.get_simulator_env()
    aircraft = simulator_env.aircraft[callsign]

    ac_tracked_state = gym_env.get_tracked_aircraft_data(callsign)
    prev_ac_tracked_state = gym_env.get_tracked_aircraft_data_previous(callsign)

    if prev_ac_tracked_state is None:
        # aircraft tracking just started (either was just spawned or
        # arrived at the point before sector entry where tracking begins).
        # therefore, there's no previous step information to assess
        # its expeditious behaviour. it can be assessed from the next step.
        reward = 0.0

    else:
        curr_dist = ac_tracked_state.track_dist_to_exit_cr
        prev_dist = prev_ac_tracked_state.track_dist_to_exit_cr
        dist_diff = curr_dist - prev_dist

        if dist_diff < 0:
            # clip difference to a minimum of -1.0
            dist_diff = max(dist_diff, -DIFF_THRESHOLD)

            # the closer `dist_diff` is to -1.0, the higher the reward
            reward = np.exp(dist_diff + DIFF_THRESHOLD)

            # scale the reward based on the aircraft's speed. (i.e., it is
            # more impressive for an aircraft with a slower speed to travel
            # faster towards its sector exit, than a faster aircraft).
            #
            # NOTE, true airspeed (tas) is currently used. should calibrated
            # airspeed (cas) be used instead? if cas is used, consider
            # including a flight level scaler too because aircraft at the same
            # cas fly differently on different flight levels (the higher the
            # flight level, the faster the aircraft).

            speed_scale = 1.0 - (aircraft.speed_tas / MAX_SPEED_TAS)
            # ensure that the scale is not completelly 0.0 for aircraft
            # flying at the max speed.
            speed_scale = max(speed_scale, 0.1)

            reward = reward * speed_scale

        else:
            reward = 0.0

    return float(reward)
